import cv2
import pygame
import time
import threading
from flask import Flask, Response


# Flask server
app = Flask(__name__)

# Alarm play code
alarm_active = False
alarm_start_time = None


def play_alarm():
    pygame.mixer.init()
    pygame.mixer.music.load("alert2.mp3")

    while alarm_active:
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():  # Așteaptă până se termină redarea sunetului
            pygame.time.Clock().tick(10)


def generate_frames():
    vid = cv2.VideoCapture(0)
    vid.set(3, 700)
    vid.set(4, 700)

    min_area = 3000  # 50
    thresh_val = 25  # 25
    blur_val = (13, 13)  # (13, 13)

    firstFrame = None
    last_update_time = time.time()  # Timpul ultimei actualizări a firstFrame
    update_interval = 5  # Actualizează firstFrame la fiecare 5 secunde

    while (True):
        global alarm_active, alarm_start_time

        _, frame = vid.read()
        presence_detected = False

        # gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, blur_val, 0)

        # Actualizează firstFrame periodic
        current_time = time.time()
        if firstFrame is None or (current_time - last_update_time) > update_interval:
            firstFrame = gray
            last_update_time = current_time  # Resetează timpul ultimei actualizări



        # frames absolute difference
        frameDif = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDif, thresh_val, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)


        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # APROX_SIMPLE da conturul in 4 pct

        # loop over the contours
        for contour in contours:
            if cv2.contourArea(contour) >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                presence_detected = True  # S-a detectat prezenta

        # Pornește alarma dacă detectează prezență și alarma nu e activă
        if presence_detected and not alarm_active:
            alarm_active = True
            alarm_start_time = time.time()
            threading.Thread(target=play_alarm).start()  # Pornește alarma pe un thread separat


        # Verifică dacă a expirat timpul alocat pentru alarmă
        if alarm_active and (time.time() - alarm_start_time) > 5:
            alarm_active = False  # Resetează alarma


        # Encodează frame-ul pentru streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n' 
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Endpoint pentru flux video
@app.route('/')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Rulează serverul Flask
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80, debug=False)


