# app.py
from flask import Flask, Response, render_template
import cv2
import numpy as np

app = Flask(__name__)
cap = cv2.VideoCapture(0)

# ================= Mango Detection (ย่อจากของคุณ) =================
def detect_mango(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    unripe = cv2.inRange(hsv, (35, 70, 70), (85, 255, 255))
    ripe   = cv2.inRange(hsv, (5, 70, 70), (45, 255, 255))

    mask = cv2.bitwise_or(unripe, ripe)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, "Mango", (x,y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    return frame
# ===============================================================

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = detect_mango(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.run(
    host="0.0.0.0",
    port=8000,
    ssl_context=("cert/cert.pem", "cert/key.pem"),
    debug=False
)
