from flask import Flask, Response, render_template_string, request, redirect, url_for
import cv2
import os
import time
from picamera2 import Picamera2
from gpiozero import AngularServo

app = Flask(__name__)

# Initialize servo
servo = AngularServo(12, initial_angle=90, min_pulse_width=0.5 / 1000, max_pulse_width=2.5 / 1000)

# Load object detection files
user = os.getlogin()
base_path = f"/home/{user}/Desktop/Object_Detection_Files"

with open(os.path.join(base_path, "coco.names"), "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

net = cv2.dnn_DetectionModel(
    os.path.join(base_path, "frozen_inference_graph.pb"),
    os.path.join(base_path, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Camera setup
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

# App state
feeding_time = "23:59:59"
servo_activated = False
detection_log = []
last_reset_date = time.strftime("%Y-%m-%d")

def activate_servo():
    global servo_activated
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    servo.angle = -90
    time.sleep(1)
    servo.angle = 90
    servo_activated = True
    detection_log.append(f"Food dispensed at {timestamp}")

def log_detection():
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    detection_log.append(f"Dog detected at {timestamp}")

def detect_objects(img, threshold=0.45, nms=0.2, draw=True, target_classes=None):
    if target_classes is None:
        target_classes = classNames
    class_ids, confidences, boxes = net.detect(img, confThreshold=threshold, nmsThreshold=nms)
    if len(class_ids) != 0:
        for class_id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), boxes):
            class_name = classNames[class_id - 1]
            if class_name in target_classes:
                log_detection()
                if draw:
                    cv2.rectangle(img, box, (0, 255, 0), 2)
                    cv2.putText(img, class_name.upper(), (box[0]+10, box[1]+30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0]+200, box[1]+30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                return True
    return False

def daily_reset_check():
    global last_reset_date, servo_activated, detection_log
    today = time.strftime("%Y-%m-%d")
    if today != last_reset_date:
        servo_activated = False
        detection_log.clear()
        last_reset_date = today

def gen_frames():
    global servo_activated
    global actual_time
    while True:
        daily_reset_check()

        actual_time = time.strftime("%H:%M:%S")
        img = picam2.capture_array("main")
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        if actual_time > feeding_time:
            if detect_objects(img, target_classes=['dog']):
                if not servo_activated:
                    activate_servo()

        _, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/', methods=['GET', 'POST'])
def index():
    global feeding_time
    if request.method == 'POST':
        new_alarm = request.form.get('feeding_time')
        if new_alarm:
            feeding_time = new_alarm.strip()
        return redirect(url_for('index'))

    return render_template_string("""
        <html>
        <head><title>PetPal</title></head>
        <body>
        <h1>PetPal!</h1>
        <img src="{{ url_for('video_feed') }}" width="640" height="480"><br><br>
        <h2>Current Feeding Time: {{ feeding_time }}</h2>
        <form method="POST">
            <label>Set Feeding Time (HH:MM:SS):</label>
            <input type="text" name="feeding_time" required>
            <input type="submit" value="Update Alarm">
        </form>

        <br>
        <form method="POST" action="{{ url_for('manual_trigger') }}">
            <button type="submit">Feed Now</button>
        </form>

        <form method="POST" action="{{ url_for('reset') }}">
            <button type="submit">Reset </button>
        </form>
                                  
        <form method="POST" action="{{ url_for('shutdown') }}">
            <button type="submit">Kill App</button>
        </form>

        <h3>Log</h3>
        <ul>
        {% for entry in log %}
            <li>{{ entry }}</li>
        {% else %}
            <li>No detections yet.</li>
        {% endfor %}
        </ul>
        </body>
        </html>
    """, feeding_time=feeding_time, log=detection_log)

@app.route('/manual_trigger', methods=['POST'])
def manual_trigger():
    activate_servo()
    return redirect(url_for('index'))

@app.route('/reset', methods=['POST'])
def reset():
    global servo_activated, detection_log
    servo_activated = False
    detection_log.clear()
    return redirect(url_for('index'))

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/shutdown', methods=['POST'])
def shutdown():
    global picam2
    picam2.stop()
    os._exit(0) 

if __name__ == "__main__":
    try:
        app.run(host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        picam2.stop()

