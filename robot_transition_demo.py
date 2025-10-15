# robot_transition_demo.py
import cv2, time, numpy as np

print("[INFO] robot_transition_demo.py start", flush=True)

INDEX = 0 # スマホカメラ（Camo）
BACKEND = cv2.CAP_AVFOUNDATION
FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

RISE_SEC, HOLD_SEC, RESET_SEC = 2.0, 1.5, 1.5
TARGET = 0.7

def robotize(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    steel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    steel = cv2.applyColorMap(steel, cv2.COLORMAP_BONE)
    edges = cv2.Canny(gray, 80, 150)
    steel[edges > 0] = (200,200,255)
    h, w = gray.shape
    eye_y1, eye_y2 = h//3, h//2
    eye_region = steel[eye_y1:eye_y2, :]
    mask = cv2.threshold(cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY), 50, 255, cv2.THRESH_BINARY)[1]
    eye_region[mask > 0] = (0,0,255)
    steel[eye_y1:eye_y2, :] = eye_region
    return steel

cap = cv2.VideoCapture(INDEX, BACKEND)
if not cap.isOpened():
    raise SystemExit("[ERROR] スマホカメラを開けませんでした")

phase = "idle"
t_phase = time.time()
alpha = 0.0

while True:
    ok, frame = cap.read()
    if not ok: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE.detectMultiScale(gray, 1.2, 5, minSize=(120,120))

    now = time.time()
    if phase == "idle" and len(faces) > 0:
        phase, t_phase = "rise", now
    elif phase == "rise":
        alpha = min(1.0, (now - t_phase)/RISE_SEC)
        if alpha >= TARGET: phase, t_phase = "hold", now
    elif phase == "hold":
        alpha = TARGET
        if now - t_phase >= HOLD_SEC: phase, t_phase = "fall", now
    elif phase == "fall":
        alpha = max(0.0, 1 - (now - t_phase)/RESET_SEC*(1/(1 - TARGET)))
        if alpha <= 0.01: alpha, phase = 0.0, "idle"

    for (x,y,w,h) in faces[:1]:
        roi = frame[y:y+h, x:x+w]
        robo = robotize(roi)
        blend = cv2.addWeighted(roi, 1-alpha, robo, alpha, 0)
        frame[y:y+h, x:x+w] = blend
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

    cv2.putText(frame, f"{phase} alpha={alpha:.2f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,200), 2)

    cv2.imshow("robot transition demo (ESC=終了, R=リセット)", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == 27: break
    elif k == ord('r'): phase, alpha = "idle", 0.0

cap.release()
cv2.destroyAllWindows()
