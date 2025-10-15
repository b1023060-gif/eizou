# phone_cam_robot.py
import cv2, time, numpy as np

INDEX  = 0  # ← スマホ( Camo ) が index=0
BACKEND = cv2.CAP_AVFOUNDATION
FACE   = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def robotize(face_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    steel = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    steel = cv2.applyColorMap(steel, cv2.COLORMAP_BONE)   # 金属トーン
    edges = cv2.Canny(g, 80, 160)
    steel[edges > 0] = (200, 200, 255)                    # 青白い輪郭光
    h, w = g.shape
    y1, y2 = h//3, h//2                                   # 目の帯を赤化
    band = steel[y1:y2].copy()
    mask = cv2.threshold(cv2.cvtColor(band, cv2.COLOR_BGR2GRAY), 60, 255, cv2.THRESH_BINARY)[1]
    band[mask > 0] = (0, 0, 255)
    steel[y1:y2] = band
    return steel

RISE_SEC, HOLD_SEC, RESET_SEC, TARGET = 1.2, 0.8, 0.8, 0.65
phase, t_phase, alpha = "idle", time.time(), 0.0

cap = cv2.VideoCapture(INDEX, BACKEND)
if not cap.isOpened():
    raise SystemExit("[ERROR] Camo(スマホ)を開けません")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
cap.set(cv2.CAP_PROP_FPS, 30)

print("[INFO] phone_cam_robot start (ESCで終了, rでリセット)")
while True:
    ok, frame = cap.read()
    if not ok: break

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE.detectMultiScale(gray, 1.2, 5, minSize=(120,120))

    now = time.time()
    if phase == "idle" and len(faces) > 0:
        phase, t_phase = "rise", now
    elif phase == "rise":
        alpha = min(1.0, (now - t_phase)/RISE_SEC)
        if alpha >= TARGET: phase, t_phase = "hold", now
    elif phase == "hold":
        alpha = TARGET
        if now - t_phase >= HOLD_SEC:
            phase, t_phase = "fall", now
    elif phase == "fall":
        alpha = max(0.0, 1 - (now - t_phase)/RESET_SEC*(1/(1 - TARGET)))
        if alpha <= 0.01: alpha, phase = 0.0, "idle"

    for (x, y, w, h) in faces[:1]:
        roi   = frame[y:y+h, x:x+w]
        robo  = robotize(roi)
        blend = cv2.addWeighted(roi, 1-alpha, robo, alpha, 0)
        frame[y:y+h, x:x+w] = blend
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,255), 2)

    cv2.putText(frame, f"phase:{phase} uncanny:{alpha:.2f}", (10,28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,200), 2)

    cv2.imshow("phone-cam main", frame)
    cv2.imshow("phone-cam preview", cv2.resize(frame, (320,240)))

    k = cv2.waitKey(1) & 0xFF
    if k == 27: break
    if k == ord('r'): phase, alpha = "idle", 0.0

cap.release()
cv2.destroyAllWindows()
