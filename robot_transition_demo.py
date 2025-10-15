print("[INFO] robot_transition_demo.py start", flush=True)

import cv2, time, numpy as np

# ====== カメラ設定 ======
INDEX = 1  # ← check_camera.py でスマホカメラがOKだった番号
BACKEND = cv2.CAP_AVFOUNDATION
FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ====== パラメータ ======
RISE_SEC = 2.0     # 人→ロボに変化する時間
HOLD_SEC = 1.5     # 不気味度MAXで停止する時間
RESET_SEC = 1.5    # ロボ→人に戻る時間
TARGET   = 0.7     # 不気味度のしきい値(0〜1)

# ====== 加工関数 ======
def robotize(face):
    """顔をロボットっぽく加工"""
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    steel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    steel = cv2.applyColorMap(steel, cv2.COLORMAP_BONE)

    # 輪郭を強調
    edges = cv2.Canny(gray, 80, 150)
    steel[edges > 0] = (200, 200, 255)

    # 目のあたりを赤く発光
    h, w = gray.shape
    eye_y1, eye_y2 = h//3, h//2
    eye_region = steel[eye_y1:eye_y2, :]
    mask = cv2.threshold(cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY), 50, 255, cv2.THRESH_BINARY)[1]
    eye_region[mask > 0] = (0, 0, 255)
    steel[eye_y1:eye_y2, :] = eye_region

    return steel

# ====== カメラ起動 ======
cap = cv2.VideoCapture(INDEX, BACKEND)
if not cap.isOpened():
    raise SystemExit("スマホカメラを開けませんでした")

phase = "idle"     # idle → rise → hold → fall
t_phase = time.time()
alpha = 0.0        # 0=人間, 1=完全ロボ

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE.detectMultiScale(gray, 1.2, 5, minSize=(120,120))

    # ===== 状態遷移 =====
    now = time.time()
    if phase == "idle" and len(faces) > 0:
        phase = "rise"; t_phase = now
    elif phase == "rise":
        alpha = min(1.0, (now - t_phase)/RISE_SEC)
        if alpha >= TARGET: phase = "hold"; t_phase = now
    elif phase == "hold":
        alpha = TARGET
        if now - t_phase >= HOLD_SEC:
            phase = "fall"; t_phase = now
    elif phase == "fall":
        alpha = max(0.0, 1 - (now - t_phase)/RESET_SEC*(1/(1-TARGET)))
        if alpha <= 0.01: phase = "idle"; alpha = 0.0

   
