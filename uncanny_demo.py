import cv2, time, os, datetime
import numpy as np

# ====== 設定 ======
INDEX = 1
BACKEND = cv2.CAP_AVFOUNDATION
FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# パラメータ
RISE_SEC = 1.2
HOLD_SEC = 0.8
RESET_SEC = 0.8
TARGET = 0.65

os.makedirs("debug", exist_ok=True)

def metallic(face):
    g = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(g, 60, 140)
    edges = cv2.GaussianBlur(edges, (3,3), 0)
    steel = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    steel = cv2.applyColorMap(steel, cv2.COLORMAP_BONE)
    steel[edges > 0] = (200,200,255)
    return steel

cap = cv2.VideoCapture(INDEX, BACKEND)
if not cap.isOpened():
    raise SystemExit("カメラ開けない")

phase = "idle"
t_phase = time.time()
alpha = 0.0

while True:
    ok, frame = cap.read()
    if not ok: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE.detectMultiScale(gray, 1.2, 5, minSize=(120,120))

    # --- 状態遷移 ---
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

    # --- 顔の加工と表示 ---
    for (x,y,w,h) in faces[:1]:
        roi = frame[y:y+h, x:x+w]
        roi_orig = roi.copy()
        robo = metallic(roi)
        blend = cv2.addWeighted(roi, 1-alpha, robo, alpha, 0)

        diff = cv2.absdiff(blend, roi_orig)
        mad = float(np.mean(diff))
        psnr = cv2.PSNR(blend, roi_orig)

        frame[y:y+h, x:x+w] = blend

        # メイン画面に情報表示
        col = (0,255,0) if alpha < 0.01 else (255,0,255)
        cv2.rectangle(frame, (x,y), (x+w,y+h), col, 2)
        cv2.putText(frame, f"phase:{phase}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"uncanny={alpha:.2f}", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame, f"MAD={mad:.1f}  PSNR={psnr:.1f}dB", (10,90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)

    # --- ウィンドウに表示 ---
    cv2.imshow("Uncanny Demo (ESC=終了, S=保存)", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:   # ESCで終了
        break
    elif k == ord('s'):  # スナップ保存
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(f"debug/{ts}_frame.png", frame)

cap.release()
cv2.destroyAllWindows()
