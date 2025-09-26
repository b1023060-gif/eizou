# phone_cam_robot.py
import cv2, time, numpy as np

print("[INFO] phone_cam_robot.py start", flush=True)

# ===== カメラ設定（Mac） =====
# まず index=1（Camo想定）を試し、ダメなら index=0 を試す
BACKEND = cv2.CAP_AVFOUNDATION
def open_cam():
    for idx in (1, 0):
        cap = cv2.VideoCapture(idx, BACKEND)
        if cap.isOpened():
            print(f"[INFO] camera opened: index={idx}", flush=True)
            return cap
    raise SystemExit("[ERROR] カメラを開けません（Camoが起動中か確認）")

cap = open_cam()

# ===== 顔検出器 =====
FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ===== ロボ加工（軽量） =====
def robotize(face_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    steel = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    steel = cv2.applyColorMap(steel, cv2.COLORMAP_BONE)  # 金属トーン
    edges = cv2.Canny(g, 80, 160)
    steel[edges > 0] = (200, 200, 255)                    # 青白い輪郭光

    # 目あたりを赤く（ざっくり上1/3〜1/2帯）
    h, w = g.shape
    y1, y2 = h//3, h//2
    band = steel[y1:y2]
    mask = cv2.threshold(cv2.cvtColor(band, cv2.COLOR_BGR2GRAY), 60, 255, cv2.THRESH_BINARY)[1]
    band[mask > 0] = (0, 0, 255)
    steel[y1:y2] = band
    return steel

# ===== 段階的ロボ化（人→ロボ→人） =====
RISE_SEC  = 1.2     # ロボ度を上げる時間
HOLD_SEC  = 0.8     # ちょうど気持ち悪い辺りで静止
RESET_SEC = 0.8     # 元に戻す時間
TARGET    = 0.65    # 不気味停止のしきい値

phase = "idle"      # idle -> rise -> hold -> fall
t_phase = time.time()
alpha  = 0.0        # 0=素顔, 1=完全ロボ

while True:
    ok, frame = cap.read()
    if not ok:
        print("[WARN] フレーム取得失敗", flush=True)
        break

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE.detectMultiScale(gray, 1.2, 5, minSize=(120,120))

    # 状態遷移
    now = time.time()
    if phase == "idle" and len(faces) > 0:
        phase, t_phase = "rise", now
    elif phase == "rise":
        alpha = min(1.0, (now - t_phase)/RISE_SEC)
        if alpha >= TARGET:
            phase, t_phase = "hold", now
    elif phase == "hold":
        alpha = TARGET
        if now - t_phase >= HOLD_SEC:
            phase, t_phase = "fall", now
    elif phase == "fall":
        alpha = max(0.0, 1 - (now - t_phase)/RESET_SEC*(1/(1 - TARGET)))
        if alpha <= 0.01:
            alpha, phase = 0.0, "idle"

    # 顔をロボ化（代表1人分）
    for (x, y, w, h) in faces[:1]:
        roi   = frame[y:y+h, x:x+w]
        robo  = robotize(roi)
        blend = cv2.addWeighted(roi, 1-alpha, robo, alpha, 0)
        frame[y:y+h, x:x+w] = blend
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,255), 2)
        break

    # 画面情報
    cv2.putText(frame, f"phase:{phase}  uncanny:{alpha:.2f}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,200), 2)

    # メイン表示
    cv2.imshow("phone-cam main (ESC to quit)", frame)

    # 小窓プレビュー（320x240）
    preview = cv2.resize(frame, (320, 240))
    cv2.imshow("phone-cam preview", preview)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # ESC
        break
    elif k == ord('r'):  # リセット
        phase, alpha = "idle", 0.0

cap.release()
cv2.destroyAllWindows()
