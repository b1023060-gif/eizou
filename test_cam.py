# test_cam.py
import cv2

INDEX = 0  # ← スマホ( Camo ) が index=0 と判明している
BACKEND = cv2.CAP_AVFOUNDATION

cap = cv2.VideoCapture(INDEX, BACKEND)
if not cap.isOpened():
    raise SystemExit("[ERROR] カメラを開けません (Camoが映っているか/Camoアプリ起動を確認)")

# 好みで解像度固定（Camo側プロファイルと合わせると安定）
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

print("[INFO] Camo(スマホ) でプレビュー開始。ESCで終了")
while True:
    ok, frame = cap.read()
    if not ok:
        print("[WARN] フレーム取得失敗")
        break
    cv2.imshow("Camo phone (index=0)", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
