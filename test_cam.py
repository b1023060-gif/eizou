import cv2

print("[DEBUG] test_cam.py start", flush=True)

cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)  # index=1がスマホ
if not cap.isOpened():
    raise SystemExit("[ERROR] カメラを開けません")

print("[INFO] カメラ起動成功")

while True:
    ok, frame = cap.read()
    if not ok:
        print("[WARN] フレーム取得失敗")
        break

    cv2.imshow("Camera Test", frame)

    # ESCキーで終了
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
