import cv2, numpy as np

INDEX = 1  # ← スマホカメラが check_camera.py で OK だった番号
BACKEND = cv2.CAP_AVFOUNDATION
FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def warp_face(face):
    """顔をちょっと変形（例: 波打たせる）"""
    h, w = face.shape[:2]
    # 座標マップを作る
    map_y, map_x = np.indices((h, w), dtype=np.float32)
    # サイン波で横方向にずらす
    map_x = map_x + 10 * np.sin(map_y / 20.0)
    # リマップして変形
    warped = cv2.remap(face, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped

cap = cv2.VideoCapture(INDEX, BACKEND)
if not cap.isOpened():
    raise SystemExit("スマホカメラを開けませんでした")

while True:
    ok, frame = cap.read()
    if not ok: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE.detectMultiScale(gray, 1.2, 5, minSize=(100,100))

    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]
        warped = warp_face(roi)
        frame[y:y+h, x:x+w] = warped
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)

    # スマホ映像の小窓プレビュー
    small = cv2.resize(frame, (320, 240))
    cv2.imshow("phone-cam preview", small)

    # メイン表示（必要なら大きく表示）
    cv2.imshow("phone-cam main", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == 27: break  # ESCで終了

cap.release()
cv2.destroyAllWindows()
