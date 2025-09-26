import cv2, numpy as np

INDEX = 1  # ← check_camera.py でスマホカメラがOKだった番号
BACKEND = cv2.CAP_AVFOUNDATION
FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def robotize(face):
    """顔をロボットっぽく加工"""
    # グレー化 → 金属感
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    steel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    steel = cv2.applyColorMap(steel, cv2.COLORMAP_BONE)  # 金属トーン

    # 輪郭を強調（ワイヤーフレームっぽく）
    edges = cv2.Canny(gray, 80, 150)
    steel[edges > 0] = (200, 200, 255)  # 青白い光を追加

    # 目の部分を赤く発光（ざっくり目の高さの1/3〜1/2）
    h, w = gray.shape
    eye_y1, eye_y2 = h//3, h//2
    eye_region = steel[eye_y1:eye_y2, :]
    mask = cv2.threshold(cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY), 50, 255, cv2.THRESH_BINARY)[1]
    eye_region[mask > 0] = (0, 0, 255)  # 赤発光
    steel[eye_y1:eye_y2, :] = eye_region

    return steel

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
        robo = robotize(roi)
        frame[y:y+h, x:x+w] = robo
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)

    # 小窓プレビュー
    small = cv2.resize(frame, (320, 240))
    cv2.imshow("phone-cam preview", small)

    # メイン表示
    cv2.imshow("phone-cam main", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == 27: break  # ESCで終了

cap.release()
cv2.destroyAllWindows()
