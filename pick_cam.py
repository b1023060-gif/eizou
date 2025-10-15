# pick_cam.py
import cv2

BACKEND = cv2.CAP_AVFOUNDATION
MAX_INDEX = 6  # 調べたいカメラインデックスの範囲

print("[INFO] カメラを順番にチェックします (0〜5) ...")

for i in range(MAX_INDEX):
    cap = cv2.VideoCapture(i, BACKEND)
    if not cap.isOpened():
        print(f"[TRY] index {i}: 開けません")
        continue

    ok, frame = cap.read()
    if not ok:
        print(f"[TRY] index {i}: フレーム取得失敗")
        cap.release()
        continue

    # 画面にインデックス番号を描画
    cv2.putText(frame, f"INDEX {i}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    print(f"[TRY] index {i}: OK → 映像を2秒表示します")

    # プレビュー表示（2秒）
    cv2.imshow("Preview", frame)
    if cv2.waitKey(2000) & 0xFF == 27:  # ESCを押すと途中終了
        cap.release()
        break

    cap.release()

cv2.destroyAllWindows()
print("[END] チェック終了")
