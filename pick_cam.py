# pick_cam.py
import cv2, time

BACKEND = cv2.CAP_AVFOUNDATION
for i in range(6):
    cap = cv2.VideoCapture(i, BACKEND)
    print(f"[TRY] index {i} ->", "OPEN" if cap.isOpened() else "NG")
    if not cap.isOpened(): 
        continue
    ok, frame = cap.read()
    if ok:
        cv2.putText(frame, f"INDEX {i}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)
        cv2.imshow("Preview (2s each / ESC=quit)", frame)
        if cv2.waitKey(2000) & 0xFF == 27:
            break
    cap.release()
cv2.destroyAllWindows()
