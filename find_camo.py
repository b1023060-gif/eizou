import cv2

for i in (0,1):
    print(f"[TRY] index {i}")
    cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
    if cap.isOpened():
        ok, frame = cap.read()
        print("  ->", "OK" if ok else "NO FRAME")
        cap.release()
    else:
        print("  -> NG")

