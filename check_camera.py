import cv2, sys

print("[START] check_camera.py", flush=True)

# Macは AVFoundation が安定
BACKEND = cv2.CAP_AVFOUNDATION

for i in range(0, 4):  # 0〜3を走査
    print(f"[TRY] index {i}", flush=True)
    cap = cv2.VideoCapture(i, BACKEND)
    if cap.isOpened():
        ok, _ = cap.read()
        cap.release()
        print("  ->", "OK" if ok else "NO FRAME", flush=True)
    else:
        print("  -> NG", flush=True)

print("[END]", flush=True)
