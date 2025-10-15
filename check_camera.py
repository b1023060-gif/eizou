import cv2, sys

print("[START] check_camera.py", flush=True)

for i in (0, 1):
    print(f"[TRY] index {i}", flush=True)
    cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
    if cap.isOpened():
        ok, frame = cap.read()
        cap.release()
        print("  ->", "OK" if ok else "NO FRAME", flush=True)
    else:
        print("  -> NG", flush=True)

print("[END]", flush=True)
