import cv2, time

INDEX = 1
BACKEND = cv2.CAP_AVFOUNDATION

cap = cv2.VideoCapture(INDEX, BACKEND)
if not cap.isOpened():
    raise SystemExit("カメラを開けません")

cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
cap.set(cv2.CAP_PROP_FPS,30)

t0=time.time(); n=0
while True:
    ok, frame = cap.read()
    if not ok: break
    n+=1; fps=n/(time.time()-t0)
    cv2.putText(frame, f"index:{INDEX}  FPS:{fps:4.1f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.imshow("Preview (ESC to quit)", frame)
    if cv2.waitKey(1)&0xFF==27: break

cap.release(); cv2.destroyAllWindows()
