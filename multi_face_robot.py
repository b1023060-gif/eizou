# morph_robot_face_rt.py
# リアルタイム 人間→ロボ顔 モーフ（修正版：白線対策＋安定化）
# - FaceMesh 468点
# - 三角ワープ + 色合わせ + 簡易リライティング（金属感）
# - Poissonブレンドは安全ラッパで実行（前処理で縁のハロ抑制）
# - “無意味さ”演出：走査線OFF（白線の原因になるため）
# 実行: python -u morph_robot_face_rt.py

import cv2, numpy as np, mediapipe as mp, time
from scipy.spatial import Delaunay

# ========= OpenCV最適化 =========
cv2.setUseOptimized(True)
cv2.setNumThreads(0)

# ========= 設定 =========
CAM_INDEX       = 0
W, H            = 1280, 720
MAX_FACES       = 1
TEMPLATE_PATH   = "robot.png"

# モーフ進行
RISE_SEC        = 8.0      # 0→1まで
START_LOCK      = 0.05     # 初期安定待ち秒

# マスク/合成（境界やわらかめ）
ALPHA_STRENGTH  = 1.0
MASK_DILATE     = 8
MASK_FEATHER    = 10       # ← 6→10（白縁抑制）

# 金属ライティング
NORMAL_SCALE    = 4.0
SPEC_POWER      = 96
SPEC_GAIN       = 0.9
DIFF_GAIN       = 0.15

# “無意味さ”演出（走査線はOFF）
GLITCH_TRI_JITTER_PX = 2.0
GLITCH_CHROM_AB      = 1.5
GLITCH_SCAN_ALPHA    = 0.0  # ← 0で完全OFF（白線の元凶）

# ========= MediaPipe =========
mp_face_mesh = mp.solutions.face_mesh
mesh_live   = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=MAX_FACES,
                                    refine_landmarks=True, min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)
mesh_static = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                    refine_landmarks=True, min_detection_confidence=0.5)

# ========= Hair / Person Segmentation =========
mp_selfie = mp.solutions.selfie_segmentation
seg = mp_selfie.SelfieSegmentation(model_selection=1)

# ========= ユーティリティ =========
def to_xy(landmarks, img):
    h,w = img.shape[:2]
    pts = []
    for lm in landmarks:
        x = int(np.clip(lm.x,0,1)*w)
        y = int(np.clip(lm.y,0,1)*h)
        pts.append((x,y))
    return np.array(pts, np.int32)

def detect_landmarks(img_bgr, static=False):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = (mesh_static if static else mesh_live).process(rgb)
    if not res.multi_face_landmarks: return None
    return to_xy(res.multi_face_landmarks[0].landmark, img_bgr)

def mask_from_pts(shape, pts, dilate=0, blur=0):
    h,w = shape[:2]
    m = np.zeros((h,w), np.uint8)
    if pts is not None and len(pts)>=3:
        cv2.fillPoly(m, [pts.astype(np.int32)], 255)
    if dilate>0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate+1, 2*dilate+1))
        m = cv2.dilate(m, k, 1)
    if blur>0:
        m = cv2.GaussianBlur(m, (0,0), blur)
    return m

def distance_field(mask):
    dst = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    if dst.max()>0: dst = dst/(dst.max()+1e-6)
    dst = cv2.GaussianBlur(dst,(0,0),7)
    return dst

def lighting_from_normal(mask, normal_scale, spec_pow, spec_gain, diff_gain):
    dist = distance_field(mask)
    gx = cv2.Sobel(dist, cv2.CV_32F,1,0,ksize=3)
    gy = cv2.Sobel(dist, cv2.CV_32F,0,1,ksize=3)
    nx = -gx*normal_scale; ny=-gy*normal_scale; nz=np.ones_like(nx)
    nrm = np.sqrt(nx*nx+ny*ny+nz*nz) + 1e-6
    nx/=nrm; ny/=nrm; nz/=nrm
    L = np.array([0.0,-0.25,1.0], np.float32); L /= (np.linalg.norm(L)+1e-6)
    Hvec = (L + np.array([0,0,1],np.float32)); Hvec/= (np.linalg.norm(Hvec)+1e-6)
    ndotl = np.clip(nx*L[0]+ny*L[1]+nz*L[2], 0, 1)
    ndoth = np.clip(nx*Hvec[0]+ny*Hvec[1]+nz*Hvec[2], 0, 1)
    spec = (ndoth**spec_pow) * 255.0 * spec_gain
    diff = (ndotl*diff_gain) * 255.0
    spec3 = np.dstack([spec]*3).astype(np.float32)
    diff3 = np.dstack([diff]*3).astype(np.float32)
    return spec3, diff3

def color_match(src_rgb, dst_rgb, mask=None):
    s = src_rgb.astype(np.float32)
    d = dst_rgb.astype(np.float32)
    m = (mask>0).astype(np.float32)[...,None] if mask is not None else np.ones_like(s[..., :1])

    def stats(img):
        n = np.maximum(1.0, m.sum())
        mean = (img * m).sum(axis=(0,1), keepdims=True) / n
        var  = (m * (img-mean)**2).sum(axis=(0,1), keepdims=True) / n
        std  = np.sqrt(np.maximum(1e-6, var))
        return mean, std

    ms, ss = stats(s); md, sd = stats(d)
    out = ((s - ms) / ss) * sd + md
    return np.clip(out, 0, 255).astype(np.uint8)

def smoothstep(x):
    x = np.clip(x,0.0,1.0)
    return x*x*(3-2*x)

def warp_piecewise(src_rgba, src_pts, dst_pts, tri, dst_size, jitter_amp=0.0):
    dw, dh = dst_size
    out = np.zeros((dh, dw, 4), np.uint8)
    for t in tri.simplices:
        s_tri = np.float32([src_pts[t[0]], src_pts[t[1]], src_pts[t[2]]])
        d_tri = np.float32([dst_pts[t[0]], dst_pts[t[1]], dst_pts[t[2]]])
        if jitter_amp>0:
            j = np.random.uniform(-jitter_amp, jitter_amp, d_tri.shape).astype(np.float32)
            d_tri = d_tri + j
        M = cv2.getAffineTransform(s_tri, d_tri)

        x1,y1 = np.min(d_tri, axis=0).astype(int)
        x2,y2 = np.max(d_tri, axis=0).astype(int)
        x1=max(0,x1); y1=max(0,y1); x2=min(dw-1,x2); y2=min(dh-1,y2)
        if x2<=x1 or y2<=y1: continue

        patch = cv2.warpAffine(src_rgba, M, (dw, dh),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_TRANSPARENT)
        tri_mask = np.zeros((dh, dw), np.uint8)
        cv2.fillConvexPoly(tri_mask, d_tri.astype(np.int32), 255)

        tri_mask_roi = tri_mask[y1:y2, x1:x2]
        patch_roi    = patch[y1:y2, x1:x2]
        a = (patch_roi[...,3:4].astype(np.float32)/255.0) * (tri_mask_roi[...,None]/255.0)

        dst_rgb = out[y1:y2, x1:x2, :3].astype(np.float32)
        src_rgb = patch_roi[..., :3].astype(np.float32)
        out[y1:y2, x1:x2, :3] = (src_rgb*a + dst_rgb*(1-a)).astype(np.uint8)

        dst_a = out[y1:y2, x1:x2, 3:4].astype(np.float32)
        out[y1:y2, x1:x2, 3:4] = np.clip(dst_a + a*255.0, 0, 255).astype(np.uint8)
    return out

def chromatic_aberration(img, mask, offset):
    h,w = img.shape[:2]
    out = img.copy().astype(np.float32)
    Mx = np.float32([[1,0, offset],[0,1,0]])
    Nx = np.float32([[1,0,-offset],[0,1,0]])
    r = cv2.warpAffine(img[:,:,2], Mx, (w,h))
    b = cv2.warpAffine(img[:,:,0], Nx, (w,h))
    out[:,:,2] = r; out[:,:,0] = b
    if mask is not None:
        m = (mask.astype(np.float32)/255.0)[...,None]
        out = out*m + img.astype(np.float32)*(1-m)
    return np.clip(out,0,255).astype(np.uint8)

def hair_mask_from_person(frame_bgr, face_mask, shrink_face=6, seg_scale=0.5, thr=0.55):
    """SelfieSegmentationで人マスク→顔マスクを引いて髪領域近傍を作る"""
    h, w = frame_bgr.shape[:2]
    sw, sh = int(w*seg_scale), int(h*seg_scale)
    small = cv2.resize(frame_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    pr = seg.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).segmentation_mask
    pr = cv2.resize(pr, (w, h), interpolation=cv2.INTER_LINEAR)
    person = (pr > thr).astype(np.uint8) * 255

    if shrink_face > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*shrink_face+1, 2*shrink_face+1))
        face_shrunk = cv2.erode(face_mask, k, 1)
    else:
        face_shrunk = face_mask

    hair = cv2.subtract(person, face_shrunk)
    hair = cv2.medianBlur(hair, 5)
    hair = cv2.GaussianBlur(hair, (0,0), 1.2)
    return hair

def safe_seamless_clone(src_bgr, dst_bgr, mask_u8):
    """Poissonブレンド安全ラッパ：縁収縮＆クロップ。失敗時はdst返す。"""
    nz = cv2.findNonZero(mask_u8)
    if nz is None: return dst_bgr

    x,y,w,h = cv2.boundingRect(nz)
    H,W = dst_bgr.shape[:2]
    MARGIN = 2
    x  = max(MARGIN, x); y  = max(MARGIN, y)
    x2 = min(W - MARGIN - 1, x + w - 1)
    y2 = min(H - MARGIN - 1, y + h - 1)
    w  = max(2, x2 - x + 1); h  = max(2, y2 - y + 1)

    src_c  = src_bgr[y:y+h, x:x+w]
    dst_c  = dst_bgr[y:y+h, x:x+w]
    mask_c = mask_u8[y:y+h, x:x+w]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask_c = cv2.erode(mask_c, kernel, iterations=1)
    if cv2.countNonZero(mask_c) < 16:
        return dst_bgr

    try:
        cx, cy = w//2, h//2
        out_c = cv2.seamlessClone(src_c, dst_c, mask_c, (cx, cy), cv2.NORMAL_CLONE)
        out = dst_bgr.copy(); out[y:y+h, x:x+w] = out_c
        return out
    except cv2.error:
        return dst_bgr

# ========= テンプレ準備 =========
robot_rgba = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_UNCHANGED)
if robot_rgba is None:
    raise SystemExit(f"[ERROR] '{TEMPLATE_PATH}' を開けません。PNGを置いてください。")
if robot_rgba.shape[2]==3:
    a = np.full(robot_rgba.shape[:2]+(1,), 255, np.uint8)
    robot_rgba = np.concatenate([robot_rgba, a], axis=2)

tpl_bgr = robot_rgba[...,:3]
tpl_pts = detect_landmarks(tpl_bgr, static=True)
if tpl_pts is None:
    # フォールバック：中央楕円から擬似点
    h0,w0 = tpl_bgr.shape[:2]
    yy, xx = np.mgrid[0:h0, 0:w0]
    oval = (((xx-w0/2)/(0.38*w0))**2 + ((yy-h0*0.52)/(0.48*h0))**2) <= 1.0
    edge = cv2.Canny((oval*255).astype(np.uint8), 50, 100)
    ys, xs = np.where(edge>0)
    sample = np.vstack([xs,ys]).T
    if len(sample)<50:
        raise SystemExit("[ERROR] テンプレからランドマーク取得不可。正面顔のPNGにしてください。")
    idx = np.linspace(0, len(sample)-1, 300).astype(int)
    tpl_pts = sample[idx]
tri_tpl = Delaunay(tpl_pts)
print(f"[INFO] template landmarks={len(tpl_pts)} triangles={len(tri_tpl.simplices)}")

# ========= カメラ =========
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_ANY)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv2.CAP_PROP_FPS, 30)
print("[INFO] start. ESC=終了, r=リセット")

# ========= 進行状態 =========
t_start   = None
alpha     = 0.0
locked    = False
last_seen = 0.0

while True:
    ok, frame = cap.read()
    if not ok: break
    frame = cv2.flip(frame, 1)
    now = time.time()

    # ---- 顔検出・進行管理 ----
    live_pts = detect_landmarks(frame, static=False)
    if live_pts is not None:
        last_seen = now
        if t_start is None:
            t_start = now + START_LOCK
            alpha   = 0.0
            locked  = False
        if not locked and now >= t_start:
            prog  = (now - t_start) / RISE_SEC
            alpha = smoothstep(prog)
            if alpha >= 0.999:
                alpha = 1.0
                locked = True
    else:
        if now - last_seen > 1.5:
            t_start = None; alpha = 0.0; locked = False
        cv2.imshow("morph-robot", frame)
        if (cv2.waitKey(1)&0xFF)==27: break
        continue

    # ---- マスク作成（※ if/else の外。インデントずれ注意）----
    face_mask = mask_from_pts(frame.shape, live_pts, dilate=MASK_DILATE, blur=MASK_FEATHER)

    # “無意味さ”強度（中盤ピーク）※ 走査線は使わない
    mid = 1.0 - abs(alpha*2.0 - 1.0)
    jitter = GLITCH_TRI_JITTER_PX * mid
    cab    = GLITCH_CHROM_AB * mid

    # ---- 三角ワープ ----
    warped_rgba = warp_piecewise(robot_rgba, tpl_pts, live_pts, tri_tpl,
                                 (frame.shape[1], frame.shape[0]),
                                 jitter_amp=jitter)

    # ★ アルファだけを軽くぼかして継ぎ目の白筋を消す
    if warped_rgba.shape[2] >= 4:
        wa = warped_rgba[..., 3]
        wa = cv2.GaussianBlur(wa, (0,0), 1.0)
        warped_rgba[..., 3] = wa

    w_rgb = warped_rgba[...,:3]
    # w_a = warped_rgba[...,3].astype(np.float32)/255.0  # （必要なら使用）

    # ---- 色合わせ（貼った感軽減）----
    live_face = frame.copy()
    live_face[(face_mask==0)] = 0
    w_rgb_matched = color_match(w_rgb, live_face, mask=face_mask)

    # ---- 簡易リライティング（金属）----
    spec3, diff3 = lighting_from_normal(face_mask, NORMAL_SCALE, SPEC_POWER, SPEC_GAIN, DIFF_GAIN)
    base_gray = cv2.cvtColor(w_rgb_matched, cv2.COLOR_BGR2GRAY).astype(np.float32)
    base3 = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
    metal = np.clip(base3*(diff3/255.0) + spec3, 0, 255).astype(np.uint8)

    # ---- ロボ質感ミックス ----
    a_face = (face_mask.astype(np.float32)/255.0)
    robo_enhanced = (w_rgb_matched.astype(np.float32)*(1-0.35*a_face[...,None]) +
                     metal.astype(np.float32)*(0.35*a_face[...,None]))
    robo_enhanced = np.clip(robo_enhanced,0,255).astype(np.uint8)

    # ★ Poissonの前にフェザーマスクで下ごしらえ合成（縁の白ハロ抑制）
    preblend = (robo_enhanced*(face_mask[...,None]/255.0) +
                frame*(1.0 - face_mask[...,None]/255.0)).astype(np.uint8)

    # ---- Poisson（安全ラッパ適用）----
    poisson = safe_seamless_clone(preblend, frame, face_mask)

    # ---- 中盤エフェクト（走査線は入れない）----
    if mid > 0.01:
        poisson = chromatic_aberration(poisson, face_mask, offset=cab)

    # ---- クロスフェード ----
    a = (alpha * ALPHA_STRENGTH) * (face_mask.astype(np.float32)/255.0)
    out = frame.astype(np.float32)*(1-a[...,None]) + poisson.astype(np.float32)*a[...,None]
    out = np.clip(out,0,255).astype(np.uint8)

    # ---- 髪は元フレームで上書き（前髪自然）----
    hair_mask = hair_mask_from_person(frame, face_mask, shrink_face=6, seg_scale=0.5, thr=0.55)
    if hair_mask is not None:
        hm = (hair_mask.astype(np.float32)/255.0)[...,None]
        out = (out.astype(np.float32)*(1-hm) + frame.astype(np.float32)*hm).astype(np.uint8)

    cv2.imshow("morph-robot", out)
    k = cv2.waitKey(1) & 0xFF
    if k==27: break
    if k==ord('r'):
        t_start=None; alpha=0.0; locked=False

cap.release()
cv2.destroyAllWindows()
