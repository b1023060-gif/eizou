import cv2
import numpy as np
# ==== run marker (drop-in) ====
import os, time, cv2
APP_NAME = "UNCANNY_DEMO"
BUILD_TS = time.strftime("%Y-%m-%d %H:%M:%S")
print(f"[START] {APP_NAME}  file={__file__}  ts={BUILD_TS}", flush=True)

WIN = f"{APP_NAME} | {BUILD_TS}"        # ←ウィンドウ名にも刻印
# =================================


# ==== スタイル制御（グローバル） ====
STYLE = 0           # 0..3 で切替（mキー）
ROBO_STRENGTH = 1.0 # 0.5〜2.0 推奨（';' と "'" で調整）
HEX_OVERLAY = True  # gキーで切替

def _posterize(bgr, levels=6):
    step = 256 // max(2, levels)
    return (bgr // step) * step

def _sobel_normals(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    n = np.dstack([gx, gy, np.ones_like(gx)])
    n = n / (np.linalg.norm(n, axis=2, keepdims=True) + 1e-6)
    return n  # (ny,nx,3) 法線もどき

def _specular_from_normals(normals, light_dir=(0.4,-0.2,0.9), shininess=48):
    l = np.array(light_dir, dtype=np.float32)
    l = l / np.linalg.norm(l)
    ndotl = np.clip((normals * l).sum(axis=2), 0, 1)
    # ハイライトを強調
    spec = (ndotl ** shininess)
    return (np.clip(spec * 255, 0, 255)).astype(np.uint8)

def _panel_lines(gray):
    # エッジ+適応二値でパネル境界を抽出
    e1 = cv2.Canny(gray, 50, 120)
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 31, 7)
    lines = cv2.bitwise_or(e1, th)
    lines = cv2.medianBlur(lines, 3)
    return lines

def _blue_chrome(bgr):
    # 青白いクローム調
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    steel = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    steel = cv2.applyColorMap(steel, cv2.COLORMAP_BONE)
    # わずかにシアン寄り
    steel = cv2.addWeighted(steel, 1.0, np.full_like(steel, (10,20,30)), 0.2, 0)
    return steel

def _hex_overlay(shape, cell=14, thickness=1):
    h, w = shape[:2]
    img = np.zeros((h, w), np.uint8)
    # 六角形グリッド描画
    r = cell/2.0
    dx = int(cell)
    dy = int(np.sqrt(3)*r)
    for y in range(0, h+dy, dy):
        shift = (y//dy) % 2
        for x in range(0, w+dx, dx):
            cx = x + (dx//2 if shift else 0)
            pts = []
            for k in range(6):
                ang = np.deg2rad(60*k + 30)
                px = int(cx + r*np.cos(ang))
                py = int(y + r*np.sin(ang))
                pts.append([px,py])
            pts = np.array([pts], np.int32)
            cv2.polylines(img, pts, True, 255, thickness)
    return img

def _scanline(bgr, step=3, dark=0.25):
    out = bgr.copy()
    out[::step, :, :] = (out[::step, :, :] * (1.0-dark)).astype(np.uint8)
    return out

def robotize(face_bgr, style=0):
    """機械化スタイルを複数用意。style=0..3"""
    h, w = face_bgr.shape[:2]
    base = _blue_chrome(face_bgr)

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    normals = _sobel_normals(gray)
    spec = _specular_from_normals(normals, shininess=40)
    spec = cv2.GaussianBlur(spec, (0,0), 1.0)

    # パネル線
    lines = _panel_lines(gray)
    lines_bgr = cv2.cvtColor(lines, cv2.COLOR_GRAY2BGR)

    # 六角グリッド（薄く）
    if HEX_OVERLAY:
        hexm = _hex_overlay(face_bgr.shape, cell=14, thickness=1)
        hexm = cv2.GaussianBlur(hexm, (3,3), 0)
        hexb = cv2.cvtColor(hexm, cv2.COLOR_GRAY2BGR)
        base = cv2.addWeighted(base, 1.0, (hexb//3), 0.5, 0)

    # 目あたり赤発光（上1/3〜1/2）
    y1, y2 = h//3, h//2
    band = base[y1:y2]
    mask = cv2.threshold(cv2.cvtColor(band, cv2.COLOR_BGR2GRAY),
                         60, 255, cv2.THRESH_BINARY)[1]
    band[mask > 0] = (0,0,255)
    base[y1:y2] = band

    # スキャンライン
    base = _scanline(base, step=3, dark=0.22)

    # 金属光沢合成
    spec_bgr = cv2.cvtColor(spec, cv2.COLOR_GRAY2BGR)
    chrome = cv2.addWeighted(base, 1.0, spec_bgr, 0.6*ROBO_STRENGTH, 0)

    # スタイル分岐
    if style == 0:
        # Mk-I: クローム + パネル線
        out = cv2.addWeighted(chrome, 1.0, lines_bgr, 0.35*ROBO_STRENGTH, 0)

    elif style == 1:
        # Mk-II: ポスタライズ＋強ハイライトでアニメ寄り
        post = _posterize(chrome, levels=5)
        out  = cv2.addWeighted(post, 1.0, spec_bgr, 0.8*ROBO_STRENGTH, 0)
        out  = cv2.addWeighted(out, 1.0, lines_bgr, 0.45*ROBO_STRENGTH, 0)

    elif style == 2:
        # Mk-III: ワイヤーフレーム強調（エッジ白焼き）
        edges = cv2.Canny(gray, 80, 160)
        edges = cv2.dilate(edges, np.ones((3,3),np.uint8), 1)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        out = cv2.addWeighted(chrome, 0.9, edges_bgr, 0.9*ROBO_STRENGTH, 0)

    else:
        # Mk-IV: 冷たい青鋼＋強パネル＋高光沢
        cool = cv2.addWeighted(chrome, 1.2, np.zeros_like(chrome), 0, -10)
        out  = cv2.addWeighted(cool, 1.0, lines_bgr, 0.6*ROBO_STRENGTH, 0)
        out  = np.clip(out, 0, 255).astype(np.uint8)

    return out
