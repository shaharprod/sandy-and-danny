# Builds coloring pages for the app from AI line-art images.
# Output per page (1600x900):
#   pages/eX_pY.png    visible black lines on transparent background
#   pages/eX_pY_w.png  "walls" for the paint bucket: the lines + invisible gap-closing segments
#   thumbs/eX_pY.jpg   320x180 preview
# Gap closing: every open line end is extended toward the nearest line in front of it
# (up to GAP px), so eyes, faces, ears etc. become closed regions for the bucket fill.
from skimage.morphology import skeletonize
import cv2, numpy as np, sys, os, glob
from skimage.morphology import skeletonize

W, H = 1600, 900
GAP = 34          # max gap to close, in page pixels
LINE_T = 175      # gray below this counts as line

def load_gray(path):
    g = cv2.imread(path, 0)
    h, w = g.shape
    # resize to cover 1600x900 then center crop
    s = max(W / w, H / h)
    g = cv2.resize(g, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    y = (g.shape[0] - H) // 2; x = (g.shape[1] - W) // 2
    return g[y:y + H, x:x + W]

def endpoints(skel):
    k = np.array([[1,1,1],[1,10,1],[1,1,1]], np.uint8)
    n = cv2.filter2D((skel > 0).astype(np.uint8), -1, k, borderType=cv2.BORDER_CONSTANT)
    ys, xs = np.where(n == 11)
    return list(zip(xs, ys))

def walk_dir(skel, x, y, steps=10):
    # follow the skeleton backwards from an endpoint to estimate the stroke direction
    px, py = x, y; path = [(x, y)]; prev = None
    for _ in range(steps):
        nxt = None
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == dy == 0: continue
                qx, qy = px + dx, py + dy
                if 0 <= qx < W and 0 <= qy < H and skel[qy, qx] and (qx, qy) not in path:
                    nxt = (qx, qy); break
            if nxt: break
        if not nxt: break
        path.append(nxt); px, py = nxt
    if len(path) < 3: return None, path
    v = np.array([x - path[-1][0], y - path[-1][1]], np.float32)
    n = np.linalg.norm(v)
    return (v / n if n else None), path

def close_gaps(line):
    """Close only real breaks: (a) a line end within SHORT px of another line straight ahead,
    (b) two line ends facing each other within PAIR px. Open design strokes (nose, smile,
    cheek lines) are left alone, so faces are not split by invisible walls."""
    SHORT, PAIR = 9, 26
    skel = skeletonize(line > 0).astype(np.uint8)
    eps = endpoints(skel)
    info = []
    for (x, y) in eps:
        d, path = walk_dir(skel, x, y)
        info.append((x, y, d, path))
    gaps = np.zeros_like(line); added = 0; used = set()
    # (b) facing end pairs
    for i, (x, y, d, p) in enumerate(info):
        if d is None or i in used: continue
        best = None
        for j, (x2, y2, d2, p2) in enumerate(info):
            if j == i or j in used or d2 is None: continue
            vx, vy = x2 - x, y2 - y; dist = np.hypot(vx, vy)
            if dist < 3 or dist > PAIR: continue
            if (vx * d[0] + vy * d[1]) / dist < 0.7: continue          # j ahead of i
            if (-vx * d2[0] - vy * d2[1]) / dist < 0.7: continue       # i ahead of j
            if best is None or dist < best[0]: best = (dist, j)
        if best:
            j = best[1]; cv2.line(gaps, (x, y), (info[j][0], info[j][1]), 1, 3)
            used.add(i); used.add(j); added += 1
    # (a) short hop to any line straight ahead
    for i, (x, y, d, p) in enumerate(info):
        if d is None or i in used: continue
        x0, x1 = max(0, x - SHORT), min(W, x + SHORT + 1); y0, y1 = max(0, y - SHORT), min(H, y + SHORT + 1)
        sy, sx = np.where(line[y0:y1, x0:x1] > 0)
        if not len(sx): continue
        vx = sx + x0 - x; vy = sy + y0 - y; dist = np.hypot(vx, vy)
        with np.errstate(invalid='ignore', divide='ignore'):
            cos = (vx * d[0] + vy * d[1]) / dist
        ok = (dist > 2.5) & (dist <= SHORT) & (cos > 0.75)
        if not ok.any(): continue
        c = np.where(ok)[0]; b = c[np.argmin(dist[c])]
        cv2.line(gaps, (x, y), (int(sx[b] + x0), int(sy[b] + y0)), 1, 3); added += 1
    return gaps, added, len(eps)

def build(src, name, outdir):
    g = load_gray(src).astype(np.float32)
    # visible layer: soft anti-aliased black lines on transparent
    dark = np.clip((235 - g) / (235 - 70), 0, 1)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark = np.maximum(dark, cv2.dilate(dark, k3) * 0.85)
    vis = np.zeros((H, W, 4), np.uint8); vis[..., 3] = (dark * 255).astype(np.uint8)
    # bucket walls
    line = (g < LINE_T).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(line, connectivity=8)
    for i in range(1, n):                      # drop isolated specks
        if stats[i, cv2.CC_STAT_AREA] < 12: line[lab == i] = 0
    line = cv2.morphologyEx(line, cv2.MORPH_CLOSE, k3)   # seal 1-2px cracks
    gaps, added, neps = close_gaps(line)
    walls = np.maximum(line, gaps)
    walls = np.maximum(walls, (vis[..., 3] > 110).astype(np.uint8))
    cv2.imwrite(f"{outdir}/pages/{name}.png", vis, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(f"{outdir}/pages/{name}_w.png", walls * 255, [cv2.IMWRITE_PNG_COMPRESSION, 9, cv2.IMWRITE_PNG_BILEVEL, 1])
    t = cv2.resize((255 * (1 - dark)).astype(np.uint8), (320, 180), interpolation=cv2.INTER_AREA)
    cv2.imwrite(f"{outdir}/thumbs/{name}.jpg", t, [cv2.IMWRITE_JPEG_QUALITY, 80])
    nreg = cv2.connectedComponents(1 - walls, connectivity=4)[0]
    print(f"{name}: endpoints {neps}, gaps closed {added}, regions {nreg}")
    return walls, gaps

if __name__ == '__main__':
    raw, out = sys.argv[1], sys.argv[2]
    for f in sorted(glob.glob(f"{raw}/e*_p*.png")):
        build(f, os.path.basename(f)[:-4], out)
