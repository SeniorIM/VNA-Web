"""
استخراج دقیق منحنی S11 از عکس VNA
نسخه پایدار با Nearest Neighbor + کالیبراسیون ۳ نقطه‌ای Y
"""
from js import window
import json

print(">>> main.py شروع به اجرا کرد")


def _mark(name, state):
    try:
        if state == "ok": window.markLibOk(name)
        elif state == "loading": window.markLibLoading(name)
        elif state == "error": window.markLibError(name)
    except Exception:
        pass


_mark("pyodide", "ok")


try:
    _mark("numpy", "loading")
    import numpy as np
    _mark("numpy", "ok")
    print(">>> numpy آماده ✅")
except Exception as e:
    _mark("numpy", "error")
    print(f">>> خطا numpy: {e}")
    raise

try:
    _mark("pillow", "loading")
    from PIL import Image
    _mark("pillow", "ok")
    print(">>> Pillow آماده ✅")
except Exception as e:
    _mark("pillow", "error")
    print(f">>> خطا Pillow: {e}")
    raise


# ============================================================
#          توابع کمکی
# ============================================================

def numpy_to_pil(arr_rgb, w, h):
    return Image.frombytes("RGB", (w, h), arr_rgb.tobytes())


def perspective_rectify_pil(pil_img, corners):
    tl, tr, br, bl = corners
    width_a = np.linalg.norm(np.array(br) - np.array(bl))
    width_b = np.linalg.norm(np.array(tr) - np.array(tl))
    height_a = np.linalg.norm(np.array(tr) - np.array(br))
    height_b = np.linalg.norm(np.array(tl) - np.array(bl))
    out_w = max(100, int(max(width_a, width_b)))
    out_h = max(100, int(max(height_a, height_b)))

    src = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = np.array([[0, 0], [out_w - 1, 0],
                    [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)

    A = []
    b = []
    for i in range(4):
        sx, sy = src[i]
        dx, dy = dst[i]
        A.append([sx, sy, 1, 0, 0, 0, -dx*sx, -dx*sy])
        A.append([0, 0, 0, sx, sy, 1, -dy*sx, -dy*sy])
        b.append(dx)
        b.append(dy)
    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    try:
        h_coef = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        h_coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    H = np.append(h_coef, 1.0).reshape(3, 3)

    u, v = np.meshgrid(np.arange(out_w), np.arange(out_h))
    ones = np.ones_like(u)
    coords = np.stack([u, v, ones], axis=-1).reshape(-1, 3).T

    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        H_inv = H

    src_coords = H_inv @ coords
    src_coords /= src_coords[2:3, :]
    xs = src_coords[0, :].reshape(out_h, out_w)
    ys = src_coords[1, :].reshape(out_h, out_w)

    xs = np.clip(xs, 0, pil_img.width - 1).astype(np.int32)
    ys = np.clip(ys, 0, pil_img.height - 1).astype(np.int32)

    arr = np.array(pil_img)
    out_arr = arr[ys, xs]
    return Image.fromarray(out_arr.astype(np.uint8), "RGB")


def detect_yellow_trace_np(rgb_arr):
    """تشخیص زرد با شرط ساده"""
    r = rgb_arr[:, :, 0].astype(np.int16)
    g = rgb_arr[:, :, 1].astype(np.int16)
    b = rgb_arr[:, :, 2].astype(np.int16)
    mask = (r > 100) & (g > 80) & (b < 120) & \
           (r > b + 30) & (g > b + 20) & \
           (np.abs(r - g) < 80)
    return mask.astype(np.uint8)


def remove_small_components_np(mask, min_width_ratio=0.02):
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    output = np.zeros_like(mask)
    min_width = max(10, int(w * min_width_ratio))

    for y0 in range(h):
        for x0 in range(w):
            if mask[y0, x0] and not visited[y0, x0]:
                stack = [(y0, x0)]
                comp = []
                visited[y0, x0] = True
                while stack:
                    y, x = stack.pop()
                    comp.append((y, x))
                    for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if mask[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = True
                                stack.append((ny, nx))
                xs_comp = [p[1] for p in comp]
                width_comp = max(xs_comp) - min(xs_comp) + 1
                if width_comp >= min_width:
                    for (y, x) in comp:
                        output[y, x] = 1
    return output


def find_curve_center_y_np(cleaned):
    h, w = cleaned.shape
    ys = []
    for x in range(w):
        rows = np.flatnonzero(cleaned[:, x])
        ys.extend(rows.tolist())
    if len(ys) < 10:
        return h // 2
    return int(np.median(ys))


def split_groups(rows, gap=3):
    if len(rows) == 0:
        return []
    groups = []
    current = [int(rows[0])]
    for r in rows[1:]:
        r = int(r)
        if r - current[-1] <= gap:
            current.append(r)
        else:
            groups.append(current)
            current = [r]
    groups.append(current)
    return groups


def track_nearest_np(cleaned, start_y):
    """ردیابی ساده Nearest Neighbor"""
    h, w = cleaned.shape
    ys = np.full(w, np.nan, dtype=float)
    start_col = None
    for x in range(w):
        if cleaned[:, x].sum() > 0:
            start_col = x
            break
    if start_col is None:
        return ys

    rows = np.flatnonzero(cleaned[:, start_col])
    if len(rows) == 0:
        return ys
    groups = split_groups(rows)
    closest = min(groups, key=lambda g: abs(np.mean(g) - start_y))
    ys[start_col] = np.mean(closest)
    prev_y = ys[start_col]

    for x in range(start_col + 1, w):
        rows = np.flatnonzero(cleaned[:, x])
        if len(rows) == 0:
            continue
        groups = split_groups(rows)
        closest = min(groups, key=lambda g: abs(np.mean(g) - prev_y))
        ys[x] = np.mean(closest)
        prev_y = ys[x]

    prev_y = ys[start_col]
    for x in range(start_col - 1, -1, -1):
        rows = np.flatnonzero(cleaned[:, x])
        if len(rows) == 0:
            continue
        groups = split_groups(rows)
        closest = min(groups, key=lambda g: abs(np.mean(g) - prev_y))
        ys[x] = np.mean(closest)
        prev_y = ys[x]

    return ys


def fill_short_gaps(ys, max_gap_ratio=0.03):
    w = len(ys)
    valid = np.isfinite(ys)
    if valid.sum() < 10:
        return ys
    xs = np.arange(w)
    ys_interp = np.interp(xs, xs[valid], ys[valid])
    max_gap = int(w * max_gap_ratio)
    invalid = ~valid
    if invalid.sum() == 0:
        return ys_interp
    gap_groups = []
    current = []
    for i, inv in enumerate(invalid):
        if inv:
            current.append(i)
        else:
            if current:
                gap_groups.append(current)
                current = []
    if current:
        gap_groups.append(current)
    result = ys_interp.copy()
    for gap in gap_groups:
        if len(gap) > max_gap:
            result[gap[0]:gap[-1] + 1] = np.nan
    return result


def remove_isolated_spikes(ys, h, max_jump_ratio=0.20):
    ys = ys.copy()
    w = len(ys)
    threshold = h * max_jump_ratio
    for i in range(2, w - 2):
        if not np.isfinite(ys[i]):
            continue
        neighbors = []
        for j in [i - 2, i - 1, i + 1, i + 2]:
            if 0 <= j < w and np.isfinite(ys[j]):
                neighbors.append(ys[j])
        if len(neighbors) < 2:
            continue
        diffs = [abs(ys[i] - n) for n in neighbors]
        if all(d > threshold for d in diffs):
            neighbor_range = max(neighbors) - min(neighbors)
            if neighbor_range < threshold:
                left_j = None
                for j in [i - 2, i - 1]:
                    if 0 <= j < w and np.isfinite(ys[j]):
                        left_j = j
                        break
                right_j = None
                for j in [i + 1, i + 2]:
                    if 0 <= j < w and np.isfinite(ys[j]):
                        right_j = j
                        break
                if left_j is not None and right_j is not None:
                    t = (i - left_j) / (right_j - left_j)
                    ys[i] = ys[left_j] * (1 - t) + ys[right_j] * t
    return ys


def smooth_subpixel(ys, window=5):
    valid = np.isfinite(ys)
    if valid.sum() < window:
        return ys
    out = ys.copy()
    half = window // 2
    for i in range(len(ys)):
        if not np.isfinite(ys[i]):
            continue
        lo = max(0, i - half)
        hi = min(len(ys), i + half + 1)
        seg = ys[lo:hi]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= 3:
            out[i] = np.median(seg)
    return out


def extract_trace(mask):
    h, w = mask.shape
    cleaned = remove_small_components_np(mask, min_width_ratio=0.02)
    if cleaned.sum() < 30:
        raise RuntimeError("منحنی پیدا نشد.")

    start_y = find_curve_center_y_np(cleaned)
    ys_raw_full = track_nearest_np(cleaned, start_y)
    valid_count = np.isfinite(ys_raw_full).sum()
    if valid_count < max(10, int(w * 0.3)):
        raise RuntimeError(f"ردیابی ناموفق: {valid_count} نقطه.")

    xs_arr = np.arange(w, dtype=float)
    valid_raw = np.isfinite(ys_raw_full)
    x_raw = xs_arr[valid_raw]
    y_raw = ys_raw_full[valid_raw]

    ys = fill_short_gaps(ys_raw_full, max_gap_ratio=0.03)
    ys = remove_isolated_spikes(ys, h, max_jump_ratio=0.20)
    ys = smooth_subpixel(ys, window=5)

    valid = np.isfinite(ys)
    x_clean = xs_arr[valid]
    y_clean = ys[valid]
    return x_clean, y_clean, x_raw, y_raw


# ============================================================
#          Wrapper برای JavaScript
# ============================================================

def process_image(img_array, corners_list, calib_dict):
    print(">>> process_image فراخوانی شد")
    try:
        calib = calib_dict.to_py() if hasattr(calib_dict, 'to_py') else calib_dict
        corners_js = corners_list.to_py() if hasattr(corners_list, 'to_py') else corners_list

        h = window.imageHeight
        w = window.imageWidth
        print(f">>> ابعاد: {w}x{h}")

        img_rgb = np.frombuffer(bytes(img_array), dtype=np.uint8).reshape((h, w, 3))
        pil_img = numpy_to_pil(img_rgb, w, h)

        corners = np.array(corners_js, dtype=float)
        rectified_pil = perspective_rectify_pil(pil_img, corners)
        print(">>> rectify انجام شد")

        rectified_rgb = np.array(rectified_pil)
        mask = detect_yellow_trace_np(rectified_rgb)
        print(f">>> mask: {np.count_nonzero(mask)} pixels")

        x_clean, y_clean, x_raw, y_raw = extract_trace(mask)
        print(f">>> استخراج: {len(x_clean)} نقطه")

        # کالیبراسیون X
        x1 = calib['x1']; f1 = calib['f1']
        x2 = calib['x2']; f2 = calib['f2']

        # کالیبراسیون Y با ۳ نقطه
        y_top = calib['y_top']; d_top = calib['d_top']
        y_mid = calib['y_mid']; d_mid = calib['d_mid']
        y_bot = calib['y_bot']; d_bot = calib['d_bot']

        rect_h = rectified_pil.height
        tl_y = corners[0][1]
        bl_y = corners[3][1]

        def map_y_to_rect(y_orig):
            return (y_orig - tl_y) / (bl_y - tl_y) * (rect_h - 1)

        y_top_r = map_y_to_rect(y_top)
        y_mid_r = map_y_to_rect(y_mid)
        y_bot_r = map_y_to_rect(y_bot)

        y_pts = np.array([y_top_r, y_mid_r, y_bot_r])
        d_pts = np.array([d_top, d_mid, d_bot])
        A_ls = np.vstack([y_pts, np.ones(3)]).T
        coef, *_ = np.linalg.lstsq(A_ls, d_pts, rcond=None)
        a_y, b_y = coef

        print(f">>> کالیبراسیون Y: d = {a_y:.4f}*y + {b_y:.4f}")

        frequency = f1 + (x_clean - x1) * (f2 - f1) / (x2 - x1)
        amplitude_db = a_y * y_clean + b_y

        lo, hi = sorted((f1, f2))
        valid = (frequency >= lo) & (frequency <= hi)
        frequency = frequency[valid]
        amplitude_db = amplitude_db[valid]

        print(f">>> نتیجه: {len(frequency)} نقطه")

        result_json = json.dumps({
            'frequency': frequency.tolist(),
            'amplitude': amplitude_db.tolist(),
            'x_raw': x_raw.tolist(),
            'y_raw': y_raw.tolist(),
        })
        return result_json

    except Exception as e:
        print(f">>> خطا: {e}")
        import traceback
        traceback.print_exc()
        raise


window.process_image = process_image
print(">>> process_image به window وصل شد ✅")