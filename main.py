"""
استخراج دقیق منحنی S11/S21 یا فاز از عکس VNA
نسخه نهایی: پشتیبانی از منحنی‌های دامنه (زرد/سفید) و فاز (فیروزه‌ای)
با اصلاح ردیابی در نقاط تیز
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


def detect_trace_smart(rgb_arr, curve_type="magnitude"):
    """
    تشخیص هوشمند رنگ:
    - magnitude: زرد یا سفید/روشن
    - phase: فیروزه‌ای (Cyan)
    """
    r = rgb_arr[:, :, 0].astype(np.int16)
    g = rgb_arr[:, :, 1].astype(np.int16)
    b = rgb_arr[:, :, 2].astype(np.int16)

    # روشنایی کلی
    brightness = (r + g + b) / 3.0
    is_bright = brightness > 120

    # تمایز از خاکستری (برای منحنی سفید)
    is_colorful = (np.abs(r - g) > 15) | (np.abs(g - b) > 15) | (np.abs(r - b) > 15)

    # زرد (برای منحنی S11)
    is_yellow = (r > 100) & (g > 80) & (b < 120) & (r > b + 30) & (g > b + 20) & (np.abs(r - g) < 80)

    # فیروزه‌ای (برای منحنی فاز)
    is_cyan = (g > 100) & (b > 100) & (r < 160) & (g > r + 20) & (b > r + 20)

    if curve_type == "phase":
        # برای فاز: فیروزه‌ای یا روشن رنگی
        mask = is_cyan | (is_bright & is_colorful)
    else:
        # برای دامنه: زرد یا سفید روشن
        mask = is_yellow | (is_bright & ~is_cyan) # نادیده گرفتن فیروزه‌ای
        
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
    """
    ردیابی هوشمند با قابلیت عبور از پرش‌های عمودی (برای منحنی فاز)
    """
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

    # حرکت به راست
    for x in range(start_col + 1, w):
        rows = np.flatnonzero(cleaned[:, x])
        if len(rows) == 0:
            continue
        groups = split_groups(rows)
        closest = min(groups, key=lambda g: abs(np.mean(g) - prev_y))
        ys[x] = np.mean(closest)
        prev_y = ys[x]

    # حرکت به چپ
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


def extract_trace(mask, curve_type="magnitude"):
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

    # تنظیمات فیلتر بر اساس نوع منحنی
    if curve_type == "phase":
        # منحنی فاز: پرش‌های عمودی واقعی هستند، نباید حذف شوند
        ys = fill_short_gaps(ys_raw_full, max_gap_ratio=0.01)
        ys = remove_isolated_spikes(ys, h, max_jump_ratio=0.95) # اجازه پرش ۱۰۰٪
        ys = smooth_subpixel(ys, window=3) # هموارسازی ملایم
    else:
        # منحنی دامنه: حذف نویز و هموارسازی عادی
        ys = fill_short_gaps(ys_raw_full, max_gap_ratio=0.03)
        ys = remove_isolated_spikes(ys, h, max_jump_ratio=0.20)
        ys = smooth_subpixel(ys, window=5)

    valid = np.isfinite(ys)
    x_clean = xs_arr[valid]
    y_clean = ys[valid]
    return x_clean, y_clean, x_raw, y_raw


# ============================================================
#     تبدیل مختصات اصلی به مختصات rectify شده
# ============================================================

def make_x_mapper(corners, rect_w):
    x_tl = float(corners[0][0])
    x_tr = float(corners[1][0])
    denom = x_tr - x_tl
    if abs(denom) < 1e-6:
        denom = 1.0

    def mapper(x_orig):
        return (float(x_orig) - x_tl) / denom * (rect_w - 1)

    return mapper


def calibrate_x(x1_r, f1, x2_r, f2, extra_refs_r):
    points = [(x1_r, f1), (x2_r, f2)]
    for (xr, fr) in extra_refs_r:
        is_dup = False
        for (xr2, _) in points:
            if abs(xr - xr2) < 2.0:
                is_dup = True
                break
        if not is_dup:
            points.append((xr, fr))

    n = len(points)
    if n >= 3:
        X = np.array([p[0] for p in points], dtype=np.float64)
        F = np.array([p[1] for p in points], dtype=np.float64)
        A = np.vstack([X, np.ones_like(X)]).T
        coef, *_ = np.linalg.lstsq(A, F, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        F_pred = a * X + b
        rmse = float(np.sqrt(np.mean((F - F_pred) ** 2)))
        method = f"least_squares({n}pts)"
        return a, b, method, rmse
    else:
        denom = x2_r - x1_r
        if abs(denom) < 1e-6:
            denom = 1.0
        a = (f2 - f1) / denom
        b = f1 - a * x1_r
        return a, b, "two_point", 0.0


def calibrate_y(y_top_r, d_top, y_mid_r, d_mid, y_bot_r, d_bot):
    y_pts = np.array([y_top_r, y_mid_r, y_bot_r], dtype=np.float64)
    d_pts = np.array([d_top, d_mid, d_bot], dtype=np.float64)
    A_ls = np.vstack([y_pts, np.ones(3)]).T
    coef, *_ = np.linalg.lstsq(A_ls, d_pts, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    return a, b


def make_y_mapper(corners, rect_h):
    tl_y = float(corners[0][1])
    bl_y = float(corners[3][1])
    denom = bl_y - tl_y
    if abs(denom) < 1e-6:
        denom = 1.0

    def mapper(y_orig):
        return (float(y_orig) - tl_y) / denom * (rect_h - 1)

    return mapper


# ============================================================
#          Wrapper برای JavaScript
# ============================================================

def process_image(img_array, corners_list, calib_dict):
    print(">>> process_image فراخوانی شد")
    try:
        calib = calib_dict.to_py() if hasattr(calib_dict, 'to_py') else calib_dict
        corners_js = corners_list.to_py() if hasattr(corners_list, 'to_py') else corners_list

        h = int(window.imageHeight)
        w = int(window.imageWidth)
        print(f">>> ابعاد: {w}x{h}")

        img_rgb = np.frombuffer(bytes(img_array), dtype=np.uint8).reshape((h, w, 3))
        pil_img = numpy_to_pil(img_rgb, w, h)

        corners = np.array(corners_js, dtype=float)
        rectified_pil = perspective_rectify_pil(pil_img, corners)
        rect_w = rectified_pil.width
        rect_h = rectified_pil.height
        print(f">>> rectify: {rect_w}x{rect_h}")

        rectified_rgb = np.array(rectified_pil)
        
        curve_type = calib.get('curve_type', 'magnitude')
        mask = detect_trace_smart(rectified_rgb, curve_type)
        print(f">>> mask ({curve_type}): {int(np.count_nonzero(mask))} pixels")

        x_clean, y_clean, x_raw, y_raw = extract_trace(mask, curve_type)
        print(f">>> استخراج: {len(x_clean)} نقطه")

        # ---------- کالیبراسیون X ----------
        x1_px = float(calib['x1']); f1 = float(calib['f1'])
        x2_px = float(calib['x2']); f2 = float(calib['f2'])

        map_x = make_x_mapper(corners, rect_w)
        x1_r = map_x(x1_px)
        x2_r = map_x(x2_px)

        extra_raw = calib.get('extra_refs', [])
        if hasattr(extra_raw, 'to_py'):
            extra_raw = extra_raw.to_py()
        extra_refs_r = []
        for ref in extra_raw:
            if hasattr(ref, 'to_py'):
                ref = ref.to_py()
            xr_orig = float(ref['x'])
            fr = float(ref['freq'])
            extra_refs_r.append((map_x(xr_orig), fr))
        
        a_x, b_x, calib_method, rmse = calibrate_x(x1_r, f1, x2_r, f2, extra_refs_r)
        print(f">>> X calib [{calib_method}]: freq = {a_x:.6f}*x + {b_x:.4f} (RMSE={rmse:.4f})")

        # ---------- کالیبراسیون Y ----------
        y_top = float(calib['y_top']); d_top = float(calib['d_top'])
        y_mid = float(calib['y_mid']); d_mid = float(calib['d_mid'])
        y_bot = float(calib['y_bot']); d_bot = float(calib['d_bot'])

        map_y = make_y_mapper(corners, rect_h)
        y_top_r = map_y(y_top)
        y_mid_r = map_y(y_mid)
        y_bot_r = map_y(y_bot)

        a_y, b_y = calibrate_y(y_top_r, d_top, y_mid_r, d_mid, y_bot_r, d_bot)
        print(f">>> Y calib: d = {a_y:.4f}*y + {b_y:.4f}")

        # ---------- محاسبه نهایی ----------
        frequency = a_x * x_clean + b_x
        amplitude_db = a_y * y_clean + b_y

        # مرتب‌سازی بر اساس فرکانس
        order = np.argsort(frequency)
        frequency = frequency[order]
        amplitude_db = amplitude_db[order]

        # محدود به بازه X1..X2
        lo, hi = sorted((f1, f2))
        valid = (frequency >= lo) & (frequency <= hi)
        frequency = frequency[valid]
        amplitude_db = amplitude_db[valid]

        print(f">>> نتیجه: {len(frequency)} نقطه [{calib_method}]")

        result_json = json.dumps({
            'frequency': frequency.tolist(),
            'amplitude': amplitude_db.tolist(),
            'x_raw': x_raw.tolist(),
            'y_raw': y_raw.tolist(),
            'calib_method': calib_method,
        })
        return result_json

    except Exception as e:
        print(f">>> خطا: {e}")
        import traceback
        traceback.print_exc()
        raise


window.process_image = process_image
print(">>> process_image به window وصل شد ✅")