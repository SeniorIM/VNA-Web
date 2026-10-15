"""
استخراج دقیق منحنی S11 از عکس VNA
نسخه 2.0 با کالیبراسیون پیشرفته X (Least Squares)
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
#          کالیبراسیون پیشرفته X با Least Squares
# ============================================================

def map_original_x_to_rectified(x_orig, corners, rect_w):
    """
    تبدیل مختصات x اصلی (روی تصویر Adject شده) به x در تصویر rectify شده.
    چون rectify فقط از 4 گوشه‌ی مستطیل انجام شده، در راستای افقی:
        x_rect = (x_orig - x_tl) / (x_tr - x_tl) * (rect_w - 1)
    """
    tl = corners[0]
    tr = corners[1]
    x_tl = tl[0]
    x_tr = tr[0]
    if abs(x_tr - x_tl) < 1e-6:
        return 0.0
    return (x_orig - x_tl) / (x_tr - x_tl) * (rect_w - 1)


def calibrate_x_least_squares(x1_px, f1, x2_px, f2, extra_refs_rect, rect_w):
    """
    کالیبراسیون X با Least Squares.

    پارامترها:
        x1_px, x2_px : مختصات x در تصویر اصلی (Adject شده)
        f1, f2       : فرکانس‌های مرجع
        extra_refs_rect : لیست [(x_rect, freq)] از خطوط اضافی (قبلاً به مختصات rectify تبدیل شده)
        rect_w       : عرض تصویر rectify

    خروجی:
        (a, b, method) که freq = a * x_rect + b
        method: 'least_squares' یا 'two_point'
    """
    # تبدیل دو نقطه اصلی به مختصات rectify
    x1_r = (x1_px / (x1_px + 1)) * 0  # placeholder, در ادامه اصلاح می‌شود
    # نکته: باید x1_px و x2_px هم به مختصات rectify تبدیل شوند
    # این کار در تابع اصلی انجام می‌شود، اینجا فقط مختصات rectify می‌گیریم

    # نقاط پایه (در مختصات rectify)
    # این تابع در واقع با نقاط rectify کار می‌کند
    pass


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
        rect_w = rectified_pil.width
        rect_h = rectified_pil.height
        print(f">>> rectify انجام شد → {rect_w}x{rect_h}")

        rectified_rgb = np.array(rectified_pil)
        mask = detect_yellow_trace_np(rectified_rgb)
        print(f">>> mask: {np.count_nonzero(mask)} pixels")

        x_clean, y_clean, x_raw, y_raw = extract_trace(mask)
        print(f">>> استخراج: {len(x_clean)} نقطه")

        # ---------- کالیبراسیون X ----------
        x1_px = calib['x1']; f1 = calib['f1']
        x2_px = calib['x2']; f2 = calib['f2']

        # تبدیل x های اصلی به مختصات rectify
        x_tl = corners[0][0]
        x_tr = corners[1][0]
        if abs(x_tr - x_tl) < 1e-6:
            raise RuntimeError("گوشه‌های بالا نامعتبرند")

        def map_x_to_rect(x_orig):
            return (x_orig - x_tl) / (x_tr - x_tl) * (rect_w - 1)

        x1_r = map_x_to_rect(x1_px)
        x2_r = map_x_to_rect(x2_px)

        # خطوط مرجع اضافی
        extra_refs_raw = calib.get('extra_refs', [])
        if hasattr(extra_refs_raw, 'to_py'):
            extra_refs_raw = extra_refs_raw.to_py()
        extra_refs = []
        for ref in extra_refs_raw:
            if hasattr(ref, 'to_py'):
                ref = ref.to_py()
            x_orig = float(ref['x'])
            freq = float(ref['freq'])
            x_r = map_x_to_rect(x_orig)
            extra_refs.append((x_r, freq))

        # ساخت مجموعه نقاط مرجع
        ref_points = [(x1_r, f1), (x2_r, f2)] + extra_refs

        # حذف تکراری‌ها (اگر کاربر خط اضافی روی همان X1/X2 گذاشت)
        unique_points = []
        for (xr, fr) in ref_points:
            is_dup = False
            for (xr2, fr2) in unique_points:
                if abs(xr - xr2) < 2.0:  # tolerance 2 پیکسل
                    is_dup = True
                    break
            if not is_dup:
                unique_points.append((xr, fr))

        calib_method = 'two_point'
        if len(unique_points) >= 3:
            # Least Squares: freq = a * x_rect + b
            X = np.array([p[0] for p in unique_points], dtype=np.float64)
            F = np.array([p[1] for p in unique_points], dtype=np.float64)
            A = np.vstack([X, np.ones_like(X)]).T
            coef, *_ = np.linalg.lstsq(A, F, rcond=None)
            a_x, b_x = coef
            calib_method = f'least_squares({len(unique_points)}pts)'

            # محاسبه خطا (RMSE)
            F_pred = a_x * X + b_x
            rmse = float(np.sqrt(np.mean((F - F_pred) ** 2)))
            print(f">>> Least Squares X: freq = {a_x:.6f}*x + {b_x:.4f}, RMSE={rmse:.4f} GHz")
        else:
            # فقط 2 نقطه
            a_x = (f2 - f1) / (x2_r - x1_r)
            b_x = f1 - a_x * x1_r
            calib_method = 'two_point'
            print(f">>> Two-point X: freq = {a_x:.6f}*x + {b_x:.4f}")

        # ---------- کالیبراسیون Y (سه نقطه‌ای) ----------
        y_top = calib['y_top']; d_top = calib['d_top']
        y_mid = calib['y_mid']; d_mid = calib['d_mid']
        y_bot = calib['y_bot']; d_bot = calib['d_bot']

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
        coef_y, *_ = np.linalg.lstsq(A_ls, d_pts, rcond=None)
        a_y, b_y = coef_y
        print(f">>> کالیبراسیون Y: d = {a_y:.4f}*y + {b_y:.4f}")

        # ---------- محاسبه‌ی نهایی ----------
        frequency = a_x * x_clean + b_x
        amplitude_db = a_y * y_clean + b_y

        # محدود کردن به بازه‌ی X1..X2
        lo, hi = sorted((f1, f2))
        valid = (frequency >= lo) & (frequency <= hi)
        frequency = frequency[valid]
        amplitude_db = amplitude_db[valid]

        print(f">>> نتیجه: {len(frequency)} نقطه ({calib_method})")

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