"""
استخراج دقیق منحنی S11 از عکس VNA
نسخه پیشرفته با الگوریتم Viterbi (Dynamic Programming)
با پارامترهای قابل تنظیم از UI
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
    """تصحیح پرسپکتیو با درون‌یابی دوخطی"""
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
    xs_f = src_coords[0, :].reshape(out_h, out_w)
    ys_f = src_coords[1, :].reshape(out_h, out_w)

    arr = np.array(pil_img).astype(np.float32)
    h_img, w_img = arr.shape[:2]
    xs_f = np.clip(xs_f, 0, w_img - 1.001)
    ys_f = np.clip(ys_f, 0, h_img - 1.001)

    x0 = np.floor(xs_f).astype(np.int32)
    y0 = np.floor(ys_f).astype(np.int32)
    x1 = np.minimum(x0 + 1, w_img - 1)
    y1 = np.minimum(y0 + 1, h_img - 1)

    wx = (xs_f - x0)[..., None]
    wy = (ys_f - y0)[..., None]

    c00 = arr[y0, x0]; c10 = arr[y0, x1]
    c01 = arr[y1, x0]; c11 = arr[y1, x1]

    out_arr = (c00 * (1 - wx) * (1 - wy) +
               c10 * wx * (1 - wy) +
               c01 * (1 - wx) * wy +
               c11 * wx * wy)

    return Image.fromarray(np.clip(out_arr, 0, 255).astype(np.uint8), "RGB")


def detect_yellow_trace_np(rgb_arr):
    """تشخیص زرد با آستانه‌گذاری ثابت"""
    r = rgb_arr[:, :, 0].astype(np.int16)
    g = rgb_arr[:, :, 1].astype(np.int16)
    b = rgb_arr[:, :, 2].astype(np.int16)

    mask = (
        (r > 100) & (g > 80) & (b < 120) &
        (r > b + 30) & (g > b + 20) &
        (np.abs(r - g) < 80) &
        ((r + g) / 2 - b > 50) &
        ((r + g + b) > 200)
    )
    return mask.astype(np.uint8)


def remove_small_components_np(mask, min_width_ratio=0.02):
    """حذف اجزای کوچک با BFS"""
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


# ============================================================
#          الگوریتم Viterbi
# ============================================================

def viterbi_track(cleaned, start_y,
                  LAMBDA_SMOOTH=0.8, SIGMA_OBS=3.0, MAX_JUMP_RATIO=0.15):
    """
    ردیابی منحنی با الگوریتم Viterbi (Dynamic Programming)
    پارامترها از UI قابل تنظیم هستن
    """
    h, w = cleaned.shape
    ys = np.full(w, np.nan, dtype=float)

    # ===== کاندیدها =====
    candidates = []
    for x in range(w):
        rows = np.flatnonzero(cleaned[:, x])
        if len(rows) == 0:
            candidates.append(np.array([], dtype=float))
            continue
        groups = []
        current = [int(rows[0])]
        for r in rows[1:]:
            r = int(r)
            if r - current[-1] <= 3:
                current.append(r)
            else:
                groups.append(current)
                current = [r]
        groups.append(current)
        centers = np.array([np.mean(g) for g in groups], dtype=float)
        candidates.append(centers)

    start_col = None
    for x in range(w):
        if len(candidates[x]) > 0:
            start_col = x
            break
    if start_col is None:
        return ys

    MAX_JUMP = h * MAX_JUMP_RATIO

    costs = {}
    prevs = {}

    init_costs = np.zeros(len(candidates[start_col]))
    costs[start_col] = init_costs
    prevs[start_col] = np.full(len(candidates[start_col]), -1, dtype=int)

    # ===== Forward: چپ به راست =====
    for x in range(start_col + 1, w):
        if len(candidates[x]) == 0:
            continue
        prev_x = None
        for px in range(x - 1, -1, -1):
            if len(candidates[px]) > 0 and px in costs:
                prev_x = px
                break
        if prev_x is None:
            costs[x] = np.zeros(len(candidates[x]))
            prevs[x] = np.full(len(candidates[x]), -1, dtype=int)
            continue

        prev_costs = costs[prev_x]
        prev_cands = candidates[prev_x]
        curr_cands = candidates[x]

        curr_costs = np.full(len(curr_cands), np.inf)
        curr_prevs = np.full(len(curr_cands), -1, dtype=int)

        gap = x - prev_x

        for i, cy in enumerate(curr_cands):
            best_cost = np.inf
            best_prev = -1
            for j, py in enumerate(prev_cands):
                dy = abs(cy - py)
                if dy > MAX_JUMP * gap:
                    continue
                trans_cost = (dy ** 2) / (2 * SIGMA_OBS ** 2)
                if dy > SIGMA_OBS * 2:
                    trans_cost += (dy - SIGMA_OBS * 2) * LAMBDA_SMOOTH
                total = prev_costs[j] + trans_cost
                if total < best_cost:
                    best_cost = total
                    best_prev = j
            curr_costs[i] = best_cost
            curr_prevs[i] = best_prev

        costs[x] = curr_costs
        prevs[x] = curr_prevs

    # ===== Forward: راست به چپ =====
    for x in range(start_col - 1, -1, -1):
        if len(candidates[x]) == 0:
            continue
        next_x = None
        for nx in range(x + 1, w):
            if len(candidates[nx]) > 0 and nx in costs:
                next_x = nx
                break
        if next_x is None:
            costs[x] = np.zeros(len(candidates[x]))
            prevs[x] = np.full(len(candidates[x]), -1, dtype=int)
            continue

        next_costs = costs[next_x]
        next_cands = candidates[next_x]
        curr_cands = candidates[x]

        curr_costs = np.full(len(curr_cands), np.inf)
        curr_prevs = np.full(len(curr_cands), -1, dtype=int)

        gap = next_x - x

        for i, cy in enumerate(curr_cands):
            best_cost = np.inf
            best_prev = -1
            for j, ny in enumerate(next_cands):
                dy = abs(cy - ny)
                if dy > MAX_JUMP * gap:
                    continue
                trans_cost = (dy ** 2) / (2 * SIGMA_OBS ** 2)
                if dy > SIGMA_OBS * 2:
                    trans_cost += (dy - SIGMA_OBS * 2) * LAMBDA_SMOOTH
                total = next_costs[j] + trans_cost
                if total < best_cost:
                    best_cost = total
                    best_prev = j
            curr_costs[i] = best_cost
            curr_prevs[i] = best_prev

        costs[x] = curr_costs
        prevs[x] = curr_prevs

    # ===== Backtracking =====
    end_x = None
    for x in range(w - 1, -1, -1):
        if x in costs and len(candidates[x]) > 0 and np.any(np.isfinite(costs[x])):
            end_x = x
            break
    if end_x is None:
        return ys

    best_idx = int(np.argmin(costs[end_x]))
    ys[end_x] = candidates[end_x][best_idx]

    x = end_x
    while x > 0:
        idx = best_idx
        prev_x = None
        for px in range(x - 1, -1, -1):
            if px in costs:
                prev_x = px
                break
        if prev_x is None:
            break
        prev_idx = prevs[x][idx]
        if prev_idx < 0:
            break
        ys[prev_x] = candidates[prev_x][prev_idx]
        x = prev_x
        best_idx = prev_idx

    # پر کردن شکاف‌های بزرگ
    valid_indices = np.where(np.isfinite(ys))[0]
    if len(valid_indices) >= 2:
        max_gap = int(w * 0.05)
        for i in range(len(valid_indices) - 1):
            if valid_indices[i+1] - valid_indices[i] > max_gap:
                ys[valid_indices[i]+1:valid_indices[i+1]] = np.nan

    return ys


def smooth_subpixel(ys, window=5):
    """هموارسازی median سبک"""
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


def extract_trace(mask, LAMBDA_SMOOTH=0.8, SIGMA_OBS=3.0, MAX_JUMP_RATIO=0.15):
    """استخراج مسیر با Viterbi"""
    h, w = mask.shape
    cleaned = remove_small_components_np(mask, min_width_ratio=0.02)
    if cleaned.sum() < 30:
        raise RuntimeError("منحنی پیدا نشد.")

    ys_all = []
    for x in range(w):
        rows = np.flatnonzero(cleaned[:, x])
        ys_all.extend(rows.tolist())
    if len(ys_all) < 10:
        raise RuntimeError("پیکسل زرد کافی پیدا نشد.")
    start_y = int(np.median(ys_all))

    ys_full = viterbi_track(
        cleaned, start_y,
        LAMBDA_SMOOTH=LAMBDA_SMOOTH,
        SIGMA_OBS=SIGMA_OBS,
        MAX_JUMP_RATIO=MAX_JUMP_RATIO
    )
    valid_count = np.isfinite(ys_full).sum()
    if valid_count < max(10, int(w * 0.3)):
        raise RuntimeError(f"Viterbi ناموفق: {valid_count} نقطه.")

    xs_arr = np.arange(w, dtype=float)
    valid_raw = np.isfinite(ys_full)
    x_raw = xs_arr[valid_raw]
    y_raw = ys_full[valid_raw]

    ys = smooth_subpixel(ys_full, window=5)

    valid = np.isfinite(ys)
    x_clean = xs_arr[valid]
    y_clean = ys[valid]
    return x_clean, y_clean, x_raw, y_raw


# ============================================================
#          Wrapper برای JavaScript
# ============================================================

def process_image(img_array, corners_list, calib_dict):
    print(">>> process_image فراخوانی شد (Viterbi)")
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

        # ===== پارامترهای Viterbi =====
        lambda_smooth = float(calib.get('LAMBDA_SMOOTH', 0.8))
        sigma_obs = float(calib.get('SIGMA_OBS', 3.0))
        max_jump_ratio = float(calib.get('MAX_JUMP_RATIO', 0.15))
        print(f">>> پروفایل Viterbi: λ={lambda_smooth}, σ={sigma_obs}, jump={max_jump_ratio}")

        x_clean, y_clean, x_raw, y_raw = extract_trace(
            mask,
            LAMBDA_SMOOTH=lambda_smooth,
            SIGMA_OBS=sigma_obs,
            MAX_JUMP_RATIO=max_jump_ratio
        )
        print(f">>> Viterbi استخراج: {len(x_clean)} نقطه")

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
print(">>> process_image به window وصل شد ✅ (Viterbi)")