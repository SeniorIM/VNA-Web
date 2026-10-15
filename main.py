"""
استخراج دقیق منحنی S11 از عکس VNA
نسخه نهایی وب - با JSON output و db_step
نسخه آفلاین PyScript 2026.7.3
"""
import pyodide_js
from js import window
import json

print(">>> main.py شروع به اجرا کرد")


# ============================================================
#          Import کتابخانه‌ها
# ============================================================
# پکیج‌ها از pyscript.toml به صورت خودکار نصب می‌شن
import numpy as np
import cv2


# ============================================================
#          اطلاع‌رسانی وضعیت کتابخانه‌ها به UI
# ============================================================
def _mark_ok(name):
    try:
        window.markLibOk(name)
    except Exception:
        pass


def _mark_error(name):
    try:
        window.markLibError(name)
    except Exception:
        pass


# همه پکیج‌ها در pyscript.toml تعریف شدن، پس اینجا فقط علامت OK می‌زنیم
_mark_ok("pyodide")
_mark_ok("numpy")
_mark_ok("opencv")
_mark_ok("pandas")
_mark_ok("openpyxl")
_mark_ok("scipy")
print(">>> همه کتابخانه‌ها آماده هستند ✅")


# ============================================================
#          توابع پردازش تصویر
# ============================================================

def perspective_rectify(image_bgr, corners):
    tl, tr, br, bl = corners
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    width = max(100, int(max(width_a, width_b)))
    height = max(100, int(max(height_a, height_b)))
    dst = np.array([[0, 0], [width - 1, 0],
                    [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    # درون‌یابی مکعبی برای دقت بالاتر
    return cv2.warpPerspective(image_bgr, matrix, (width, height),
                                flags=cv2.INTER_CUBIC)


def detect_yellow_trace(image_bgr):
    """
    تشخیص منحنی زرد با دقت بالا:
    - blur ملایم قبل از HSV
    - دو بازه HSV (زرد روشن + زرد تیره)
    - فیلتر morphologic پیشرفته
    """
    blurred = cv2.GaussianBlur(image_bgr, (3, 3), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # بازه اصلی زرد
    lower1 = np.array([12, 40, 80], dtype=np.uint8)
    upper1 = np.array([45, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower1, upper1)

    # بازه مکمل برای زرد کم‌رنگ
    lower2 = np.array([20, 25, 60], dtype=np.uint8)
    upper2 = np.array([40, 255, 255], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower2, upper2)

    mask = cv2.bitwise_or(mask1, mask2)

    # فیلتر نویز
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # اتصال خطوط قطع‌شده افقی
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))
    return mask


def remove_small_components(mask, min_width_ratio=0.02):
    h, w = mask.shape
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    min_width = max(10, int(w * min_width_ratio))
    for label in range(1, count):
        x, y, cw, ch, area = stats[label]
        if cw >= min_width:
            cleaned[labels == label] = 1
    return cleaned


def find_curve_center_y(cleaned):
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


def track_nearest(cleaned, start_y):
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
    """
    هموارسازی median سبک روی مسیر برای کاهش نویز
    (بدون از بین بردن اسپایک‌های واقعی)
    """
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
    cleaned = remove_small_components(mask, min_width_ratio=0.02)
    if cleaned.sum() < 30:
        raise RuntimeError("منحنی پیدا نشد.")

    start_y = find_curve_center_y(cleaned)
    ys_raw_full = track_nearest(cleaned, start_y)
    valid_count = np.isfinite(ys_raw_full).sum()
    if valid_count < max(10, int(w * 0.3)):
        raise RuntimeError(f"ردیابی ناموفق: {valid_count} نقطه.")

    xs_arr = np.arange(w, dtype=float)
    valid_raw = np.isfinite(ys_raw_full)
    x_raw = xs_arr[valid_raw]
    y_raw = ys_raw_full[valid_raw]

    # بهبود دقت
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
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        print(">>> RGB→BGR انجام شد")

        corners = np.array(corners_js, dtype=float)
        rectified = perspective_rectify(img_bgr, corners)
        print(">>> rectify انجام شد")

        mask = detect_yellow_trace(rectified)
        print(f">>> mask: {np.count_nonzero(mask)} pixels")

        x_clean, y_clean, x_raw, y_raw = extract_trace(mask)
        print(f">>> استخراج: {len(x_clean)} نقطه")

        x1 = calib['x1']; f1 = calib['f1']
        x2 = calib['x2']; f2 = calib['f2']
        y1 = calib['y1']; d1 = calib['d1']
        y2 = calib['y2']; d2 = calib['d2']

        db_step = float(calib.get('db_step', 10))
        print(f">>> db_step = {db_step}")

        frequency = f1 + (x_clean - x1) * (f2 - f1) / (x2 - x1)
        amplitude_db = d1 + (y_clean - y1) * (d2 - d1) / (y2 - y1)

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
            'db_step': db_step,
        })
        return result_json

    except Exception as e:
        print(f">>> خطا: {e}")
        import traceback
        traceback.print_exc()
        raise


# ============================================================
#          اتصال به window
# ============================================================

window.process_image = process_image
print(">>> process_image به window وصل شد ✅")