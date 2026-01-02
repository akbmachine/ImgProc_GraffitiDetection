import cv2
import numpy as np


def auto_canny(gray, sigma=0.5):
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)


def find_contours_compat(bin_img):
    res = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return res[0] if len(res) == 2 else res[1]


def detect_graffiti_traditional_v2(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError("Görüntü yüklenemedi")

    h, w = img.shape[:2]
    output = img.copy()

    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]

    # OTSU yerine sabit ama yumuşak eşik (daha stabil)
    sat_mask = cv2.inRange(s, 40, 255)  # 40 -> 30 da deneyebilirsin

    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    edge_mask = auto_canny(gray, sigma=0.5)

    # AND yerine OR (küçük/uzak grafiti için daha affedici)
    combined = cv2.bitwise_or(sat_mask, edge_mask)

    k = max(3, (min(h, w) // 350) | 1)
    kernel = np.ones((k, k), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours = find_contours_compat(combined)

    min_area = 0.00005 * (h * w)
    max_area = 0.25 * (h * w)

    found = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, ww, hh = cv2.boundingRect(cnt)

        aspect = ww / (hh + 1e-6)
        if aspect > 12 or aspect < 0.08:
            continue

        roi_s = s[y:y+hh, x:x+ww]
        roi_e = edge_mask[y:y+hh, x:x+ww]

        mean_s = float(np.mean(roi_s))
        if mean_s < 30:
            continue

        edge_ratio = float(np.count_nonzero(roi_e)) / (ww * hh + 1e-6)
        if edge_ratio < 0.008:
            continue

        extent = area / (ww * hh + 1e-6)
        if extent > 0.90:
            continue

        found += 1
        cv2.rectangle(output, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
        cv2.putText(output, "Graffiti?", (x, max(0, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return output, combined, sat_mask, edge_mask, found


if __name__ == "__main__":
    image_path = "images/duvar.jpg"

    out, combined, sat_mask, edge_mask, found = detect_graffiti_traditional_v2(image_path)

    cv2.imwrite("dbg_sat_mask.png", sat_mask)
    cv2.imwrite("dbg_edge_mask.png", edge_mask)
    cv2.imwrite("out_mask.png", combined)
    cv2.imwrite("out_result.png", out)

    print("Bitti 🌿")
    print("Bulunan aday sayısı:", found)
    print("Kaydedilen dosyalar:")
    print("- out_result.png")
    print("- out_mask.png")
    print("- dbg_sat_mask.png")
    print("- dbg_edge_mask.png")
