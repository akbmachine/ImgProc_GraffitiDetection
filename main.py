import cv2
import numpy as np

def auto_canny(gray, sigma=0.33):
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)

def find_contours_compat(bin_img):
    res = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return res[0] if len(res) == 2 else res[1]

def detect_graffiti_traditional_v2(image_path, show=True):
    img = cv2.imread(image_path)
    if img is None:
        print("Hata: Görüntü yüklenemedi. Dosya yolunu kontrol edin.")
        return

    h, w = img.shape[:2]
    output = img.copy()

    # 1) Daha hafif blur (detay öldürmesin)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # 2) HSV -> Saturation mask (Otsu ile adaptif)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    s_blur = cv2.GaussianBlur(s, (5, 5), 0)
    _, sat_mask = cv2.threshold(s_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3) Edge mask (auto-canny)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    edge_mask = auto_canny(gray, sigma=0.33)

    # 4) Birleştir
    combined = cv2.bitwise_or(sat_mask, edge_mask)

    # 5) Morfoloji (dinamik kernel)
    # Görsel büyükse kernel biraz büyür, küçükse küçülür.
    k = max(3, (min(h, w) // 250) | 1)   # tek sayı olsun
    kernel = np.ones((k, k), np.uint8)

    # Önce azıcık gürültü temizle (open çok hafif!)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    # Sonra parçaları birleştir (close)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 6) Kontur + alan filtresi (resim boyutuna göre)
    contours = find_contours_compat(combined)
    min_area = 0.0015 * (h * w)  # resmin %0.15'i altını ele
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            x, y, ww, hh = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
            cv2.putText(output, "Graffiti", (x, max(0, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if show:
        # Ekrana sığması için ölçekle
        scale = 900 / max(h, w)
        if scale < 1:
            img_s = cv2.resize(img, (int(w*scale), int(h*scale)))
            mask_s = cv2.resize(combined, (int(w*scale), int(h*scale)))
            out_s = cv2.resize(output, (int(w*scale), int(h*scale)))
        else:
            img_s, mask_s, out_s = img, combined, output

        cv2.imshow("1. Orijinal", img_s)
        cv2.imshow("2. Maske", mask_s)
        cv2.imshow("3. Sonuc", out_s)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return output, combined

detect_graffiti_traditional_v2("test-image.png")
