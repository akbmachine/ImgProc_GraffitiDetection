import cv2
import numpy as np

def detect_graffiti_traditional(image_path):
    # 1. Görüntüyü Oku
    img = cv2.imread(image_path)
    if img is None:
        print("Hata: Görüntü yüklenemedi. Dosya yolunu kontrol edin.")
        return

    # Orijinal görüntüyü kopyala (sonuçları bunun üzerine çizeceğiz)
    output_img = img.copy()

    # 2. Gürültü Azaltma (Blur)
    # Tuğla desenlerini yumuşatmak için güçlü bir bulanıklaştırma kullanıyoruz.
    # (21, 21) kernel boyutu resmin çözünürlüğüne göre artırılabilir/azaltılabilir.
    blurred = cv2.GaussianBlur(img, (21, 21), 0)

    # 3. Renk Uzayı Dönüşümü (BGR -> HSV)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 4. Saturation (Doygunluk) Kanalını Ayırma
    # H (Renk özü), S (Doygunluk), V (Parlaklık). 
    # Grafitiler genelde duvardan daha 'doygun' renklere sahiptir.
    h_channel, s_channel, v_channel = cv2.split(hsv)

    # 5. Eşikleme (Thresholding)
    # Doygunluğu belli bir seviyenin (örn: 50) üzerinde olan pikselleri beyaz yap.
    # Bu değer duvarın rengine göre (50-80 arası) değiştirilebilir.
    # Düz tuğla duvarlar genelde düşük doygunluktadır.
    _, mask = cv2.threshold(s_channel, 60, 255, cv2.THRESH_BINARY)

    # 6. Morfolojik İşlemler (Çok Önemli Adım)
    # Maskede oluşan küçük noktaları temizle ve grafiti parçalarını birleştir.
    kernel = np.ones((15, 15), np.uint8)
    
    # Erode + Dilate (Opening): Küçük gürültüleri (tek tük tuğla lekelerini) siler.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Dilate + Erode (Closing): Grafiti harflerini birbirine bağlar, delikleri kapatır.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 7. Kontur Bulma (Sınırları Çizme)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found_graffiti = False
    
    for contour in contours:
        # Alan Filtrelemesi:
        # Çok küçük alanları (hatalı tespitleri) görmezden gel.
        area = cv2.contourArea(contour)
        if area > 500:  # Bu değer resim boyutuna göre ayarlanmalı (örn: 1000 - 5000 arası)
            found_graffiti = True
            
            # Dikdörtgen içine al
            x, y, w, h = cv2.boundingRect(contour)
            
            # Tespit edilen bölgeyi yeşil kutu içine al
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            # Üzerine yazı yaz
            cv2.putText(output_img, "Grafiti", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 8. Sonuçları Göster
    # Maske halini de gösteriyorum ki algoritmanın neyi görüp neyi görmediğini anlayın.
    # Ekran sığması için yeniden boyutlandırma (opsiyonel)
    img_s = cv2.resize(img, (600, 400))
    mask_s = cv2.resize(mask, (600, 400))
    out_s = cv2.resize(output_img, (600, 400))

    cv2.imshow('1. Orijinal', img_s)
    cv2.imshow('2. Islenmis Maske (Algoritmanin Gordugu)', mask_s)
    cv2.imshow('3. Sonuc', out_s)

    print("İşlem tamamlandı. Çıkmak için bir tuşa basın.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect_graffiti_traditional('test-image.png')
