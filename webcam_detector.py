# İyileştirilmiş Telefon Algılama - Hassas Boyutlar ve Marka İsimleri
import cv2
import time
import numpy as np
from ultralytics import YOLO

def improve_detection(model_path, camera_index=1):
    """
    Doğru boyutlandırma ve marka isimleri ile geliştirilmiş telefon algılama
    
    Args:
        model_path: Eğitilmiş YOLO modeli yolu
        camera_index: Kamera indeksi (iPhone için genellikle 1)
    """
    print("\n" + "="*50)
    print("İyileştirilmiş Telefon Algılama")
    print("="*50)
    
    # Model yükleme
    try:
        print("\nYOLO modeli yükleniyor...")
        model = YOLO(model_path)
        print("✅ Model başarıyla yüklendi!")
        
        # Model sınıflarını yazdır
        print(f"Model sınıfları: {model.names}")
    except Exception as e:
        print(f"❌ Model yüklenirken hata oluştu: {e}")
        return
    
    # Sınıf isimleri
    try:
        # Model sınıflarını al, bulamazsa manuel tanımla
        class_names = model.names
        if not class_names or len(class_names) < 2:
            raise ValueError("Model sınıfları bulunamadı")
    except:
        # Manuel sınıf isimleri
        class_names = {0: 'Samsung', 1: 'iPhone'}
        print(f"Manuel sınıf isimleri: {class_names}")
    
    # Kamera bağlantısı
    print(f"\nKamera bağlanıyor (indeks: {camera_index})...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"❌ Kamera indeksi {camera_index} açılamadı!")
        return
    
    # Kamera ayarlarını optimize et
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Otomatik odak aç
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Daha yüksek çözünürlük
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Kamera özelliklerini al
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✅ Kamera başarıyla başlatıldı! Çözünürlük: {width}x{height}")
    
    # Görüntü işleme ayarları
    brightness = 0
    contrast = 1.0
    focus_area = None  # Odaklanma bölgesi
    show_focus = False  # Odak bölgesini göster/gizle
    
    # YOLO parametreleri
    conf_threshold = 0.45  # Güven eşiği
    nms_iou = 0.5  # IoU eşiği (büyük çakışan kutuları azaltır)
    
    # Renk tanımları
    colors = {
        'Samsung': (0, 180, 0),    # Yeşil
        'iPhone': (0, 0, 200),     # Kırmızı
        'default': (200, 100, 0)   # Turuncu
    }
    
    print("\n📱 Algılama başlatıldı! Kontroller:")
    print("- Çıkmak için 'q'")
    print("- Güven eşiğini azaltmak için '-', artırmak için '+'")
    print("- Parlaklık artırmak için 'b'")
    print("- Kontrast artırmak için 'c'")
    print("- Ayarları sıfırlamak için 'r'")
    print("- Odak bölgesini göster/gizle için 'f'")
    
    # Ana döngü
    try:
        while True:
            # Kameradan görüntü al
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Kamera görüntüsü alınamadı!")
                break
            
            # Görüntü işleme (kontrast ve parlaklık)
            processed_frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
            
            # Algılama için kopyala
            detection_frame = processed_frame.copy()
            
            # Odaklanma bölgesi varsa, o alanı işaretle
            if focus_area and show_focus:
                x, y, w, h = focus_area
                cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
                detection_frame = processed_frame[y:y+h, x:x+w]
            
            # YOLO ile algılama
            results = model.predict(
                detection_frame, 
                conf=conf_threshold,
                iou=nms_iou
            )
            
            # Algılama başarılı mı kontrol et
            if results and len(results) > 0:
                result = results[0]  # İlk sonucu al
                
                # Algılanan nesneleri işaretle
                for box in result.boxes:
                    # Bounding box koordinatları
                    coords = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, coords)
                    
                    # Odaklanma bölgesi varsa koordinatları düzelt
                    if focus_area and show_focus:
                        fx, fy, _, _ = focus_area
                        x1, y1 = x1 + fx, y1 + fy
                        x2, y2 = x2 + fx, y2 + fy
                    
                    # Sınıf ve güven skorunu al
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Sınıf adını al
                    class_name = class_names.get(class_id, f"Sınıf {class_id}")
                    
                    # Rengi seç
                    color = colors.get(class_name, colors['default'])
                    
                    # Bounding box boyutunu düzelt (çok büyük kutuları küçült)
                    box_w, box_h = x2 - x1, y2 - y1
                    
                    # Box düzeltme (çok büyükse küçült)
                    if box_w > width * 0.8 or box_h > height * 0.8:
                        # Box çok büyük, merkezi koruyarak küçült
                        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                        new_w, new_h = int(box_w * 0.7), int(box_h * 0.7)
                        x1 = max(0, center_x - new_w // 2)
                        y1 = max(0, center_y - new_h // 2)
                        x2 = min(width, center_x + new_w // 2)
                        y2 = min(height, center_y + new_h // 2)
                    
                    # Bounding box çiz
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Etiket metni
                    label = f"{class_name}: {conf:.2f}"
                    
                    # Etiket arka planı ve metni
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(processed_frame, (x1, y1-25), (x1+text_w+10, y1), color, -1)
                    cv2.putText(
                        processed_frame, 
                        label, 
                        (x1+5, y1-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (255, 255, 255), 
                        2
                    )
                    
                    # Nesneye odaklanma bölgesi belirle (ilk nesne için)
                    if focus_area is None:
                        # Nesnenin etrafında daha geniş bir odak alanı oluştur
                        padding = 50  # Nesnenin etrafında piksel olarak boşluk
                        fx = max(0, x1 - padding)
                        fy = max(0, y1 - padding)
                        fw = min(width - fx, x2 - x1 + padding * 2)
                        fh = min(height - fy, y2 - y1 + padding * 2)
                        focus_area = (fx, fy, fw, fh)
            
            # Ekranda bilgiler
            cv2.putText(
                processed_frame, 
                f"Güven Eşiği: {conf_threshold:.2f}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
            
            # Parlaklık/kontrast bilgisi
            cv2.putText(
                processed_frame, 
                f"Par: {brightness} | Kon: {contrast:.1f}", 
                (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
            
            # Odak modu
            focus_mode = "ODAK: AÇIK" if show_focus else "ODAK: KAPALI"
            cv2.putText(
                processed_frame, 
                focus_mode, 
                (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
            
            # Görüntüyü göster
            cv2.imshow("Telefon Algılama", processed_frame)
            
            # Tuş kontrolü
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Uygulama kapatılıyor...")
                break
            elif key == ord('-'):  # Güven eşiğini azalt
                conf_threshold = max(0.1, conf_threshold - 0.05)
                print(f"Güven eşiği: {conf_threshold:.2f}")
            elif key == ord('+') or key == ord('='):  # Güven eşiğini artır
                conf_threshold = min(0.95, conf_threshold + 0.05)
                print(f"Güven eşiği: {conf_threshold:.2f}")
            elif key == ord('b'):  # Parlaklık artır
                brightness += 10
                print(f"Parlaklık: {brightness}")
            elif key == ord('c'):  # Kontrast artır
                contrast += 0.1
                print(f"Kontrast: {contrast:.1f}")
            elif key == ord('r'):  # Ayarları sıfırla
                brightness = 0
                contrast = 1.0
                conf_threshold = 0.45
                focus_area = None
                print("Ayarlar sıfırlandı")
            elif key == ord('f'):  # Odak bölgesini göster/gizle
                show_focus = not show_focus
                if not show_focus:
                    focus_area = None  # Odak bölgesini sıfırla
                print(f"Odak modu: {'Açık' if show_focus else 'Kapalı'}")
    
    except KeyboardInterrupt:
        print("\nProgram kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\nBir hata oluştu: {e}")
    finally:
        # Kaynakları serbest bırak
        cap.release()
        cv2.destroyAllWindows()
        print("\nProgram sonlandırıldı.")
        
if __name__ == "__main__":
    # Eğitilmiş model dosya yolu
    model_path = "best.pt"
    
    # Kamera indeksi
    camera_index = 0  # iPhone kamerası için değiştirilebilir
    
    # Uygulamayı başlat
    improve_detection(model_path, camera_index)