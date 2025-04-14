# Samsung ve iPhone Telefon Algılama Projesi

Bu proje, dersde verilen ödev kapsamında geliştirildi. Samsung ve iPhone telefonlarını kamera üzerinden algılayabilen bir yapay zeka modeli oluşturduk.

## Proje Hakkında

Projede YOLO (You Only Look Once) modeli kullanarak Samsung ve iPhone telefonlarını algılayabilen bir uygulama geliştirdik. Telefonları 1-2 metre mesafeden tanıyıp marka isimlerini gösterebiliyor.

## Kullanılan Teknolojiler

- Python 3
- OpenCV
- Ultralytics YOLOv8
- Roboflow (veri etiketleme için)
- Google Colab (model eğitimi için)

## Nasıl Çalışır?

Projemiz 3 temel aşamadan oluşuyor:

1. **Veri Toplama ve Etiketleme**: Samsung ve iPhone telefonlarının fotoğraflarını topladık, Roboflow'da etiketledik
2. **Model Eğitimi**: Etiketlediğimiz verileri kullanarak YOLOv8 modelini eğittik
3. **Kamera ile Algılama**: Eğitilmiş modeli kullanarak web kamerası üzerinden gerçek zamanlı algılama yapıyoruz

## Kurulum

Projeyi çalıştırmak için:

```bash
# Gerekli kütüphaneleri kur
pip install ultralytics opencv-python

# Algılama programını çalıştır
python improved_detection.py
```

## Kullanım

Programı çalıştırdıktan sonra, telefonunuzu kameraya 1-2 metre mesafeden gösterin. Program telefonu algılayacak ve ekranda Samsung veya iPhone olarak etiketleyecektir.

Klavye kontrolleri:
- q: Çıkış
- f: Odak modu açık/kapalı
- b: Parlaklık artır
- c: Kontrast artır 
- r: Ayarları sıfırla
- -/+: Güven eşiğini azalt/artır

## Proje Zorlukları

Projeyi yaparken bayağı zorlandığım yerler oldu:
- MacBook kamera sorunlarını çözmek zor oldu
- Doğru miktarda veri toplamak zaman aldı
- Modelin telefonları doğru boyutlarda algılamasını sağlamak için çok uğraştım
- Kamera odaklama konusunda sorunlar yaşadım

