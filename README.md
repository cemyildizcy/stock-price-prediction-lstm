# 📈 Derin Öğrenme ile Finansal Tahmin Sistemi (LSTM - RNN - Dense)

[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Streamlit App](https://static.streamlit.io/badge_gradient.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Bu proje; **LSTM (Long Short-Term Memory)**, **SimpleRNN** ve **Dense** (Standart Çok Katmanlı Yapay Sinir Ağı) mimarilerini kullanarak finansal varlıkların (Bitcoin, Ethereum, Hisse Senetleri vb.) geçmiş fiyat verilerinden geleceğe yönelik fiyat tahminleri gerçekleştiren gelişmiş bir makine öğrenmesi uygulamasıdır. 

Uygulama, hem teknik analiz indikatörlerini (RSI, MA, Volatilite) hesaplar hem de eğitilen modelleri birleştirerek (Ensemble Model) **geleceğe yönelik 7 günlük recursive (özyinelemeli) tahmin** üretir ve her gün için güven analizleri sunar.

---

## ✨ Özellikler

*   **Çoklu Model Desteği & Karşılaştırma**: 
    *   **LSTM**: Uzun vadeli bağımlılıkları ve trendleri yakalamada başarılı.
    *   **SimpleRNN**: Daha hafif ve hızlı eğitilen tekrarlayan sinir ağı.
    *   **Dense**: Hızlı eğitim sunan standart ileri beslemeli yapay sinir ağı.
    *   **Ensemble Tahmin**: Üç modelin tahminlerinin ağırlıklı ortalamasıyla daha stabil sonuçlar.
*   **Çift Veri Kaynağı**: Yahoo Finance API (Hisse senetleri ve pariteler için) ve Binance API (Kripto paralar için).
*   **Teknik Analiz Entegrasyonu**: Girdi olarak sadece kapanış fiyatı değil, 7 ve 21 günlük Hareketli Ortalamalar (MA), Göreceli Güç Endeksi (RSI) ve Tarihsel Volatilite de hesaplanıp modele beslenir.
*   **Özyinelemeli (Recursive) 7 Günlük Gelecek Tahmini**: Son verileri kullanarak gelecek 7 günün fiyatlarını gün gün tahmin eder ve her adımda teknik indikatörleri dinamik olarak yeniden hesaplar.
*   **Güven Oranları & Al/Sat Sinyalleri**: Modellerin yön tahminlerinin tutarlılığına göre gün bazında güven oranları ve otomatik AL / SAT / TUT sinyalleri üretir.
*   **İnteraktif Grafikler**: Plotly kullanılarak hazırlanmış, yakınlaştırılabilir, detaylı geçmiş test ve gelecek tahmin grafikleri (güven bantları ile).
*   **Erken Durdurma (Early Stopping)**: Modelin aşırı öğrenmesini (overfitting) önleyen ve eğitim süresini optimize eden akıllı mekanizma.

---

## 🛠️ Teknolojiler ve Kütüphaneler

*   **Çekirdek**: `Python`
*   **Derin Öğrenme**: `TensorFlow` & `Keras`
*   **Veri Analizi**: `pandas`, `numpy`, `scikit-learn`
*   **Veri Çekme**: `yfinance`, `requests` (Binance REST API)
*   **Görselleştirme**: `plotly`
*   **Arayüz**: `streamlit`

---

## 💻 Kurulum ve Yerel Çalıştırma

Projeyi bilgisayarınızda çalıştırmak için aşağıdaki adımları takip edebilirsiniz.

### 1. Depoyu Klonlayın veya İndirin
```bash
git clone https://github.com/KULLANICI_ADINIZ/stock-price-prediction-lstm.git
cd stock-price-prediction-lstm
```

### 2. Sanal Ortam Oluşturun (Önerilen)
```bash
# Windows için:
python -m venv venv
venv\Scripts\activate

# macOS/Linux için:
python3 -m venv venv
source venv/bin/activate
```

### 3. Gerekli Kütüphaneleri Yükleyin
```bash
pip install -r requirements.txt
```

### 4. Uygulamayı Başlatın
Uygulamayı modern web arayüzü ile çalıştırmak için (Önerilen):
```bash
streamlit run app.py
```
Tarayıcınızda otomatik olarak `http://localhost:8501` adresi açılacaktır.

Terminal üzerinden çalıştırmak isterseniz:
```bash
python main.py
```

---

## 📂 Proje Yapısı

```
stock-price-prediction-lstm/
├── data/                  # İndirilen finansal verilerin tutulduğu klasör (Git'e gönderilmez)
├── models/                # Eğitilen model dosyalarının (.keras) kaydedildiği klasör (Git'e gönderilmez)
├── outputs/               # Grafik çıktılarının ve raporların kaydedildiği klasör
├── app.py                 # Streamlit web arayüzü (Ana uygulama dosyası)
├── main.py                # Terminal tabanlı eğitim ve test scripti
├── requirements.txt       # Gerekli Python kütüphaneleri listesi
├── .gitignore             # Git dışı bırakılacak dosyalar (veri ve modeller)
└── README.md              # Proje açıklama dokümanı
```

---

## 🚀 Projeyi GitHub'a Yükleme Rehberi

Projeyi kendi GitHub hesabınıza eklemek için aşağıdaki adımları terminalde çalıştırın:

1.  **Yerel Git Deposunu İlklendirin:**
    ```bash
    git init
    ```
2.  **Dosyaları Hazırlık Alanına Ekleyin:**
    ```bash
    git add .
    ```
    *(Oluşturduğumuz `.gitignore` dosyası sayesinde `models/` klasöründeki ağır model dosyaları ve veri setleri GitHub'a yüklenmeyecektir.)*

3.  **İlk Commit'i Oluşturun:**
    ```bash
    git commit -m "İlk commit: Finansal Tahmin Sistemi hazır"
    ```

4.  **GitHub'da Yeni Bir Depo (Repository) Oluşturun:**
    *   [GitHub](https://github.com/) hesabınıza giriş yapın.
    *   Sağ üstteki **"+"** simgesine tıklayıp **"New repository"** seçin.
    *   Depo adı olarak `stock-price-prediction-lstm` yazın.
    *   Depoyu oluşturun (README veya .gitignore eklemeden direkt oluşturun).

5.  **Yerel Deponuzu GitHub'a Bağlayın ve Push Edin:**
    ```bash
    git remote add origin https://github.com/KULLANICI_ADINIZ/stock-price-prediction-lstm.git
    git branch -M main
    git push -u origin main
    ```
    *(Not: `KULLANICI_ADINIZ` kısmını kendi GitHub kullanıcı adınızla değiştirmeyi unutmayın!)*

---

## ☁️ Streamlit Cloud Üzerinde Yayına Alma (Deploy)

Uygulamanızı internet üzerinde ücretsiz olarak herkesin erişimine açmak için **Streamlit Community Cloud**'u kullanabilirsiniz:

### Adım 1: Streamlit Cloud Hesabı Açın
1.  [Streamlit Share](https://share.streamlit.io/) sitesine gidin.
2.  **"Sign up with GitHub"** butonuna tıklayarak GitHub hesabınızla giriş yapın ve yetki verin.

### Adım 2: Yeni Uygulama Ekleyin
1.  Giriş yaptıktan sonra sağ üstteki **"New app"** butonuna tıklayın.
2.  Açılan ekranda projenizin bulunduğu GitHub deposunu seçin:
    *   **Repository**: `KULLANICI_ADINIZ/stock-price-prediction-lstm`
    *   **Branch**: `main`
    *   **Main file path**: `app.py`
3.  **"Deploy!"** butonuna tıklayın.

### Adım 3: Yayına Alma Süreci
*   Streamlit Cloud, projenizdeki `requirements.txt` dosyasını otomatik olarak okuyacak ve gerekli tüm kütüphaneleri (TensorFlow, Plotly vb.) kendi sunucularına yükleyecektir.
*   Bu süreç (özellikle TensorFlow kurulumu nedeniyle) ilk seferde **3-5 dakika** sürebilir.
*   Kurulum tamamlandığında uygulamanız size özel bir alt alan adı ile (örn: `https://kullaniciadi-projeadi-app.streamlit.app/`) yayında olacaktır!

---

## 📝 Lisans

Bu proje **MIT** lisansı altında lisanslanmıştır. Detaylar için lisans standartlarına göz atabilirsiniz.

---
**Yazar:** Cem Yıldız
