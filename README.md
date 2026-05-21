# 📈 Derin Öğrenme ile Finansal Tahmin Sistemi (LSTM - RNN - Dense)

[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Streamlit App](https://static.streamlit.io/badge_gradient.svg)](https://cemyildiz-ai-finance.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Bu proje; **LSTM (Long Short-Term Memory)**, **SimpleRNN** ve **Dense** (Standart Çok Katmanlı Yapay Sinir Ağı) mimarilerini kullanarak finansal varlıkların (Bitcoin, Ethereum, Hisse Senetleri vb.) geçmiş fiyat verilerinden geleceğe yönelik fiyat tahminleri gerçekleştiren gelişmiş bir makine öğrenmesi uygulamasıdır. 

Uygulama, hem teknik analiz indikatörlerini (RSI, MA, Volatilite) hesaplar hem de eğitilen modelleri birleştirerek (Ensemble Model) **geleceğe yönelik 7 günlük recursive (özyinelemeli) tahmin** üretir ve her gün için güven analizleri sunar.

🔗 **Canlı Uygulama Adresi:** [cemyildiz-ai-finance.streamlit.app](https://cemyildiz-ai-finance.streamlit.app/)

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

## 📂 Proje Yapisi

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

## 💻 Kurulum ve Yerel Çalıştırma

Projeyi bilgisayarınızda yerel olarak çalıştırmak için aşağıdaki adımları takip edebilirsiniz.

### 1. Depoyu Klonlayın
```bash
git clone https://github.com/cemyildizcy/stock-price-prediction-lstm.git
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

## 📊 Model Mimarileri

| Model | Yapı | Parametreler |
|-------|------|-----------|
| **LSTM** | LSTM(64) → Dropout → LSTM(64) → Dropout → Dense(32) → Dense(1) | ~53K |
| **SimpleRNN** | RNN(64) → Dropout → RNN(64) → Dropout → Dense(32) → Dense(1) | ~20K |
| **Dense** | Flatten → Dense(128) → Dropout → Dense(64) → Dropout → Dense(32) → Dense(1) | ~500K+ |

---

## 📝 Lisans

Bu proje **MIT** lisansı altında lisanslanmıştır. Detaylar için lisans standartlarına göz atabilirsiniz.

---
**Yazar:** Cem Yıldız
