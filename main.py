# ============================================================
# Finansal Piyasa Tahmin Sistemi - LSTM ile Fiyat Tahmini
# ============================================================
# Bu proje BTC-USD fiyat verilerini kullanarak gelecekteki
# fiyat hareketini tahmin eden bir LSTM modeli geliştirir.
# ============================================================

# ============================================================
# AŞAMA 1: VERİ ÇEKME
# ============================================================
# yfinance kütüphanesi Yahoo Finance'den geçmiş fiyat
# verilerini çekmemizi sağlar. API key falan gerekmez,
# doğrudan sembol ve tarih aralığı vererek kullanırız.
# ============================================================

import yfinance as yf   # Yahoo Finance API
import pandas as pd      # Tablo/veri işleme
import os

# --- Proje Dizini ---
# Scriptin nerede calistirildigina bakmaksizin, dosya yollarini
# scriptin kendi konumuna gore belirleriz.
# __file__ = bu script dosyasinin yolu
# os.path.dirname = klasor yolunu alir
# os.path.abspath = mutlak yola cevirir
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Alt klasorlerin var oldugundan emin ol
os.makedirs(os.path.join(PROJECT_DIR, "data"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_DIR, "outputs"), exist_ok=True)

# --- Ayarlar ---
# BTC-USD: Bitcoin'in ABD Doları cinsinden fiyatı
# Kripto piyasası 7/24 çalıştığı için tatil günü boşluğu olmaz,
# bu da veriyi daha düzenli yapar.
symbol = "BTC-USD"
start_date = "2020-01-01"
end_date = "2026-05-20"   # Bugune kadar guncel veri

# --- Veriyi İndir ---
# yf.download() fonksiyonu belirtilen sembolün geçmiş fiyat
# verilerini indirir. Dönen veri bir pandas DataFrame'dir.
# Kolonlar: Open, High, Low, Close, Volume
#   Open   = Günün açılış fiyatı
#   High   = Gün içindeki en yüksek fiyat
#   Low    = Gün içindeki en düşük fiyat
#   Close  = Günün kapanış fiyatı (biz bunu tahmin edeceğiz)
#   Volume = O gün yapılan işlem hacmi
print(f"'{symbol}' verisi indiriliyor...")
print(f"Tarih araligi: {start_date} - {end_date}")
print("-" * 50)

df = yf.download(symbol, start=start_date, end=end_date)

# --- Veriyi Kaydet ---
# Ham veriyi CSV dosyasına kaydediyoruz ki her seferinde
# internetten indirmek zorunda kalmayalım.
df.to_csv(os.path.join(PROJECT_DIR, "data", "raw_data.csv"))

# --- Veriyi İncele ---
# İlk 5 satır: verinin başlangıcı
print("\n[VERI] Ilk 5 satir:")
print(df.head())

# Son 5 satır: verinin sonu
print("\n[VERI] Son 5 satir:")
print(df.tail())

# Veri hakkında genel bilgi
print(f"\n[VERI] Toplam {len(df)} gunluk veri indirildi.")
print(f"[VERI] Kolonlar: {list(df.columns)}")
print(f"[VERI] Veri data/raw_data.csv dosyasina kaydedildi. [OK]")

# ============================================================
# ASAMA 2: VERI TEMIZLEME VE TEKNIK INDIKATORLER
# ============================================================
# Ham veri dogrudan modele verilemez. Once temizlememiz,
# sonra "teknik indikatorler" ekleyerek zenginlestirmemiz gerekir.
#
# Teknik indikator nedir?
# Gecmis fiyat verilerinden matematiksel formüllerle turetilen
# yardimci sinyallerdir. Trader'lar bunlari kullanarak
# "fiyat yukselecek mi, dusecek mi?" sorusuna cevap arar.
# Biz de LSTM modeline bu ekstra bilgileri vererek
# tahmin kalitesini artirmayi hedefliyoruz.
# ============================================================

print("\n" + "=" * 50)
print("ASAMA 2: Veri Temizleme ve Teknik Indikatorler")
print("=" * 50)

# --- MultiIndex Duzeltme ---
# yfinance bazen kolonlari MultiIndex (cok katmanli) olarak dondurur.
# Ornegin: ('Close', 'BTC-USD') seklinde.
# Biz sadece 'Close', 'High' gibi basit isimler istiyoruz.
# droplevel(1) ikinci katmani siler.
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)
    print("[TEMIZLIK] MultiIndex duzeltildi -> basit kolon isimleri")

# --- Eksik Verileri Temizle ---
# Bazi gunlerde veri eksik olabilir (NaN = Not a Number).
# Eksik veri modeli yaniltir, bu yuzden o satirlari siliyoruz.
# dropna() = NaN iceren tum satirlari sil
eksik_oncesi = len(df)
df.dropna(inplace=True)
eksik_sonrasi = len(df)
print(f"[TEMIZLIK] Eksik satirlar silindi: {eksik_oncesi - eksik_sonrasi} satir")

# --- Tarih Indeksi ---
# DataFrame'in indeksinin tarih tipinde oldugundan emin olalim.
# Bu, zaman serisinde siralama ve grafik icin onemli.
df.index = pd.to_datetime(df.index)
print(f"[TEMIZLIK] Tarih indeksi ayarlandi")
print(f"[TEMIZLIK] Kalan veri: {len(df)} satir")

# ============================================================
# TEKNIK INDIKATOR 1: MOVING AVERAGE (Hareketli Ortalama)
# ============================================================
# Moving Average, belirli bir gun sayisindaki fiyatlarin
# ortalamasini alir. Bu, fiyattaki kisa sureli dalgalanmalari
# yumusatarak genel trendi gormemizi saglar.
#
# MA_7  = Son 7 gunun ortalamasi  (kisa donem trendi)
# MA_21 = Son 21 gunun ortalamasi (uzun donem trendi)
#
# Ornek:
#   Fiyatlar: 100, 102, 98, 105, 103, 101, 99
#   MA_7 = (100+102+98+105+103+101+99) / 7 = 101.14
#
# Neden onemli?
#   - MA_7 > MA_21 ise: Kisa donem trend yukari (alis sinyali)
#   - MA_7 < MA_21 ise: Kisa donem trend asagi (satis sinyali)
# ============================================================

print("\n--- Teknik Indikatorler Hesaplaniyor ---")

# rolling(window=7) = son 7 satiri al
# .mean() = ortalamasini hesapla
df["MA_7"] = df["Close"].rolling(window=7).mean()
df["MA_21"] = df["Close"].rolling(window=21).mean()
print("[INDIKATOR] MA_7 ve MA_21 hesaplandi (Hareketli Ortalama)")

# ============================================================
# TEKNIK INDIKATOR 2: RSI (Relative Strength Index)
# ============================================================
# RSI, fiyatin "asiri alim" veya "asiri satim" bolgesinde
# olup olmadigini olcer. Degeri 0-100 arasindadir.
#
#   RSI > 70 = Asiri alim (fiyat cok yukseldi, dusebilir)
#   RSI < 30 = Asiri satim (fiyat cok dustu, yukelebilir)
#   RSI ~ 50 = Notr bolge
#
# Nasil hesaplanir?
#   1. Her gunun fiyat degisimini bul (bugun - dun)
#   2. Yukselen gunlerin ortalamasini al (avg_gain)
#   3. Dusen gunlerin ortalamasini al (avg_loss)
#   4. RS = avg_gain / avg_loss
#   5. RSI = 100 - (100 / (1 + RS))
#
# 14 gunluk pencere standart RSI periyodudur.
# ============================================================

def calculate_rsi(data, window=14):
    """RSI (Relative Strength Index) hesapla."""
    # diff() = ardisik gunler arasindaki fark
    # Ornek: [100, 105, 102] -> [NaN, +5, -3]
    delta = data.diff()

    # Pozitif degisimler (kazanclar)
    # where(delta > 0, 0) = delta pozitifse onu al, degilse 0 yaz
    gain = delta.where(delta > 0, 0)

    # Negatif degisimler (kayiplar) - mutlak deger alinir
    loss = -delta.where(delta < 0, 0)

    # Son 14 gunun ortalama kazanc ve kaybi
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    # RS = Ortalama Kazanc / Ortalama Kayip
    rs = avg_gain / avg_loss

    # RSI formulu
    rsi = 100 - (100 / (1 + rs))

    return rsi

df["RSI"] = calculate_rsi(df["Close"])
print("[INDIKATOR] RSI hesaplandi (Guclu Gosterge Endeksi)")

# ============================================================
# TEKNIK INDIKATOR 3: VOLATILITY (Oynaklık)
# ============================================================
# Volatility, fiyatin ne kadar dalgalandigini olcer.
# Yuksek volatilite = fiyat cok oynuyor (riskli)
# Dusuk volatilite = fiyat stabil (sakin piyasa)
#
# Hesaplama: Son 7 gunun standart sapmasi
# Standart sapma buyukse, fiyatlar ortalamadan cok sapiyordur.
# ============================================================

df["Volatility"] = df["Close"].rolling(window=7).std()
print("[INDIKATOR] Volatility hesaplandi (Oynaklık)")

# --- Son Temizlik ---
# rolling() ve diff() islemleri basta NaN degerler uretir.
# Ornegin MA_21 icin ilk 20 satirda yeterli veri yok -> NaN.
# Bu NaN'leri de siliyoruz.
onceki = len(df)
df.dropna(inplace=True)
print(f"\n[TEMIZLIK] Indikator NaN'leri silindi: {onceki - len(df)} satir")
print(f"[TEMIZLIK] Final veri boyutu: {len(df)} satir")

# --- Sonucu kontrol et ---
print("\n[KONTROL] Kullanilacak kolonlar:")
print(f"  {list(df.columns)}")
print(f"\n[KONTROL] Ornek veri (son 5 satir):")
print(df[["Close", "MA_7", "MA_21", "RSI", "Volatility"]].tail())

print("\nAsama 2 tamamlandi! [OK]")

# ============================================================
# ASAMA 3: VERIYI MODELE HAZIRLAMA
# ============================================================
# LSTM modeli dogrudan fiyat degerleriyle iyi calisamaz.
# Neden? Cunku:
#   - Close ~76,000 iken RSI ~34 gibi cok farkli olceklerde degerler var
#   - Model buyuk sayilara takilir, kucuk ama onemli degerleri gormezden gelir
#
# Cozum: Tum verileri 0-1 arasina cekmek (normalizasyon).
# Boylece model her kolona esit sekilde odaklanabilir.
# ============================================================

import numpy as np
from sklearn.preprocessing import MinMaxScaler

print("\n" + "=" * 50)
print("ASAMA 3: Veriyi Modele Hazirlama")
print("=" * 50)

# --- Adim 3.1: Kullanilacak Kolonlari Sec ---
# Modele sadece bu 5 kolonu verecegiz:
#   Close      = Kapanış fiyati (BUNU TAHMIN EDECEGIZ)
#   MA_7       = 7 gunluk hareketli ortalama
#   MA_21      = 21 gunluk hareketli ortalama
#   RSI        = Guc gostergesi
#   Volatility = Oynaklık
#
# Not: Open, High, Low, Volume kolonlarini kullanmiyoruz
# cunku indikatorler zaten fiyat bilgisini iceriyor.
features = ["Close", "MA_7", "MA_21", "RSI", "Volatility"]
data = df[features]
print(f"[HAZIRLIK] Secilen ozellikler: {features}")
print(f"[HAZIRLIK] Veri boyutu: {data.shape}")  # (satir, kolon)

# --- Adim 3.2: MinMaxScaler ile Olcekleme ---
# MinMaxScaler her kolondaki degerleri 0-1 arasina cevirir.
#
# Formul: X_scaled = (X - X_min) / (X_max - X_min)
#
# Ornek: Close kolonu min=7000, max=100000 ise
#   7000  -> (7000-7000)/(100000-7000)  = 0.0
#   100000 -> (100000-7000)/(100000-7000) = 1.0
#   53500  -> (53500-7000)/(100000-7000)  = 0.5
#
# ONEMLI: scaler objesini sakliyoruz cunku tahminleri
# gercek fiyata geri cevirirken lazim olacak!
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

print(f"[OLCEKLEME] Veri 0-1 arasina olceklendi")
print(f"[OLCEKLEME] Olceklenmis veri ornegi (ilk satir):")
print(f"  Close={scaled_data[0][0]:.4f}, MA_7={scaled_data[0][1]:.4f}, "
      f"MA_21={scaled_data[0][2]:.4f}, RSI={scaled_data[0][3]:.4f}, "
      f"Volatility={scaled_data[0][4]:.4f}")

# --- Adim 3.3: Zaman Penceresi (Sliding Window) Olustur ---
# ============================================================
# LSTM tek bir satir veri ile tahmin yapamaz.
# "Son N gune bakarak bir sonraki gunu tahmin et" mantigi ile calisir.
#
# Biz sequence_length = 60 kullanacagiz, yani:
#   - Modele son 60 gunu veriyoruz
#   - Model 61. gunun fiyatini tahmin ediyor
#
# Gorsel olarak:
#   X[0] = [gun1, gun2, ..., gun60]    -> y[0] = gun61'in Close degeri
#   X[1] = [gun2, gun3, ..., gun61]    -> y[1] = gun62'nin Close degeri
#   X[2] = [gun3, gun4, ..., gun62]    -> y[2] = gun63'un Close degeri
#   ...
#
# Bu "kaydirilmis pencere" (sliding window) yaklasimi,
# modelin zaman icerisindeki PATERNLERI ogrenmesini saglar.
# ============================================================

sequence_length = 60  # Son 60 gun -> 1 sonraki gun

X = []  # Giris verisi (gecmis 60 gun)
y = []  # Hedef (tahmin edilecek fiyat)

for i in range(sequence_length, len(scaled_data)):
    # i=60 icin: X'e 0-59 indekslerini (60 gun) ekle
    X.append(scaled_data[i - sequence_length:i])
    # y'ye i. gunun Close degerini ekle (indeks 0 = Close kolonu)
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

print(f"\n[PENCERE] Zaman penceresi olusturuldu")
print(f"[PENCERE] Pencere uzunlugu: {sequence_length} gun")
print(f"[PENCERE] X shape: {X.shape}")
print(f"  -> {X.shape[0]} ornek, her biri {X.shape[1]} gun, her gun {X.shape[2]} ozellik")
print(f"[PENCERE] y shape: {y.shape}")
print(f"  -> {y.shape[0]} adet tahmin edilecek fiyat degeri")

# --- Adim 3.4: Train-Test Ayirimi ---
# ============================================================
# KRITIK KURAL: Zaman serilerinde veriyi RASTGELE bolmek YANLISTIR!
#
# Neden? Cunku:
#   - Eger rastgele bolersen, model gelecekteki veriyi gorerek
#     gecmisi "tahmin" eder (data leakage / veri sizintisi)
#   - Gercek hayatta gelecegi gormeden tahmin yapman gerekir
#
# Dogru yontem: KRONOLOJIK bolme
#   - Ilk %80 = Egitim (2020 - ~2024)
#   - Son %20 = Test   (~2024 - 2026)
#
# Model 2020-2024 arasini ogrenip, 2024-2026 arasini
# hic gormeden tahmin edecek. Boylece gercekci bir test olur.
# ============================================================

train_size = int(len(X) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

print(f"\n[BOLME] Train-Test ayrildi (kronolojik)")
print(f"[BOLME] Egitim seti : {X_train.shape[0]} ornek ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"[BOLME] Test seti   : {X_test.shape[0]} ornek ({X_test.shape[0]/len(X)*100:.0f}%)")
print(f"[BOLME] X_train shape: {X_train.shape}  (ornek, gun, ozellik)")
print(f"[BOLME] X_test shape : {X_test.shape}")

print("\nAsama 3 tamamlandi! [OK]")

# ============================================================
# ASAMA 4: LSTM MODELI KURMA VE EGITME
# ============================================================
# LSTM (Long Short-Term Memory) nedir?
#
# Normal sinir aglari her girdiyi bagimsiz degerlendirir.
# Ama fiyat tahmini icin "siralama" onemlidir:
#   "Onceki 60 gun ne oldu?" sorusunun cevabi lazim.
#
# LSTM, ozel "hafiza hucreleri" sayesinde gecmisteki
# onemli bilgileri HATIRLAR, gereksizleri UNUTUR.
#
# Mimarimiz:
#   LSTM(64) -> Dropout(0.2) -> LSTM(64) -> Dropout(0.2) -> Dense(32) -> Dense(1)
#
#   LSTM(64)    = 64 nöronlu LSTM katmani (zaman paternlerini ogrenir)
#   Dropout(0.2)= Rastgele %20 noronu kapatir (ezberlemeyi onler)
#   Dense(32)   = 32 nöronlu sıkıştırma katmani
#   Dense(1)    = Tek cikis: tahmin edilen fiyat
# ============================================================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

print("\n" + "=" * 50)
print("ASAMA 4: LSTM Modeli Kurma ve Egitme")
print("=" * 50)

# --- Adim 4.1: Model Mimarisini Olustur ---
# Sequential = katmanlari sirayla ekle
model = Sequential()

# 1. LSTM Katmani (64 noron)
# return_sequences=True: Bir sonraki LSTM katmanina veri aktarir
# input_shape: (60 gun, 5 ozellik) - modele ilk katmanda soylememiz lazim
model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))  # %20 noron rastgele kapatilir

# 2. LSTM Katmani (64 noron)
# return_sequences=False: Son LSTM katmani, artik Dense katmana geciyor
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))

# 3. Dense (Yogun) Katmanlar
# Bilgiyi sikistirip tek bir fiyat tahminine donusturur
model.add(Dense(32, activation='relu'))  # 32 noron, ReLU aktivasyon
model.add(Dense(1))  # Cikis: 1 sayi (tahmin edilen fiyat)

# --- Adim 4.2: Modeli Derle ---
# optimizer='adam': En populer optimizasyon algoritmasi, ogrenme hizini otomatik ayarlar
# loss='mean_squared_error': Tahmin ile gercek arasindaki farkin karesinin ortalamasi
#   MSE kuculdukce tahminler gercege yaklasir
model.compile(optimizer="adam", loss="mean_squared_error")

# Model ozetini goster
print("\n[MODEL] Model mimarisi:")
model.summary()

# --- Adim 4.3: Akilli Durdurma (EarlyStopping) ---
# ============================================================
# Sabit epoch (ornegin 30) yerine AKILLI DURDURMA kullaniyoruz.
#
# Neden? Cunku:
#   - Cok az epoch = model yeterince ogrenemez (underfitting)
#   - Cok fazla epoch = model ezberlemeye baslar (overfitting)
#   - EarlyStopping tam dogru noktada durur!
#
# Nasil calisir?
#   - Her epoch sonunda validation loss'a (val_loss) bakar
#   - Eger val_loss 10 epoch boyunca iyilesmezse DURUR
#   - restore_best_weights=True: En iyi epoch'un agirliklarini geri yukler
#
# ReduceLROnPlateau: Ogrenme hizi azaltma
#   - val_loss 5 epoch boyunca iyilesmezse ogrenme hizini yarisina dusurur
#   - Bu, modelin "platoda sikismasi"ni onler
# ============================================================

early_stop = EarlyStopping(
    monitor='val_loss',       # Neyi izle: validasyon kaybi
    patience=10,              # 10 epoch boyunca iyilesme yoksa dur
    restore_best_weights=True, # En iyi agirliklari geri yukle
    verbose=1                  # Durdugunda bilgilendir
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',       # Neyi izle: validasyon kaybi
    factor=0.5,               # Ogrenme hizini yarisina dusur
    patience=5,               # 5 epoch iyilesme yoksa dusur
    min_lr=0.00001,           # Minimum ogrenme hizi
    verbose=1                  # Dusurdugunde bilgilendir
)

print("\n[EGITIM] Akilli durdurma aktif (patience=10)")
print("[EGITIM] Ogrenme hizi azaltma aktif (patience=5)")
print("[EGITIM] Maksimum epoch: 200 (ama muhtemelen 30-60 arasi duracak)")
print("[EGITIM] Egitim basliyor...\n")

# --- Adim 4.4: Modeli Egit ---
# epochs=200: Maksimum epoch siniri (EarlyStopping daha once durduracak)
# batch_size=32: Her adimda 32 ornek ile ogrenme
# validation_data: Her epoch sonunda test verisiyle performans olcumu
history = model.fit(
    X_train,
    y_train,
    epochs=200,               # Ust sinir (EarlyStopping durduracak)
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr],
    verbose=1                  # Her epoch'u goster
)

# --- Egitim Sonuclari ---
print(f"\n[SONUC] Egitim tamamlandi!")
print(f"[SONUC] Toplam epoch: {len(history.history['loss'])}")
print(f"[SONUC] Son train loss : {history.history['loss'][-1]:.6f}")
print(f"[SONUC] Son val loss   : {history.history['val_loss'][-1]:.6f}")
print(f"[SONUC] En iyi val loss: {min(history.history['val_loss']):.6f} "
      f"(epoch {history.history['val_loss'].index(min(history.history['val_loss']))+1})")

print("\nAsama 4 tamamlandi! [OK]")

# ============================================================
# ASAMA 5: TAHMIN VE GORSELLESTIRME
# ============================================================
# Simdi modelimizi test verisi uzerinde deneyecegiz.
# Adimlar:
#   1. Test verisini modele ver, tahmin al
#   2. Tahminleri 0-1'den gercek fiyata geri cevir
#   3. Gercek vs Tahmin grafigi ciz
#   4. Trend tahmini analizi yap
#   5. Modeli kaydet
# ============================================================

import matplotlib.pyplot as plt

print("\n" + "=" * 50)
print("ASAMA 5: Tahmin ve Gorsellestirme")
print("=" * 50)

# --- Adim 5.1: Test Verisi Uzerinde Tahmin Yap ---
# model.predict() test verisindeki her 60 gunluk pencere icin
# bir sonraki gunun fiyatini tahmin eder.
# Sonuc 0-1 arasinda olceklenmis halde gelir.
print("\n[TAHMIN] Test verisi uzerinde tahmin yapiliyor...")
predictions = model.predict(X_test)
print(f"[TAHMIN] {len(predictions)} adet tahmin uretildi")

# --- Adim 5.2: Tahminleri Gercek Fiyata Cevir ---
# ============================================================
# Problem: Tahminler 0-1 arasinda. Gercek fiyata cevirmeliyiz.
#
# Ama scaler 5 kolonla egitildi (Close, MA_7, MA_21, RSI, Volatility).
# inverse_transform() da 5 kolon bekliyor.
# Bizim elimizde sadece Close tahmini var (1 kolon).
#
# Cozum: Sahte bir 5 kolonlu dizi olustur,
# sadece ilk kolona (Close) gercek/tahmin degerini yaz,
# digerleri 0 olsun. inverse_transform() yap,
# sonra sadece ilk kolonu (Close) al.
# ============================================================

# Tahmin edilen fiyatlar
predicted_full = np.zeros((len(predictions), len(features)))
predicted_full[:, 0] = predictions[:, 0]  # Sadece Close kolonunu doldur

# Gercek fiyatlar
actual_full = np.zeros((len(y_test), len(features)))
actual_full[:, 0] = y_test  # Sadece Close kolonunu doldur

# Ters olcekleme: 0-1 -> gercek fiyat
predicted_prices = scaler.inverse_transform(predicted_full)[:, 0]
actual_prices = scaler.inverse_transform(actual_full)[:, 0]

print(f"\n[FIYAT] Tahminler gercek fiyata cevrildi")
print(f"[FIYAT] Gercek fiyat araligi : ${actual_prices.min():,.0f} - ${actual_prices.max():,.0f}")
print(f"[FIYAT] Tahmin fiyat araligi : ${predicted_prices.min():,.0f} - ${predicted_prices.max():,.0f}")

# --- Adim 5.3: Gercek Fiyat vs Tahmin Grafigi ---
# ============================================================
# Bu projenin en onemli ciktisi!
# Iki cizgiyi ust uste koyarak modelin ne kadar
# basarili tahmin yaptigini gorecegiz.
# ============================================================

plt.style.use('dark_background')  # Koyu tema

fig, ax = plt.subplots(figsize=(14, 7))

# Gercek fiyat cizgisi (mavi)
ax.plot(actual_prices, label="Gercek Fiyat", color="#00BFFF",
        linewidth=1.5, alpha=0.9)

# Tahmin edilen fiyat cizgisi (turuncu)
ax.plot(predicted_prices, label="Tahmin Edilen Fiyat", color="#FF6347",
        linewidth=1.5, alpha=0.9)

ax.set_title("BTC-USD: Gercek Fiyat vs LSTM Tahmini", fontsize=16, fontweight='bold')
ax.set_xlabel("Gun (Test Donemi)", fontsize=12)
ax.set_ylabel("Fiyat (USD)", fontsize=12)
ax.legend(fontsize=12, loc='upper left')
ax.grid(True, alpha=0.3)

# Fiyat farkini gostermek icin hafif dolgu
ax.fill_between(range(len(actual_prices)), actual_prices, predicted_prices,
                alpha=0.1, color='yellow')

plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR, "outputs", "actual_vs_prediction.png"), dpi=150, bbox_inches='tight')
plt.close()
print("[GRAFIK] Gercek vs Tahmin grafigi kaydedildi -> outputs/actual_vs_prediction.png")

# --- Adim 5.4: Trend Tahmini ---
# ============================================================
# Fiyatin tam degerini bilmek kadar, YONUNU bilmek de onemli.
# Trader icin kritik soru: "Fiyat yukselecek mi, dusecek mi?"
#
# Trend analizi:
#   - Gercek fiyat artisi = 1 (yukari), dusus = 0 (asagi)
#   - Tahmin fiyat artisi = 1 (yukari), dusus = 0 (asagi)
#   - Kac tanesinde yon dogru tahmin edildi? = Trend Accuracy
# ============================================================

actual_trend = np.where(np.diff(actual_prices) > 0, 1, 0)
predicted_trend = np.where(np.diff(predicted_prices) > 0, 1, 0)

trend_accuracy = np.mean(actual_trend == predicted_trend)
print(f"\n[TREND] Trend dogrulugu: %{trend_accuracy * 100:.1f}")
print(f"[TREND] ({np.sum(actual_trend == predicted_trend)}/{len(actual_trend)} "
      f"gunde yonu dogru tahmin etti)")

# Trend grafigi
fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(actual_trend, label="Gercek Trend", color="#00BFFF",
        linewidth=1, alpha=0.7)
ax.plot(predicted_trend, label="Tahmin Trend", color="#FF6347",
        linewidth=1, alpha=0.7)

ax.set_title(f"BTC-USD: Trend Tahmini (Dogruluk: %{trend_accuracy*100:.1f})",
             fontsize=16, fontweight='bold')
ax.set_xlabel("Gun (Test Donemi)", fontsize=12)
ax.set_ylabel("Trend: 1=Yukari, 0=Asagi", fontsize=12)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_yticks([0, 1])
ax.set_yticklabels(["Asagi", "Yukari"])

plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR, "outputs", "trend_prediction.png"), dpi=150, bbox_inches='tight')
plt.close()
print("[GRAFIK] Trend tahmini grafigi kaydedildi -> outputs/trend_prediction.png")

# --- Adim 5.5: Egitim Loss Grafigi ---
fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(history.history['loss'], label='Egitim Loss', color='#00BFFF', linewidth=1.5)
ax.plot(history.history['val_loss'], label='Validasyon Loss', color='#FF6347', linewidth=1.5)

best_epoch = history.history['val_loss'].index(min(history.history['val_loss']))
ax.axvline(x=best_epoch, color='#32CD32', linestyle='--', alpha=0.7,
           label=f'En Iyi Epoch ({best_epoch+1})')

ax.set_title("Model Egitim Sureci (Loss)", fontsize=16, fontweight='bold')
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss (MSE)", fontsize=12)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR, "outputs", "training_loss.png"), dpi=150, bbox_inches='tight')
plt.close()
print("[GRAFIK] Egitim loss grafigi kaydedildi -> outputs/training_loss.png")

# --- Adim 5.6: Modeli Kaydet ---
# ============================================================
# Egitilmis modeli .h5 formatinda kaydediyoruz.
# Boylece tekrar egitmeden yukleyip kullanabilirsin.
# ============================================================
model.save(os.path.join(PROJECT_DIR, "models", "lstm_model.h5"))
print(f"\n[MODEL] Model kaydedildi -> models/lstm_model.h5")

# ============================================================
# SONUC OZETI
# ============================================================
print("\n" + "=" * 50)
print("PROJE TAMAMLANDI!")
print("=" * 50)
print(f"Sembol        : {symbol}")
print(f"Veri araligi  : {start_date} - {end_date}")
print(f"Toplam veri   : {len(df)} gun")
print(f"Egitim seti   : {X_train.shape[0]} ornek")
print(f"Test seti     : {X_test.shape[0]} ornek")
print(f"Toplam epoch  : {len(history.history['loss'])}")
print(f"En iyi epoch  : {best_epoch + 1}")
print(f"En iyi val_loss: {min(history.history['val_loss']):.6f}")
print(f"Trend dogrulugu: %{trend_accuracy * 100:.1f}")
print(f"\nCiktilar:")
print(f"  - data/raw_data.csv")
print(f"  - models/lstm_model.h5")
print(f"  - outputs/actual_vs_prediction.png")
print(f"  - outputs/trend_prediction.png")
print(f"  - outputs/training_loss.png")
