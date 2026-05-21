import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta
import os, warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(PROJECT_DIR, "data"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_DIR, "outputs"), exist_ok=True)

# ============================================================
# SAYFA AYARLARI
# ============================================================
st.set_page_config(
    page_title="Finansal Tahmin Sistemi",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .main-header h1 {
        color: #fff;
        font-size: 2.2rem;
        margin: 0;
        font-weight: 700;
    }
    .main-header p {
        color: #a0a0c0;
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-card h3 {
        color: #8be9fd;
        font-size: 0.85rem;
        margin: 0 0 0.3rem 0;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card p {
        color: #f8f8f2;
        font-size: 1.6rem;
        margin: 0;
        font-weight: 700;
    }

    .prediction-card {
        background: linear-gradient(135deg, #1a1a2e, #0d1b2a);
        border: 1px solid rgba(0,191,255,0.2);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,191,255,0.1);
    }
    .prediction-card h3 {
        color: #8be9fd;
        font-size: 0.8rem;
        margin: 0 0 0.5rem 0;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .prediction-card .price {
        font-size: 2rem;
        margin: 0;
        font-weight: 700;
    }
    .prediction-card .change-up {
        color: #50fa7b;
        font-size: 1rem;
        margin: 0.3rem 0 0 0;
        font-weight: 600;
    }
    .prediction-card .change-down {
        color: #ff5555;
        font-size: 1rem;
        margin: 0.3rem 0 0 0;
        font-weight: 600;
    }

    .signal-card {
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .signal-buy {
        background: linear-gradient(135deg, rgba(80,250,123,0.15), rgba(80,250,123,0.05));
        border: 2px solid #50fa7b;
        color: #50fa7b;
    }
    .signal-sell {
        background: linear-gradient(135deg, rgba(255,85,85,0.15), rgba(255,85,85,0.05));
        border: 2px solid #ff5555;
        color: #ff5555;
    }
    .signal-hold {
        background: linear-gradient(135deg, rgba(255,184,108,0.15), rgba(255,184,108,0.05));
        border: 2px solid #ffb86c;
        color: #ffb86c;
    }

    .confidence-high { color: #50fa7b; }
    .confidence-mid { color: #ffb86c; }
    .confidence-low { color: #ff5555; }

    .model-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .badge-lstm { background: rgba(0,191,255,0.2); color: #00BFFF; border: 1px solid #00BFFF; }
    .badge-rnn { background: rgba(255,99,71,0.2); color: #FF6347; border: 1px solid #FF6347; }
    .badge-dense { background: rgba(50,205,50,0.2); color: #32CD32; border: 1px solid #32CD32; }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }

    .status-box {
        background: rgba(0,191,255,0.1);
        border: 1px solid rgba(0,191,255,0.3);
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        color: #8be9fd;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# VERI CEKME FONKSIYONLARI
# ============================================================
@st.cache_data(show_spinner=False)
def fetch_yahoo(symbol, start, end):
    """Yahoo Finance'den veri cek."""
    df = yf.download(symbol, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data(show_spinner=False)
def fetch_binance(symbol, start, end):
    """Binance API'den veri cek (API key gerekmez)."""
    url = "https://api.binance.com/api/v3/klines"
    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end).timestamp() * 1000)

    all_data = []
    current = start_ms
    while current < end_ms:
        params = {
            "symbol": symbol,
            "interval": "1d",
            "startTime": current,
            "endTime": end_ms,
            "limit": 1000
        }
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            st.error(f"Binance API hatasi: {resp.status_code} - {resp.text}")
            return pd.DataFrame()
        data = resp.json()
        if not data:
            break
        all_data.extend(data)
        current = data[-1][0] + 1

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        "Open Time", "Open", "High", "Low", "Close", "Volume",
        "Close Time", "Quote Volume", "Trades", "Taker Buy Base",
        "Taker Buy Quote", "Ignore"
    ])
    df["Date"] = pd.to_datetime(df["Open Time"], unit='ms')
    df.set_index("Date", inplace=True)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)
    return df


# ============================================================
# TEKNIK INDIKATORLER
# ============================================================
def add_indicators(df):
    """MA, RSI, Volatility ekle."""
    df = df.copy()
    df["MA_7"] = df["Close"].rolling(window=7).mean()
    df["MA_21"] = df["Close"].rolling(window=21).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["Volatility"] = df["Close"].rolling(window=7).std()
    df.dropna(inplace=True)
    return df


# ============================================================
# VERI HAZIRLAMA
# ============================================================
def prepare_data(df, features, seq_len=60, train_ratio=0.8):
    """Olcekleme, pencere ve train/test bolme."""
    data = df[features].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len:i])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)

    split = int(len(X) * train_ratio)
    return X[:split], X[split:], y[:split], y[split:], scaler


# ============================================================
# MODEL OLUSTURMA
# ============================================================
def build_model(model_type, input_shape):
    """LSTM, RNN veya Dense model olustur."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Dropout, Flatten

    model = Sequential()

    if model_type == "LSTM":
        model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0.2))
    elif model_type == "SimpleRNN":
        model.add(SimpleRNN(64, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(SimpleRNN(64, return_sequences=False))
        model.add(Dropout(0.2))
    elif model_type == "Dense":
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))

    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


# ============================================================
# EGITIM
# ============================================================
def train_model(model, X_train, y_train, X_test, y_test, model_name, progress_placeholder):
    """EarlyStopping ile model egit."""
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback

    class StreamlitCallback(Callback):
        def __init__(self, placeholder, name):
            self.placeholder = placeholder
            self.name = name
            self.losses = []
        def on_epoch_end(self, epoch, logs=None):
            self.losses.append(logs)
            self.placeholder.markdown(
                f'<div class="status-box">🔄 <b>{self.name}</b> — '
                f'Epoch {epoch+1} | loss: {logs["loss"]:.6f} | val_loss: {logs["val_loss"]:.6f}</div>',
                unsafe_allow_html=True
            )

    early_stop = EarlyStopping(monitor='val_loss', patience=10,
                                restore_best_weights=True, verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                   patience=5, min_lr=0.00001, verbose=0)
    st_cb = StreamlitCallback(progress_placeholder, model_name)

    history = model.fit(
        X_train, y_train,
        epochs=200, batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, reduce_lr, st_cb],
        verbose=0
    )
    return history


# ============================================================
# METRIKLER
# ============================================================
def calculate_metrics(actual, predicted):
    """MAE, RMSE, MAPE ve Trend Accuracy hesapla."""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    actual_trend = np.where(np.diff(actual) > 0, 1, 0)
    pred_trend = np.where(np.diff(predicted) > 0, 1, 0)
    trend_acc = np.mean(actual_trend == pred_trend) * 100

    return {"MAE ($)": f"${mae:,.2f}", "RMSE ($)": f"${rmse:,.2f}",
            "MAPE (%)": f"%{mape:.2f}", "Trend (%)": f"%{trend_acc:.1f}",
            "_mae": mae, "_rmse": rmse, "_mape": mape, "_trend": trend_acc}


# ============================================================
# TERS OLCEKLEME
# ============================================================
def inverse_scale(values, scaler, n_features):
    """0-1 arasindaki tahmini gercek fiyata cevir."""
    full = np.zeros((len(values), n_features))
    full[:, 0] = values.flatten()
    return scaler.inverse_transform(full)[:, 0]


# ============================================================
# GELECEK TAHMINI (YENİ!)
# ============================================================
def predict_future(model, df, scaler, features, seq_len, n_days=7):
    """
    Recursive gelecek fiyat tahmini.
    Son seq_len günü kullanarak bir sonraki günü tahmin eder,
    sonra tahmini girdiye ekleyerek sonraki günleri tahmin eder.
    Her adımda teknik indikatörler yeniden hesaplanır.
    """
    close_history = df["Close"].values.tolist()

    # Tüm veriyi ölçekle
    data = df[features].values
    scaled = scaler.transform(data)
    current_window = scaled[-seq_len:].copy()

    predictions = []

    for day in range(n_days):
        # Tahmin yap
        input_data = current_window.reshape(1, seq_len, len(features))
        pred_scaled = model.predict(input_data, verbose=0)[0, 0]

        # Gerçek fiyata çevir
        pred_full = np.zeros((1, len(features)))
        pred_full[0, 0] = pred_scaled
        pred_price = scaler.inverse_transform(pred_full)[0, 0]
        predictions.append(pred_price)

        # Close geçmişini güncelle
        close_history.append(pred_price)
        close_series = pd.Series(close_history)

        # İndikatörleri yeniden hesapla
        new_ma7 = close_series.rolling(7).mean().iloc[-1]
        new_ma21 = close_series.rolling(21).mean().iloc[-1]

        delta = close_series.diff()
        gain_s = delta.where(delta > 0, 0)
        loss_s = -delta.where(delta < 0, 0)
        avg_gain = gain_s.rolling(14).mean().iloc[-1]
        avg_loss = loss_s.rolling(14).mean().iloc[-1]
        if avg_loss == 0:
            new_rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            new_rsi = 100 - (100 / (1 + rs))

        new_vol = close_series.rolling(7).std().iloc[-1]

        # Yeni satırı ölçekle
        new_row = np.array([[pred_price, new_ma7, new_ma21, new_rsi, new_vol]])
        new_row_scaled = scaler.transform(new_row)

        # Pencereyi kaydır
        current_window = np.vstack([current_window[1:], new_row_scaled])

    return predictions


def generate_future_dates(last_date, n_days, is_crypto):
    """Gelecek tarihleri üret. Hisse senetleri için hafta sonlarını atla."""
    if is_crypto:
        return pd.date_range(start=last_date + timedelta(days=1), periods=n_days, freq='D')
    else:
        return pd.bdate_range(start=last_date + timedelta(days=1), periods=n_days)


def calculate_day_confidence(day_index, base_mape, model_std, current_price):
    """
    Her tahmin günü için güven oranı hesapla.
    Güven, her gün bileşik belirsizlik nedeniyle azalır.
    """
    base_confidence = max(10, min(95, 100 - base_mape))
    decay = 0.88 ** day_index  # Her gün ~%12 azalma

    # Model uyumu faktörü
    if current_price > 0 and model_std > 0:
        relative_std = (model_std / current_price) * 100
        agreement_factor = max(0.5, 1 - relative_std * 0.5)
    else:
        agreement_factor = 0.85

    confidence = base_confidence * decay * agreement_factor
    return min(95, max(5, confidence))


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Ayarlar")

    # Veri Kaynağı
    source = st.radio("📡 Veri Kaynağı", ["Yahoo Finance", "Binance"],
                       help="Yahoo: hisse+kripto | Binance: sadece kripto")

    # Sembol
    if source == "Yahoo Finance":
        presets = ["BTC-USD", "ETH-USD", "AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "SOL-USD"]
        symbol = st.selectbox("💹 Sembol", presets, index=0)
        custom = st.text_input("veya özel sembol gir:", placeholder="GOOGL")
        if custom.strip():
            symbol = custom.strip().upper()
    else:
        presets = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
        symbol = st.selectbox("💹 Sembol", presets, index=0)
        custom = st.text_input("veya özel sembol gir:", placeholder="AVAXUSDT")
        if custom.strip():
            symbol = custom.strip().upper()

    st.markdown("---")

    # Tarih
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("📅 Başlangıç", value=pd.Timestamp("2020-01-01"))
    with col2:
        end_date = st.date_input("📅 Bitiş", value=pd.Timestamp.today())

    st.markdown("---")

    # Model Seçimi
    st.markdown("### 🧠 Modeller")
    use_lstm = st.checkbox("LSTM", value=True)
    use_rnn = st.checkbox("Simple RNN", value=True)
    use_dense = st.checkbox("Dense (Baseline)", value=True)

    selected_models = []
    if use_lstm: selected_models.append("LSTM")
    if use_rnn: selected_models.append("SimpleRNN")
    if use_dense: selected_models.append("Dense")

    st.markdown("---")

    # Parametreler
    with st.expander("🔧 Gelişmiş Parametreler"):
        seq_len = st.slider("Pencere uzunluğu (gün)", 20, 120, 60)
        train_ratio = st.slider("Eğitim oranı (%)", 60, 90, 80) / 100
        forecast_days = st.slider("Tahmin süresi (gün)", 1, 14, 7)

    st.markdown("---")

    # Başla
    run_btn = st.button("🚀 Analiz & Tahmin Başlat", type="primary",
                         disabled=len(selected_models) == 0)


# ============================================================
# ANA ALAN
# ============================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>📈 Finansal Piyasa Tahmin Sistemi</h1>
    <p>LSTM / RNN / Dense model karşılaştırması ile fiyat tahmini ve gelecek öngörüsü</p>
</div>
""", unsafe_allow_html=True)

if not run_btn:
    # Bekleme ekranı
    st.info("👈 Sol panelden ayarlarını yap ve **Analiz & Tahmin Başlat** butonuna bas.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Veri Kaynağı</h3>
            <p>Yahoo & Binance</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Model Seçenekleri</h3>
            <p>LSTM / RNN / Dense</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Gelecek Tahmini</h3>
            <p>7 Güne Kadar</p>
        </div>""", unsafe_allow_html=True)

    st.stop()


# ============================================================
# ANALİZ AKIŞI
# ============================================================

# 1. Veri çek
with st.spinner(f"📡 {source}'dan {symbol} verisi indiriliyor..."):
    if source == "Yahoo Finance":
        df = fetch_yahoo(symbol, str(start_date), str(end_date))
    else:
        df = fetch_binance(symbol, str(start_date), str(end_date))

if df.empty or len(df) < 100:
    st.error(f"❌ '{symbol}' için yeterli veri bulunamadı. Sembol veya tarih aralığını kontrol et.")
    st.stop()

# Kripto mu kontrol et
is_crypto = source == "Binance" or any(
    x in symbol.upper() for x in ["-USD", "USDT", "BTC", "ETH"]
)

# Güncel fiyat bilgisi (indikatörlerden önce al)
current_price = float(df["Close"].iloc[-1])
prev_price = float(df["Close"].iloc[-2])
price_change = current_price - prev_price
price_change_pct = (price_change / prev_price) * 100

# 2. İndikatörler
df = add_indicators(df)
features = ["Close", "MA_7", "MA_21", "RSI", "Volatility"]

current_rsi = float(df["RSI"].iloc[-1])
current_vol = float(df["Volatility"].iloc[-1])

# ============================================================
# 📊 PİYASA DURUMU
# ============================================================
st.markdown("### 📊 Piyasa Durumu")
mc1, mc2, mc3, mc4 = st.columns(4)
with mc1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Güncel Fiyat</h3>
        <p>${current_price:,.2f}</p>
    </div>""", unsafe_allow_html=True)
with mc2:
    chg_class = "change-up" if price_change >= 0 else "change-down"
    chg_icon = "📈" if price_change >= 0 else "📉"
    chg_color = "#50fa7b" if price_change >= 0 else "#ff5555"
    st.markdown(f"""
    <div class="metric-card">
        <h3>Son Değişim</h3>
        <p style="color:{chg_color}">{chg_icon} %{price_change_pct:+.2f}</p>
    </div>""", unsafe_allow_html=True)
with mc3:
    rsi_status = "Aşırı Alım" if current_rsi > 70 else "Aşırı Satım" if current_rsi < 30 else "Nötr"
    rsi_color = "#ff5555" if current_rsi > 70 else "#50fa7b" if current_rsi < 30 else "#ffb86c"
    st.markdown(f"""
    <div class="metric-card">
        <h3>RSI ({rsi_status})</h3>
        <p style="color:{rsi_color}">{current_rsi:.1f}</p>
    </div>""", unsafe_allow_html=True)
with mc4:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Volatilite (7g)</h3>
        <p>${current_vol:,.0f}</p>
    </div>""", unsafe_allow_html=True)

st.markdown("---")
st.success(f"✅ {len(df)} günlük veri yüklendi | {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")

# Veri önizleme
with st.expander("📊 Ham Veri Önizleme", expanded=False):
    st.dataframe(df.tail(20), use_container_width=True)

# 3. Veri hazırlama
X_train, X_test, y_train, y_test, scaler = prepare_data(df, features, seq_len, train_ratio)

st.markdown(f"**Eğitim:** {len(X_train)} örnek | **Test:** {len(X_test)} örnek | "
            f"**Pencere:** {seq_len} gün | **Özellik:** {len(features)}")

# 4. Modelleri eğit
st.markdown("---")
st.markdown("### 🧠 Model Eğitimi")

results = {}
model_colors = {"LSTM": "#00BFFF", "SimpleRNN": "#FF6347", "Dense": "#32CD32"}
badge_classes = {"LSTM": "badge-lstm", "SimpleRNN": "badge-rnn", "Dense": "badge-dense"}

for model_name in selected_models:
    st.markdown(f'<span class="model-badge {badge_classes[model_name]}">{model_name}</span>',
                unsafe_allow_html=True)
    progress_ph = st.empty()

    with st.spinner(f"{model_name} eğitiliyor..."):
        model = build_model(model_name, (X_train.shape[1], X_train.shape[2]))
        history = train_model(model, X_train, y_train, X_test, y_test, model_name, progress_ph)

    # Test tahmini
    preds_scaled = model.predict(X_test, verbose=0)
    predicted = inverse_scale(preds_scaled, scaler, len(features))
    actual = inverse_scale(y_test, scaler, len(features))

    # Metrikler
    metrics = calculate_metrics(actual, predicted)

    # Gelecek tahmini
    with st.spinner(f"{model_name} ile gelecek tahmini yapılıyor..."):
        future_preds = predict_future(model, df, scaler, features, seq_len, forecast_days)

    results[model_name] = {
        "predicted": predicted,
        "actual": actual,
        "history": history,
        "metrics": metrics,
        "model": model,
        "future": future_preds
    }

    progress_ph.markdown(
        f'<div class="status-box">✅ <b>{model_name}</b> tamamlandı — '
        f'{len(history.history["loss"])} epoch | '
        f'val_loss: {min(history.history["val_loss"]):.6f}</div>',
        unsafe_allow_html=True
    )


# ============================================================
# 🔮 GELECEK FİYAT TAHMİNLERİ
# ============================================================
st.markdown("---")
st.markdown("### 🔮 Gelecek Fiyat Tahminleri")

# Gelecek tarihleri oluştur
last_date = df.index[-1]
future_dates = generate_future_dates(last_date, forecast_days, is_crypto)

# En iyi model
best_model = min(results.keys(), key=lambda k: results[k]["metrics"]["_rmse"])
best_mape = results[best_model]["metrics"]["_mape"]

# Ensemble (tüm modellerin ortalaması)
all_futures = np.array([r["future"] for r in results.values()])
ensemble_future = np.mean(all_futures, axis=0)
ensemble_std = np.std(all_futures, axis=0)

# Yarın ve hafta sonu tahmini
tomorrow_price = ensemble_future[0]
tomorrow_change = ((tomorrow_price - current_price) / current_price) * 100
week_end_price = ensemble_future[-1]
week_change = ((week_end_price - current_price) / current_price) * 100

# Tahmin kartları
pc1, pc2, pc3, pc4 = st.columns(4)
with pc1:
    t_class = "change-up" if tomorrow_change >= 0 else "change-down"
    t_icon = "📈" if tomorrow_change >= 0 else "📉"
    t_color = "#50fa7b" if tomorrow_change >= 0 else "#ff5555"
    st.markdown(f"""
    <div class="prediction-card">
        <h3>Yarın Tahmini</h3>
        <p class="price" style="color:#f8f8f2">${tomorrow_price:,.2f}</p>
        <p class="{t_class}">{t_icon} %{tomorrow_change:+.2f}</p>
    </div>""", unsafe_allow_html=True)

with pc2:
    w_class = "change-up" if week_change >= 0 else "change-down"
    w_icon = "📈" if week_change >= 0 else "📉"
    st.markdown(f"""
    <div class="prediction-card">
        <h3>{forecast_days} Gün Sonra</h3>
        <p class="price" style="color:#f8f8f2">${week_end_price:,.2f}</p>
        <p class="{w_class}">{w_icon} %{week_change:+.2f}</p>
    </div>""", unsafe_allow_html=True)

with pc3:
    # Sinyal belirle
    up_models = sum(1 for r in results.values() if r["future"][-1] > current_price)
    total_models = len(results)

    if up_models == total_models and current_rsi < 70:
        signal = "AL"
        signal_class = "signal-buy"
        signal_icon = "🟢"
    elif up_models == 0 or current_rsi > 80:
        signal = "SAT"
        signal_class = "signal-sell"
        signal_icon = "🔴"
    else:
        signal = "BEKLE"
        signal_class = "signal-hold"
        signal_icon = "🟡"

    st.markdown(f"""
    <div class="prediction-card">
        <h3>Piyasa Sinyali</h3>
        <div class="signal-card {signal_class}">
            {signal_icon} {signal}
        </div>
    </div>""", unsafe_allow_html=True)

with pc4:
    agreement = (1 - (ensemble_std[0] / current_price)) * 100
    agreement = min(99, max(10, agreement))
    conf_class = "confidence-high" if agreement > 70 else "confidence-mid" if agreement > 50 else "confidence-low"
    st.markdown(f"""
    <div class="prediction-card">
        <h3>Model Uyumu</h3>
        <p class="price {conf_class}">%{agreement:.0f}</p>
        <p style="color:#a0a0c0; font-size:0.85rem; margin-top:0.3rem">{up_models}/{total_models} model yükseliş</p>
    </div>""", unsafe_allow_html=True)

# Gün gün tahmin tablosu
st.markdown("#### 📅 Gün Gün Tahmin Detayları")

table_data = []
for i in range(forecast_days):
    day_preds = {name: r["future"][i] for name, r in results.items()}
    day_ensemble = ensemble_future[i]
    day_std = ensemble_std[i]
    day_change_pct = ((day_ensemble - current_price) / current_price) * 100

    confidence = calculate_day_confidence(i, best_mape, day_std, current_price)

    if day_ensemble > current_price:
        direction = "📈 Yükseliş"
    else:
        direction = "📉 Düşüş"

    row = {
        "📅 Tarih": future_dates[i].strftime("%Y-%m-%d (%A)") if i < len(future_dates) else f"Gün {i+1}",
    }
    for name, p in day_preds.items():
        row[f"{name} ($)"] = f"${p:,.2f}"
    row["Ensemble ($)"] = f"${day_ensemble:,.2f}"
    row["Değişim (%)"] = f"%{day_change_pct:+.2f}"
    row["Güven (%)"] = f"%{confidence:.0f}"
    row["Yön"] = direction

    table_data.append(row)

forecast_df = pd.DataFrame(table_data)
st.dataframe(forecast_df, use_container_width=True, hide_index=True)


# Gelecek tahmin grafiği
st.markdown("#### 📈 Gelecek Tahmin Grafiği")

fig_future = go.Figure()

# Son 30 günlük gerçek fiyat (bağlam için)
context_days = min(30, len(df))
context_dates = df.index[-context_days:]
context_prices = df["Close"].values[-context_days:]

# Gerçek fiyat çizgisi
fig_future.add_trace(go.Scatter(
    x=context_dates, y=context_prices,
    mode='lines', name='Gerçek Fiyat',
    line=dict(color='white', width=2)
))

# Her modelin tahmini
for name, r in results.items():
    fig_future.add_trace(go.Scatter(
        x=future_dates, y=r["future"],
        mode='lines+markers', name=f'{name} Tahmini',
        line=dict(color=model_colors[name], width=1.5, dash='dash'),
        marker=dict(size=6)
    ))

# Ensemble tahmini
fig_future.add_trace(go.Scatter(
    x=future_dates, y=ensemble_future,
    mode='lines+markers', name='Ensemble Tahmini',
    line=dict(color='#bd93f9', width=2.5),
    marker=dict(size=8, symbol='diamond')
))

# Güven bandı
upper_band = ensemble_future + ensemble_std
lower_band = ensemble_future - ensemble_std
fig_future.add_trace(go.Scatter(
    x=list(future_dates) + list(future_dates[::-1]),
    y=list(upper_band) + list(lower_band[::-1]),
    fill='toself', fillcolor='rgba(189,147,249,0.1)',
    line=dict(color='rgba(189,147,249,0)'),
    name='Güven Aralığı',
    showlegend=True
))

# Bugünü ayıran dikey çizgi (add_vline yerine manual shape kullanıyoruz)
last_date_str = str(last_date)
fig_future.add_shape(
    type="line", x0=last_date_str, x1=last_date_str, y0=0, y1=1,
    yref="paper", line=dict(color="rgba(255,255,255,0.3)", dash="dot")
)
fig_future.add_annotation(
    x=last_date_str, y=1, yref="paper",
    text="Bugün", showarrow=False,
    font=dict(color="rgba(255,255,255,0.5)", size=11),
    yshift=10
)

fig_future.update_layout(
    template="plotly_dark",
    title=f"{symbol} — Gelecek {forecast_days} Gün Tahmin",
    xaxis_title="Tarih",
    yaxis_title="Fiyat (USD)",
    height=500,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, t=60, b=20)
)
st.plotly_chart(fig_future, use_container_width=True)


# Sinyal açıklaması
st.markdown("#### 💡 Sinyal Açıklaması")
if signal == "AL":
    st.success(
        f"🟢 **AL Sinyali** — Tüm modeller ({total_models}/{total_models}) yükseliş öngörüyor "
        f"ve RSI ({current_rsi:.0f}) aşırı alım bölgesinde değil."
    )
elif signal == "SAT":
    st.error(
        f"🔴 **SAT Sinyali** — Modeller düşüş öngörüyor "
        f"veya RSI ({current_rsi:.0f}) aşırı alım bölgesinde."
    )
else:
    st.warning(
        f"🟡 **BEKLE Sinyali** — Modeller arasında görüş ayrılığı var "
        f"({up_models}/{total_models} yükseliş). Net bir sinyal oluşmadı."
    )

st.caption("⚠️ Bu sinyaller yatırım tavsiyesi değildir. Model tahminleri geçmiş veriye dayalıdır ve gelecek performansı garanti etmez.")


# ============================================================
# 📋 MODEL KARŞILAŞTIRMA (TEST DÖNEMİ)
# ============================================================
st.markdown("---")
st.markdown("### 📋 Model Karşılaştırma (Test Dönemi)")

# Metrik tablosu
metric_df = pd.DataFrame({
    name: {"MAE ($)": r["metrics"]["MAE ($)"],
           "RMSE ($)": r["metrics"]["RMSE ($)"],
           "MAPE (%)": r["metrics"]["MAPE (%)"],
           "Trend Doğruluğu (%)": r["metrics"]["Trend (%)"]}
    for name, r in results.items()
})
st.table(metric_df)

# En iyi model
st.success(f"🏆 En düşük RMSE ile en iyi model: **{best_model}**")


# ============================================================
# 📈 GERÇEK FİYAT VS TAHMİNLER (GERÇEK TARİHLİ)
# ============================================================
st.markdown("---")
st.markdown("### 📈 Gerçek Fiyat vs Tahminler (Test Dönemi)")

# Test dönemi tarihlerini hesapla
n_test = len(list(results.values())[0]["actual"])
split_idx = int((len(df) - seq_len) * train_ratio)
test_start = seq_len + split_idx
test_dates = df.index[test_start:test_start + n_test]

# Grafik
fig = go.Figure()
actual_prices = list(results.values())[0]["actual"]
fig.add_trace(go.Scatter(x=test_dates, y=actual_prices, mode='lines', name='Gerçek Fiyat',
                          line=dict(color='white', width=2)))

for name, r in results.items():
    fig.add_trace(go.Scatter(x=test_dates, y=r["predicted"], mode='lines', name=f'{name} Tahmini',
                              line=dict(color=model_colors[name], width=1.5)))

fig.update_layout(
    template="plotly_dark",
    title=f"{symbol} — Gerçek Fiyat vs Model Tahminleri",
    xaxis_title="Tarih",
    yaxis_title="Fiyat (USD)",
    height=500,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, t=60, b=20)
)
st.plotly_chart(fig, use_container_width=True)


# Loss grafiği
st.markdown("### 📉 Eğitim Loss Karşılaştırması")
fig2 = make_subplots(rows=1, cols=2, subplot_titles=("Train Loss", "Validation Loss"))

for name, r in results.items():
    fig2.add_trace(go.Scatter(y=r["history"].history['loss'], mode='lines',
                               name=f'{name}', line=dict(color=model_colors[name])),
                   row=1, col=1)
    fig2.add_trace(go.Scatter(y=r["history"].history['val_loss'], mode='lines',
                               name=f'{name} val', line=dict(color=model_colors[name], dash='dash'),
                               showlegend=False),
                   row=1, col=2)

fig2.update_layout(template="plotly_dark", height=350,
                    margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig2, use_container_width=True)


# Trend grafiği
st.markdown("### 🔄 Trend Doğruluğu Karşılaştırması")
trend_data = {}
for name, r in results.items():
    trend_data[name] = r["metrics"]["_trend"]

fig3 = go.Figure(data=[
    go.Bar(x=list(trend_data.keys()), y=list(trend_data.values()),
           marker_color=[model_colors[k] for k in trend_data.keys()],
           text=[f"%{v:.1f}" for v in trend_data.values()],
           textposition='auto')
])
fig3.update_layout(template="plotly_dark", title="Trend Tahmini Doğruluğu",
                    yaxis_title="Doğruluk (%)", height=350,
                    margin=dict(l=20, r=20, t=40, b=20))
fig3.add_hline(y=50, line_dash="dash", line_color="gray",
               annotation_text="Rastgele (%50)")
st.plotly_chart(fig3, use_container_width=True)


# ============================================================
# 📥 SONUÇLARI İNDİR
# ============================================================
st.markdown("---")
st.markdown("### 📥 Sonuçları İndir")

download_data = {
    "Tarih": [d.strftime("%Y-%m-%d") for d in future_dates],
}
for name, r in results.items():
    download_data[f"{name} Tahmin ($)"] = [f"{p:.2f}" for p in r["future"]]
download_data["Ensemble Tahmin ($)"] = [f"{p:.2f}" for p in ensemble_future]
download_data["Değişim (%)"] = [
    f"{((ensemble_future[i] - current_price) / current_price) * 100:+.2f}"
    for i in range(forecast_days)
]
download_data["Güven (%)"] = [
    f"{calculate_day_confidence(i, best_mape, ensemble_std[i], current_price):.0f}"
    for i in range(forecast_days)
]

download_df = pd.DataFrame(download_data)
csv = download_df.to_csv(index=False)
st.download_button(
    label="📥 Tahmin Sonuçlarını CSV Olarak İndir",
    data=csv,
    file_name=f"{symbol}_tahmin_{pd.Timestamp.today().strftime('%Y%m%d')}.csv",
    mime="text/csv"
)


# ============================================================
# MODEL KAYDET
# ============================================================
best_result = results[best_model]
save_path = os.path.join(PROJECT_DIR, "models", f"{best_model.lower()}_model.keras")
best_result["model"].save(save_path)
st.info(f"💾 En iyi model kaydedildi: `{save_path}`")


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#666; font-size:0.85rem;">'
    'Finansal Piyasa Tahmin Sistemi v3.0 | LSTM & RNN & Dense | '
    'Gelecek Tahmini & Ensemble | Cem Yildiz</p>',
    unsafe_allow_html=True
)
