# ============================================================
# MMS-CMAX Sinyal Botu (GitHub Actions - tek tarama modu)
# ============================================================

import os
import json
import ccxt
import pandas as pd
import numpy as np
import requests

# ---- Telegram (GitHub Secrets'tan okunur) ----
TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# ---- Coin ve Zaman Dilimi ----
SEMBOL   = "ADA/USDT"
INTERVAL = "5m"
LIMIT    = 500

# ---- Pine Varsayılan Parametreleri ----
MIN_PERIYOT      = 3
MAX_PERIYOT      = 30
OYNAKLIK_UZUNLUK = 9
FILTRE_UZUNLUK   = 9
MOMENTUM_UZUNLUK = 15
YUZDELIK_UZUNLUK = 50
YUZDELIK_SEVIYE  = 75.0
UST_CARPAN       = 0.7
ALT_CARPAN       = 0.7

STATE_FILE = "state.json"

# ---- MEXC Bağlantısı ----
exchange = ccxt.mexc({
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'},
})

# ------------------------------------------------------------
# VERİ ÇEKME
# ------------------------------------------------------------
def veri_cek_mexc(sembol, interval, limit=1000):
    try:
        ohlcv = exchange.fetch_ohlcv(sembol, timeframe=interval, limit=limit)
        if not ohlcv:
            return pd.DataFrame()
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.astype(float)
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        print(f"❌ MEXC veri hatası: {e}")
        return pd.DataFrame()

# ------------------------------------------------------------
# PINE FONKSİYON KARŞILIKLARI
# ------------------------------------------------------------
def sma(series, length):
    return series.rolling(length).mean()

def stdev_population(series, length):
    return series.rolling(length).std(ddof=0)

def percentile_nearest_rank(series, length, percent):
    def _calc(window):
        w = window[~np.isnan(window)]
        if len(w) == 0:
            return np.nan
        w_sorted = np.sort(w)
        rank = int(np.ceil(percent / 100.0 * len(w_sorted)))
        rank = max(1, min(rank, len(w_sorted)))
        return w_sorted[rank - 1]
    return series.rolling(length).apply(_calc, raw=True)

# ------------------------------------------------------------
# MMS-CMAX HESAPLAMA
# ------------------------------------------------------------
def mms_cmax_hesapla(df):
    kaynak = df["Close"].astype(float)
    n = len(kaynak)

    fiyat_farki  = kaynak.diff()
    oynaklik     = stdev_population(fiyat_farki, OYNAKLIK_UZUNLUK)
    oynaklik_ort = sma(oynaklik, OYNAKLIK_UZUNLUK)

    alt_esik      = oynaklik_ort * 0.25
    ust_esik      = oynaklik_ort * 1.75
    oynaklik_alan = ust_esik - alt_esik

    uyarlanan_periyot = np.where(
        oynaklik_alan > 0,
        np.where(
            oynaklik <= alt_esik, MAX_PERIYOT,
            np.where(
                oynaklik >= ust_esik, MIN_PERIYOT,
                MAX_PERIYOT - (MAX_PERIYOT - MIN_PERIYOT)
                * (oynaklik - alt_esik) / oynaklik_alan
            )
        ),
        MAX_PERIYOT
    )
    uyarlanan_periyot = pd.Series(uyarlanan_periyot, index=kaynak.index)

    katsayi = 2.0 / (uyarlanan_periyot + 1.0)

    uyarlanan_ort = np.full(n, np.nan)
    k_arr = katsayi.values
    y_arr = kaynak.values
    for i in range(n):
        if i == 0 or np.isnan(uyarlanan_ort[i - 1]):
            uyarlanan_ort[i] = y_arr[i]
        else:
            k = k_arr[i] if not np.isnan(k_arr[i]) else 1.0
            uyarlanan_ort[i] = uyarlanan_ort[i - 1] + k * (y_arr[i] - uyarlanan_ort[i - 1])
    uyarlanan_ort = pd.Series(uyarlanan_ort, index=kaynak.index)

    uo = uyarlanan_ort.values
    ehlers_out = np.full(n, np.nan)

    for i in range(n):
        pay, payda = 0.0, 0.0
        for j in range(FILTRE_UZUNLUK):
            idx_now  = i - j
            idx_past = i - j - MOMENTUM_UZUNLUK
            if idx_now < 0 or idx_past < 0:
                continue
            simdiki = uo[idx_now]
            gecmis  = uo[idx_past]
            if np.isnan(simdiki) or np.isnan(gecmis):
                continue
            agirlik = abs(simdiki - gecmis)
            pay    += agirlik * simdiki
            payda  += agirlik
        ehlers_out[i] = pay / payda if payda > 0 else np.nan

    yedek = sma(uyarlanan_ort, FILTRE_UZUNLUK)
    trend = np.where(np.isnan(ehlers_out), yedek.values, ehlers_out)
    trend = pd.Series(trend, index=kaynak.index)

    sapma = (kaynak - trend).abs()
    yuzdelik_mesafe = percentile_nearest_rank(sapma, YUZDELIK_UZUNLUK, YUZDELIK_SEVIYE)

    ust_bant = trend + yuzdelik_mesafe * UST_CARPAN
    alt_bant = trend - yuzdelik_mesafe * ALT_CARPAN

    alis  = (kaynak > ust_bant).values
    satis = (kaynak < alt_bant).values

    yon = np.zeros(n, dtype=int)
    for i in range(n):
        if alis[i]:
            yon[i] = 1
        elif satis[i]:
            yon[i] = -1
        else:
            yon[i] = yon[i - 1] if i > 0 else 0

    yon = pd.Series(yon, index=kaynak.index)
    yon_onceki = yon.shift(1)

    alis_sinyal  = (yon == 1)  & (yon_onceki != 1)
    satis_sinyal = (yon == -1) & (yon_onceki != -1)

    sonuc = df.copy()
    sonuc["trend"]        = trend
    sonuc["ust_bant"]     = ust_bant
    sonuc["alt_bant"]     = alt_bant
    sonuc["yon"]          = yon
    sonuc["alis_sinyal"]  = alis_sinyal
    sonuc["satis_sinyal"] = satis_sinyal
    return sonuc

# ------------------------------------------------------------
# TELEGRAM
# ------------------------------------------------------------
def telegram_gonder(mesaj):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": mesaj,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, data=payload, timeout=10)
        if not r.ok:
            print(f"⚠️ Telegram: {r.status_code} - {r.text}")
        return r.ok
    except Exception as e:
        print(f"❌ Telegram hatası: {e}")
        return False

def mesaj_olustur(df, sembol, interval):
    son = df.iloc[-1]
    fiyat = float(son["Close"])
    zaman = df.index[-1].strftime("%Y-%m-%d %H:%M")
    yon_txt = {1: "🔼 LONG", -1: "🟨 SHORT", 0: "⚪ NÖTR"}.get(int(son["yon"]), "⚪")

    satirlar = [
        f"〽️ {sembol} | {yon_txt}",
        f"<b>Fiyat:</b> {fiyat:.6f}",
        f"{interval} | {zaman} (UTC)",
    ]

    if bool(son["alis_sinyal"]):
        satirlar.append("")
        satirlar.append("🔼 <b>LONG SİNYALİ</b> 🔼")
    elif bool(son["satis_sinyal"]):
        satirlar.append("")
        satirlar.append("🟨 <b>SHORT SİNYALİ</b> 🟨")

    return "\n".join(satirlar)

# ------------------------------------------------------------
# TEK TARAMA
# ------------------------------------------------------------
def tek_tarama():
    df = veri_cek_mexc(SEMBOL, INTERVAL, LIMIT)
    if df.empty or len(df) < (MOMENTUM_UZUNLUK + FILTRE_UZUNLUK + YUZDELIK_UZUNLUK + 5):
        print("Veri yetersiz, atlandı.")
        return

    df = df.iloc[:-1]            # açık (kapanmamış) mumu at
    df = mms_cmax_hesapla(df)

    try:
        with open(STATE_FILE) as f:
            son_zaman = json.load(f).get("son", "")
    except Exception:
        son_zaman = ""

    # Cron gecikirse sinyal kaçmasın diye son 3 kapanmış muma bak
    adaylar = df.tail(3)
    adaylar = adaylar[adaylar["alis_sinyal"] | adaylar["satis_sinyal"]]

    if adaylar.empty:
        print("Yeni sinyal yok.")

    for zaman in adaylar.index:
        z = zaman.strftime("%Y-%m-%d %H:%M")
        if z <= son_zaman:
            continue
        mesaj = mesaj_olustur(df.loc[:zaman], SEMBOL, INTERVAL)
        ok = telegram_gonder(mesaj)
        print(f"{z} sinyal gönderildi: {ok}")
        if ok:
            son_zaman = z

    with open(STATE_FILE, "w") as f:
        json.dump({"son": son_zaman}, f)

if __name__ == "__main__":
    tek_tarama()
