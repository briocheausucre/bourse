import os
import time
import math
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf

# =========================
# Liste des tickers Euro Stoxx 50 format yahoo
# =========================
EUROSTOXX50_COMPONENTS: List[Dict[str, str]] = [
    # France
    {"ticker": "MC.PA",   "company": "LVMH"},
    {"ticker": "OR.PA",   "company": "L'Oreal"},
    {"ticker": "AIR.PA",  "company": "Airbus"},
    {"ticker": "AI.PA",   "company": "Air Liquide"},
    {"ticker": "CS.PA",   "company": "AXA"},
    {"ticker": "DG.PA",   "company": "Vinci"},
    {"ticker": "BN.PA",   "company": "Danone"},
    {"ticker": "EL.PA",   "company": "EssilorLuxottica"},
    {"ticker": "RMS.PA",  "company": "Hermes"},
    {"ticker": "SU.PA",   "company": "Schneider Electric"},
    {"ticker": "SAF.PA",  "company": "Safran"},
    {"ticker": "SAN.PA",  "company": "Sanofi"},
    {"ticker": "TTE.PA",  "company": "TotalEnergies"},
    {"ticker": "KER.PA",  "company": "Kering"},
    {"ticker": "CAP.PA",  "company": "Capgemini"},
    {"ticker": "SGO.PA",  "company": "Saint-Gobain"},

    # Allemagne
    {"ticker": "SAP.DE",  "company": "SAP"},
    {"ticker": "SIE.DE",  "company": "Siemens"},
    {"ticker": "ENR.DE",  "company": "Siemens Energy"},
    {"ticker": "IFX.DE",  "company": "Infineon"},
    {"ticker": "VOW3.DE", "company": "Volkswagen"},
    {"ticker": "DTE.DE",  "company": "Deutsche Telekom"},
    {"ticker": "ALV.DE",  "company": "Allianz"},
    {"ticker": "MBG.DE",  "company": "Mercedes-Benz"},
    {"ticker": "BMW.DE",  "company": "BMW"},
    {"ticker": "BAYN.DE", "company": "Bayer"},
    {"ticker": "DB1.DE",  "company": "Deutsche Boerse"},
    {"ticker": "MUV2.DE", "company": "Munich Re"},
    {"ticker": "BAS.DE",  "company": "BASF"},
    {"ticker": "ADS.DE",  "company": "Adidas"},
    {"ticker": "RHM.DE",  "company": "Rheinmetall"},
    {"ticker": "DHL.DE",  "company": "Deutsche Post"},

    # Pays-Bas
    {"ticker": "ASML.AS", "company": "ASML"},
    {"ticker": "ADYEN.AS","company": "Adyen"},
    {"ticker": "AD.AS",   "company": "Koninklijke Ahold Delhaize"},
    {"ticker": "PRX.AS",  "company": "Prosus"},
    {"ticker": "WKL.AS",  "company": "Wolters Kluwer"},
    {"ticker": "INGA.AS", "company": "ING Groep"},

    # Espagne
    {"ticker": "SAN.MC",  "company": "Banco Santander"},
    {"ticker": "BBVA.MC", "company": "BBVA"},
    {"ticker": "ITX.MC",  "company": "Inditex"},
    {"ticker": "IBE.MC",  "company": "Iberdrola"},
    {"ticker": "TEF.MC",  "company": "Telefonica"},

    # Italie
    {"ticker": "ENEL.MI", "company": "Enel"},
    {"ticker": "ENI.MI",  "company": "ENI"},
    {"ticker": "ISP.MI",  "company": "Intesa Sanpaolo"},
    {"ticker": "UCG.MI",  "company": "UniCredit"},
    {"ticker": "RACE.MI", "company": "Ferrari"},
    {"ticker": "STMMI.MI",  "company": "STMicroelectronics"},

    # Belgique
    {"ticker": "ABI.BR",  "company": "AB InBev"},
    {"ticker": "ARGX.BR", "company": "argenx"},

    # Finlande
    {"ticker": "NDA-FI.HE", "company": "Nordea Bank"}
]

TRADING_DAYS_PER_YEAR = 252
LAST_MONTH_TRADING_DAYS = 21  # ~1 mois de bourse


# =========================
# Fonctions utilitaires
# =========================

def compute_max_drawdown(price_series: pd.Series) -> float:
    """Max drawdown (ratio) sur une série de prix."""
    s = price_series.dropna()
    if s.empty:
        return np.nan
    roll_max = s.cummax()
    drawdown = s / roll_max - 1.0
    return float(drawdown.min())


def compute_beta(stock_returns: pd.Series, index_returns: pd.Series) -> float:
    """Bêta = cov(rs, ri) / var(ri)."""
    # Harmoniser les index de temps pour éviter "tz-naive vs tz-aware"
    sr = stock_returns.copy()
    ir = index_returns.copy()

    if isinstance(sr.index, pd.DatetimeIndex):
        if sr.index.tz is not None:
            sr.index = sr.index.tz_convert(None)
    if isinstance(ir.index, pd.DatetimeIndex):
        if ir.index.tz is not None:
            ir.index = ir.index.tz_convert(None)

    df = pd.concat([sr, ir], axis=1).dropna()
    if len(df) < 60:
        return np.nan

    rs = df.iloc[:, 0]
    ri = df.iloc[:, 1]
    var_ri = np.var(ri, ddof=1)
    if var_ri <= 0 or np.isnan(var_ri):
        return np.nan
    cov = np.cov(rs, ri, ddof=1)[0, 1]
    return float(cov / var_ri)


def get_history_for_ticker(ticker: str, period: str = "400d") -> pd.DataFrame:
    """
    Récupère l'historique d'un ticker via yfinance.Ticker(...).history,
    avec ajustement des prix et des dividendes.
    """
    t = yf.Ticker(ticker)
    # actions=True pour avoir la colonne "Dividends"
    hist = t.history(period=period, interval="1d", auto_adjust=True)
    if not hist.empty:
        hist = hist.copy()
        hist.index = pd.to_datetime(hist.index)
        if isinstance(hist.index, pd.DatetimeIndex) and hist.index.tz is not None:
            hist.index = hist.index.tz_convert(None)
    return hist


def compute_metrics_for_ticker(
    ticker: str,
    company: str,
    index_returns: pd.Series
) -> Dict[str, float]:
    """
    Calcule les 10 métriques demandées pour un ticker donné.
    Retourne un dict {colonne: valeur}.
    """
    print(f"Processing {ticker}...")

    try:
        hist = get_history_for_ticker(ticker, period="400d")
    except Exception as e:
        print(f"  [ERROR] Historique introuvable pour {ticker}: {e}")
        hist = pd.DataFrame()

    if hist.empty or "Close" not in hist.columns or "Volume" not in hist.columns:
        print(f"  [WARN] Pas de données suffisantes pour {ticker}, remplissage NaN.")
        return {
            "ticker": ticker,
            "company": company,
            "market_cap_eur": np.nan,
            "momentum_12_1": np.nan,
            "return_6m": np.nan,
            "return_3m": np.nan,
            "vol_60d_ann": np.nan,
            "beta_1y_vs_stoxx50": np.nan,
            "max_drawdown_1y": np.nan,
            "adv_60d_shares": np.nan,
            "adv_60d_eur": np.nan,
            "dividend_yield_ttm": np.nan,
        }

    close = hist["Close"].astype(float).dropna()
    volume = hist["Volume"].astype(float).dropna()
    dividends = hist["Dividends"].astype(float) if "Dividends" in hist.columns else pd.Series(0.0, index=hist.index)

    if close.empty:
        print(f"  [WARN] Série de prix vide pour {ticker}.")
        return {
            "ticker": ticker,
            "company": company,
            "market_cap_eur": np.nan,
            "momentum_12_1": np.nan,
            "return_6m": np.nan,
            "return_3m": np.nan,
            "vol_60d_ann": np.nan,
            "beta_1y_vs_stoxx50": np.nan,
            "max_drawdown_1y": np.nan,
            "adv_60d_shares": np.nan,
            "adv_60d_eur": np.nan,
            "dividend_yield_ttm": np.nan,
        }

    last_price = float(close.iloc[-1])
    daily_returns = close.pct_change()

    # 1) market_cap_eur
    market_cap_eur = np.nan
    try:
        t = yf.Ticker(ticker)
        fi = getattr(t, "fast_info", None)
        if fi is not None:
            # fast_info est un objet, on accède aux attributs
            market_cap_eur = float(getattr(fi, "market_cap", np.nan))
        # petit délai pour ne pas spammer Yahoo
        time.sleep(0.2)
    except Exception as e:
        print(f"  [WARN] Impossible de récupérer la market cap pour {ticker}: {e}")
        market_cap_eur = np.nan

    # 2) momentum_12_1 : perf de (t-252 à t-21)
    if len(close) >= TRADING_DAYS_PER_YEAR:
        try:
            p_t_21 = close.iloc[-(LAST_MONTH_TRADING_DAYS + 1)]
            p_t_252 = close.iloc[-TRADING_DAYS_PER_YEAR]
            momentum_12_1 = float(p_t_21 / p_t_252 - 1.0)
        except Exception:
            momentum_12_1 = np.nan
    else:
        momentum_12_1 = np.nan

    # 3) return_6m (~126 jours ouvrés)
    def period_return(prices: pd.Series, days: int) -> float:
        prices = prices.dropna()
        if len(prices) <= days:
            return np.nan
        try:
            return float(prices.iloc[-1] / prices.iloc[-days] - 1.0)
        except Exception:
            return np.nan

    return_6m = period_return(close, 126)

    # 4) return_3m (~63 jours ouvrés)
    return_3m = period_return(close, 63)

    # 5) vol_60d_ann
    vol_60d_ann = np.nan
    r60 = daily_returns.dropna().iloc[-60:]
    if len(r60) >= 20:
        vol_60d_ann = float(r60.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))

    # 6) beta_1y_vs_stoxx50
    beta_1y_vs_stoxx50 = np.nan
    if not index_returns.empty:
        beta_1y_vs_stoxx50 = compute_beta(daily_returns, index_returns)

    # 7) max_drawdown_1y
    px_for_dd = close if len(close) <= TRADING_DAYS_PER_YEAR else close.iloc[-TRADING_DAYS_PER_YEAR:]
    max_dd_1y = compute_max_drawdown(px_for_dd)

    # 8) adv_60d_shares et 9) adv_60d_eur
    adv_60d_shares = np.nan
    adv_60d_eur = np.nan
    if len(volume) >= 60:
        adv_60d_shares = float(volume.iloc[-60:].mean())
        adv_60d_eur = float(adv_60d_shares * last_price)

    # 10) dividend_yield_ttm
    dividend_yield_ttm = np.nan
    if not dividends.empty and (dividends != 0).any():
        last_day = dividends.index.max()
        last_365 = dividends[dividends.index >= (last_day - pd.Timedelta(days=365))]
        total_div_12m = float(last_365.sum()) if not last_365.empty else 0.0
        if last_price > 0:
            dividend_yield_ttm = total_div_12m / last_price

    return {
        "ticker": ticker,
        "company": company,
        "market_cap_eur": market_cap_eur,
        "momentum_12_1": momentum_12_1,
        "return_6m": return_6m,
        "return_3m": return_3m,
        "vol_60d_ann": vol_60d_ann,
        "beta_1y_vs_stoxx50": beta_1y_vs_stoxx50,
        "max_drawdown_1y": max_dd_1y,
        "adv_60d_shares": adv_60d_shares,
        "adv_60d_eur": adv_60d_eur,
        "dividend_yield_ttm": dividend_yield_ttm,
    }


def main():
    # Création du dossier data/
    os.makedirs("data", exist_ok=True)

    # 1) Historique de l'indice Euro Stoxx 50 pour le bêta
    print("Downloading Euro Stoxx 50 index (^STOXX50E) history...")
    try:
        idx_hist = yf.download("^STOXX50E", period="400d", interval="1d", auto_adjust=True, progress=False)
        if idx_hist.empty:
            print("  [WARN] Impossible de récupérer l'historique de ^STOXX50E, bêta sera NaN.")
            index_returns = pd.Series(dtype=float)
        else:
            index_returns = idx_hist["Close"].pct_change()
        # Normaliser l'index temporel pour enlever la timezone éventuelle
        if isinstance(idx_hist.index, pd.DatetimeIndex) and idx_hist.index.tz is not None:
            idx_hist.index = idx_hist.index.tz_convert(None)
        index_returns = idx_hist["Close"].pct_change()
    except Exception as e:
        print(f"  [ERROR] Indice ^STOXX50E indisponible: {e}")
        index_returns = pd.Series(dtype=float)

    # 2) Boucle sur les composants Euro Stoxx 50
    rows = []
    for comp in EUROSTOXX50_COMPONENTS:
        tk = comp["ticker"]
        name = comp["company"]
        metrics = compute_metrics_for_ticker(tk, name, index_returns)
        rows.append(metrics)
        # petit délai pour limiter les risques de rate limit
        time.sleep(0.3)

    # 3) DataFrame + CSV
    df = pd.DataFrame(rows)
    df = df.sort_values(["company", "ticker"], na_position="last").reset_index(drop=True)

    output_path = os.path.join("data", "metrics.csv")
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n✅ Fichier écrit : {output_path}")
    print(f"{len(df)} lignes enregistrées.")


if __name__ == "__main__":
    main()