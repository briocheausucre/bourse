import os
import time
import math
import random
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# =========================
# PARAMÈTRES GLOBAUX
# =========================
N_INSTANCES = 1000
TRADING_DAYS_PER_YEAR = 252
LAST_MONTH_TRADING_DAYS = 21   # 1 mois
HORIZON_3M_DAYS = 63           # 3 mois

START_DATE = "2004-01-01"
END_DATE = "2025-01-01"

DOWNLOAD_START_DATE = (pd.Timestamp(START_DATE) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")

HIST_MIN_DATE = pd.Timestamp(START_DATE)
HIST_MAX_DATE = pd.Timestamp("2024-12-31")
COVID_START = pd.Timestamp("2020-01-01")
COVID_END = pd.Timestamp("2021-12-31")

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
    {"ticker": "STMMI.MI","company": "STMicroelectronics"},

    # Belgique
    {"ticker": "ABI.BR",  "company": "AB InBev"},
    {"ticker": "ARGX.BR", "company": "argenx"},

    # Finlande
    {"ticker": "NDA-FI.HE", "company": "Nordea Bank"},
]

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
    """Bêta = cov(rs, ri) / var(ri), en harmonisant les index temporels."""
    sr = stock_returns.copy()
    ir = index_returns.copy()

    if isinstance(sr.index, pd.DatetimeIndex) and sr.index.tz is not None:
        sr.index = sr.index.tz_convert(None)
    if isinstance(ir.index, pd.DatetimeIndex) and ir.index.tz is not None:
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


def download_history(ticker: str) -> pd.DataFrame:
    """
    Télécharge l'historique complet pour un ticker.
    On commence 60 jours avant START_DATE pour avoir assez de données pour le bêta.
    """
    print(f"Downloading history for {ticker}...")
    df = yf.download(
        ticker,
        start=DOWNLOAD_START_DATE,   # <-- CHANGEMENT ICI
        end=END_DATE,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        return df
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    
    if "Dividends" not in df.columns:
        df["Dividends"] = 0.0
    return df


def compute_metrics_at_position(
    ticker: str,
    company: str,
    hist: pd.DataFrame,
    idx_pos: int,
    index_hist: pd.DataFrame,
    shares_outstanding: float,
) -> Dict[str, float]:
    """
    Calcule les 10 métriques à une date donnée (idx_pos dans hist),
    plus le rendement futur à 3 mois (en %).
    """
    date = hist.index[idx_pos]
    hist_up_to = hist.iloc[: idx_pos + 1]
    close = hist_up_to["Close"].astype(float).dropna()
    volume = hist_up_to["Volume"].astype(float).dropna()
    dividends = hist_up_to["Dividends"].astype(float)

    if close.empty or volume.empty:
        return {}

    last_price = float(close.iloc[-1])
    daily_returns = close.pct_change()

    # 1) market_cap_eur
    if not np.isnan(shares_outstanding) and last_price > 0:
        market_cap_eur = float(last_price * shares_outstanding)
    else:
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

    # 6) beta_1y_vs_stoxx50 (1 an de données index jusqu'à date)
    beta_1y_vs_stoxx50 = np.nan
    if not index_hist.empty:
        idx_slice = index_hist.loc[:date]
        idx_returns = idx_slice["Close"].astype(float).pct_change()
        beta_1y_vs_stoxx50 = compute_beta(daily_returns, idx_returns)

    # 7) max_drawdown_1y
    if len(close) <= TRADING_DAYS_PER_YEAR:
        px_for_dd = close
    else:
        px_for_dd = close.iloc[-TRADING_DAYS_PER_YEAR:]
    max_drawdown_1y = compute_max_drawdown(px_for_dd)

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
        "date": date,
        "ticker": ticker,
        "company": company,
        "market_cap_eur": market_cap_eur,
        "momentum_12_1": momentum_12_1,
        "return_6m": return_6m,
        "return_3m": return_3m,
        "vol_60d_ann": vol_60d_ann,
        "beta_1y_vs_stoxx50": beta_1y_vs_stoxx50,
        "max_drawdown_1y": max_drawdown_1y,
        "adv_60d_shares": adv_60d_shares,
        "adv_60d_eur": adv_60d_eur,
        "dividend_yield_ttm": dividend_yield_ttm,
    }


def build_eligible_positions(
    hist: pd.DataFrame,
    min_lookback_days: int,
    forward_days: int,
) -> List[int]:
    """
    Retourne la liste des indices de lignes utilisables pour un ticker :
    - suffisamment de lookback (min_lookback_days)
    - suffisamment de données futures (forward_days)
    - date entre START_DATE et HIST_MAX_DATE
    - hors période Covid (2020-2021 inclus)
    """
    eligible = []
    n = len(hist)
    for idx in range(min_lookback_days, n - forward_days):
        date = hist.index[idx]
        # IMPORTANT : on ne garde que les dates >= START_DATE (HIST_MIN_DATE)
        if date < HIST_MIN_DATE or date > HIST_MAX_DATE:
            continue
        if COVID_START <= date <= COVID_END:
            continue
        eligible.append(idx)
    return eligible


def main():
    os.makedirs("data", exist_ok=True)

    # 1) Télécharger l'indice Euro Stoxx 50 pour le bêta
    print("Downloading Euro Stoxx 50 index (^STOXX50E) history...")
    try:
        idx_hist = yf.download(
            "^STOXX50E",
            start=DOWNLOAD_START_DATE,   # <-- CHANGEMENT ICI
            end=END_DATE,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if idx_hist.empty:
            print("  [WARN] Impossible de récupérer l'historique de ^STOXX50E, bêta sera NaN.")
            idx_hist = pd.DataFrame()
        else:
            idx_hist = idx_hist.copy()
            idx_hist.index = pd.to_datetime(idx_hist.index)
            if isinstance(idx_hist.index, pd.DatetimeIndex) and idx_hist.index.tz is not None:
                idx_hist.index = idx_hist.index.tz_convert(None)
    except Exception as e:
        print(f"  [ERROR] Indice ^STOXX50E indisponible: {e}")
        idx_hist = pd.DataFrame()

    # 2) Pré-télécharger les historiques des tickers et préparer les positions éligibles
    histories: Dict[str, pd.DataFrame] = {}
    eligible_map: Dict[str, List[int]] = {}
    shares_map: Dict[str, float] = {}

    for comp in EUROSTOXX50_COMPONENTS:
        tk = comp["ticker"]
        hist = download_history(tk)
        if hist.empty or "Close" not in hist.columns or "Volume" not in hist.columns:
            print(f"  [WARN] Pas de données pour {tk}, ignoré.")
            continue
        histories[tk] = hist

        eligible_idx = build_eligible_positions(
            hist,
            min_lookback_days=TRADING_DAYS_PER_YEAR,  # pour momentum 12m, beta, DD, etc.
            forward_days=HORIZON_3M_DAYS,
        )
        if not eligible_idx:
            print(f"  [WARN] Aucun point éligible pour {tk}.")
        eligible_map[tk] = eligible_idx

        # récupérer le nombre d'actions en circulation
        try:
            info = yf.Ticker(tk).fast_info
            shares_out = float(getattr(info, "shares", np.nan))
        except Exception:
            shares_out = np.nan
        shares_map[tk] = shares_out

        time.sleep(0.2)

    # Construire la liste globale (ticker, idx_pos) éligible
    all_candidates: List[Tuple[str, int]] = []
    for comp in EUROSTOXX50_COMPONENTS:
        tk = comp["ticker"]
        if tk in eligible_map:
            for idx_pos in eligible_map[tk]:
                all_candidates.append((tk, idx_pos))

    if not all_candidates:
        print("Aucun point éligible trouvé. Abandon.")
        return

    print(f"Total de points éligibles disponibles: {len(all_candidates)}")

    # 3) Tirage aléatoire de N instances
    rows = []
    for n in range(N_INSTANCES):
        tk, idx_pos = random.choice(all_candidates)
        hist = histories[tk]
        comp = next(c for c in EUROSTOXX50_COMPONENTS if c["ticker"] == tk)
        company = comp["company"]
        shares_out = shares_map.get(tk, np.nan)

        metrics = compute_metrics_at_position(
            ticker=tk,
            company=company,
            hist=hist,
            idx_pos=idx_pos,
            index_hist=idx_hist,
            shares_outstanding=shares_out,
        )
        if not metrics:
            continue

        # rendement futur à 3 mois
        if idx_pos + HORIZON_3M_DAYS >= len(hist):
            continue
        price_t = float(hist["Close"].iloc[idx_pos])
        price_t_fut = float(hist["Close"].iloc[idx_pos + HORIZON_3M_DAYS])
        if price_t <= 0:
            continue
        future_return_3m_pct = (price_t_fut / price_t - 1.0) * 100.0

        metrics["future_return_3m_pct"] = future_return_3m_pct
        rows.append(metrics)

    # 4) DataFrame + CSV
    df = pd.DataFrame(rows)
    output_path = os.path.join("data", "dataset_3m.csv")
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\nDataset téléchargé : {output_path}")
    print(f"{len(df)} lignes enregistrées (demandé: {N_INSTANCES}).")
    print(df.head())


if __name__ == "__main__":
    main()