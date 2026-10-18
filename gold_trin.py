#!/usr/bin/env python3
"""
GoldTRIN - Arms Index (TRIN) breadth data builder for gold miners.

TRIN = (Advancers / Decliners) / (Advancing Volume / Declining Volume)
Volume is relative volume (today / prior 20-day average) so mixed exchanges are comparable.

Signals use the 5-day MA of TRIN ranked against that basket's OWN trailing year, COT-index style:
    MA in the top 10% of the previous 252 sessions    -> potential bottom
    MA in the bottom 10% of the previous 252 sessions -> potential top
The levels are recalculated every day from sessions BEFORE that day, so every historical
signal uses only information that existed at the time (no look-ahead).

Output: docs/data.json  (rendered by index.html)
"""
from __future__ import annotations

import json
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

# ----------------------------------------------------------------- CONFIG
NY = ZoneInfo("America/New_York")
SG = ZoneInfo("Asia/Singapore")
CLOSE_CUTOFF = time(16, 15)      # bars dated today are used only after this ET time
MA_WINDOW = 5
BOTTOM_PCT = 0.90                # MA in its top 10% of the trailing window    -> potential bottom
TOP_PCT = 0.10                   # MA in its bottom 10% of the trailing window -> potential top
LOOKBACK = 252                   # sessions used to rank each day (about 1 year)
LOOKBACK_MIN = 126               # a day gets levels only if at least this many MA values precede it
MIN_COVERAGE = 0.70              # min share of basket with valid data on a day
DISPLAY_DAYS = 252               # sessions shown on the charts (about 12 months)
DOWNLOAD_DAYS = 820              # calendar days downloaded: display + lookback + warm-up
OUTPUT = Path("docs/data.json")

# Yahoo tickers. International listings allowed. Edit freely.
# GDX: from GDX_asof_20261016.xlsx (59 equities). GDX tracks the GDM index.
GDX_BASKET = [
    "NEM", "AEM", "B", "WPM", "FNV", "AU", "GFI", "KGC", "NST.AX", "PAAS", "RGLD", "CDE",
    "AGI", "EQX", "EVN.AX", "PE&OLES.MX", "EDV.L", "IAG", "2259.HK", "HL", "AG", "LUG.TO",
    "FRES.L", "DPM.TO", "BTG", "GMIN.TO", "OGC.TO", "HMY", "BVN", "EGO", "SSRM", "DSV.TO",
    "PRU.AX", "OR", "RMS.AX", "AMMN.JK", "ARTG.V", "1818.HK", "KNT.TO", "MAU.TO", "TXG.TO",
    "CMM.AX", "AUGO", "CGAU", "GMD.AX", "ARIS", "VAU.AX", "AYA.TO", "WDO.TO", "SKE",
    "GGP.AX", "FSM", "RRL.AX", "BRMS.JK", "HOC.L", "WGX.AX", "EXK", "SA", "SVM",
]

# Fidelity Select Gold: from fidelitygoldholdings.pdf (31 Aug 2026). Barrick uses the US listing "B".
FIDELITY_BASKET = [
    "NEM", "CDE", "AEM.TO", "WPM.TO", "B", "FNV.TO", "ELD.TO", "DPM.TO", "GMIN.TO", "IMG.TO",
    "MAU.TO", "EQX.TO", "SKE.TO", "AYA.TO", "SSRM.TO", "KNT.TO", "CG.TO", "K.TO", "FM.TO",
    "WGX.AX", "OGC.TO", "MAI.TO", "WDO.TO", "NG.TO", "HMMC.TO", "FDY.TO", "ARTG.V", "TALA.TO",
    "HBM.TO", "RIO.TO", "AU", "2259.HK", "2899.HK", "GFI",
]

# Your GDM CSV with 11 non-gold rows removed and 8 unmappable names skipped.
GDM_SCREEN = [
    "PAAS", "HOC.L", "KCN.AX", "KGC", "HMY", "GFI", "WPM", "NEM", "FRES.L", "2899.HK", "B", "AEM",
    "RMS.AX", "DRD", "VAU.AX", "FSM", "CDE", "AU", "EXK", "SVM", "EGO", "HL", "FNV", "IAG", "MUX",
    "DPM.TO", "LUG.TO", "ORE.TO", "SA", "BVN", "AG", "DSV.TO", "RRL.AX", "NG", "THM", "GAU",
    "NST.AX", "AGI", "GGD.TO", "WDO.TO", "PAN.JO", "USAS", "OGC.TO", "1818.HK", "CYL.AX", "WAF.AX",
    "BTG", "OBM.AX", "KNT.TO", "OR", "3330.HK", "3939.HK", "WGX.AX", "NEWP", "0340.HK", "1787.HK",
    "MTA", "EQX", "GMIN.TO", "SGD.V", "MAU.TO", "BNZ.AX", "GROY", "CGAU", "NFG", "SKE", "GMD.AX",
    "VZLA", "DC", "IAUX", "TFPM", "HMMC.TO", "TALA.TO", "6693.HK", "AAUC", "2259.HK", "ARIS",
]

# Your GICS file with outdated tickers corrected and acquired names dropped. An approximation.
GICS_BASKET = [
    "NEM", "AEM", "B", "WPM", "FNV", "AU", "GFI", "KGC", "RGLD", "AGI", "PAAS", "LUG.TO", "EQX",
    "IAG", "NG", "EGO", "OR", "BTG", "SA", "HMY", "GROY", "GLDG", "MUX", "CDE", "DRD", "THM",
    "TRX", "VGZ", "AMX.V",
]

PANELS = [
    # To make panel 1 the index basket instead of the wider screen, use GDX_BASKET.
    {"name": "Global Gold Miners (GDM)", "basket": GDM_SCREEN,      "price": None},
    {"name": "VanEck Gold Miners ETF (GDX)",        "basket": GDX_BASKET,      "price": "GDX"},
    {"name": "Fidelity Select Gold (FSAGX)",        "basket": FIDELITY_BASKET, "price": "FSAGX"},
    {"name": "GICS Gold (15104030) ",        "basket": GICS_BASKET,     "price": None},
]


# ------------------------------------------------------------- TIME LOCK
def last_completed_session_date() -> pd.Timestamp:
    """Latest calendar date whose US session is guaranteed finished."""
    now = datetime.now(NY)
    d = now.date()
    if now.weekday() >= 5 or now.time() < CLOSE_CUTOFF:
        d -= timedelta(days=1)
    return pd.Timestamp(d)


# ------------------------------------------------------------------ DATA
def download(tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = (datetime.now(NY) - timedelta(days=DOWNLOAD_DAYS)).strftime("%Y-%m-%d")
    raw = yf.download(sorted(set(tickers)), start=start, interval="1d",
                      auto_adjust=False, progress=False, threads=True)
    close, vol = raw["Close"].copy(), raw["Volume"].copy()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    vol.index = close.index
    cutoff = last_completed_session_date()
    close, vol = close[close.index <= cutoff], vol[vol.index <= cutoff]
    return close, vol


def compute(close: pd.DataFrame, vol: pd.DataFrame, basket: list[str]) -> pd.DataFrame:
    members = [t for t in basket if t in close.columns and close[t].notna().sum() > 20]
    c, v = close[members], vol[members]
    v = v / v.rolling(20, min_periods=10).mean().shift(1)   # relative volume: unit-free across exchanges
    valid = c.notna() & c.shift().notna() & v.notna()
    chg = c.diff()
    up, dn = (chg > 0) & valid, (chg < 0) & valid

    adv, dec = up.sum(axis=1), dn.sum(axis=1)
    advv, decv = v.where(up).sum(axis=1), v.where(dn).sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        trin = (adv / dec.replace(0, np.nan)) / (advv / decv.replace(0, np.nan))
    trin = trin.replace([np.inf, -np.inf], np.nan)
    trin[(valid.sum(axis=1) / max(len(members), 1)) < MIN_COVERAGE] = np.nan

    out = pd.DataFrame({"adv": adv, "dec": dec, "trin": trin})
    out["ma"] = out["trin"].clip(upper=3.0).rolling(MA_WINDOW, min_periods=3).mean()
    out["composite"] = (1 + c.pct_change(fill_method=None).mean(axis=1)).cumprod()
    out.attrs["members"] = members
    out.attrs["missing"] = sorted(set(basket) - set(members))

    # Rolling levels: each day is ranked against the sessions BEFORE it (shift 1) -> no look-ahead.
    roll = out["ma"].rolling(LOOKBACK, min_periods=LOOKBACK_MIN)
    out["hi"] = roll.quantile(BOTTOM_PCT).shift(1).clip(lower=1.05)   # floors keep "bottom" above 1
    out["lo"] = roll.quantile(TOP_PCT).shift(1).clip(upper=0.95)      # and "top" below 1

    state = np.where(out["ma"] >= out["hi"], 1, np.where(out["ma"] <= out["lo"], -1, 0))
    state = pd.Series(state, index=out.index).where(out["ma"].notna(), 0)
    out["state"] = state
    out["entry"] = (state != state.shift()) & (state != 0)
    return out


# ----------------------------------------------------------------- OUTPUT
def to_list(s: pd.Series, nd: int = 4) -> list:
    return [None if pd.isna(x) else round(float(x), nd) for x in s]


def num(x, nd: int = 2):
    return None if pd.isna(x) else round(float(x), nd)


def build() -> None:
    all_tickers = {t for p in PANELS for t in p["basket"]} | {p["price"] for p in PANELS if p["price"]}
    close, vol = download(sorted(all_tickers))
    asof = close.index.max()

    panels, warnings = [], []
    for p in PANELS:
        df = compute(close, vol, p["basket"])
        if df["ma"].dropna().empty:
            warnings.append(f"{p['name']}: not enough data")
            continue
        if df["hi"].dropna().empty:
            warnings.append(f"{p['name']}: not enough history to rank against a trailing year")
            continue
        if df.attrs["missing"]:
            warnings.append(f"{p['name']}: no data for {', '.join(df.attrs['missing'])}")

        price = close[p["price"]] if p["price"] in close.columns else df["composite"]
        df = df.tail(DISPLAY_DAYS)
        price = price.reindex(df.index).ffill()
        if price.notna().any():
            price = price / price.dropna().iloc[0] * 100
        last = df.iloc[-1]   # the latest completed day, even if its MA is blank

        panels.append({
            "name": p["name"],
            "members": len(df.attrs["members"]),
            "levels": {"bottom": num(last["hi"]), "top": num(last["lo"])},
            "last": {
                "trin": num(last["trin"]),
                "ma": num(last["ma"]),
                "adv": int(last["adv"]),
                "dec": int(last["dec"]),
                "state": int(last["state"]),
            },
            "series": {
                "dates": [d.strftime("%Y-%m-%d") for d in df.index],
                "trin": to_list(df["trin"]),
                "ma": to_list(df["ma"]),
                "hi": to_list(df["hi"], 3),
                "lo": to_list(df["lo"], 3),
                "price": to_list(price, 2),
                "signal": [int(s) if e else 0 for s, e in zip(df["state"], df["entry"])],
            },
        })

    payload = {
        "asof": asof.strftime("%a %d %b %Y"),
        "updated": datetime.now(SG).strftime("%Y-%m-%d %H:%M SGT"),
        "params": {"ma": MA_WINDOW, "lookback": LOOKBACK, "bottom_pct": BOTTOM_PCT, "top_pct": TOP_PCT},
        "warnings": warnings,
        "panels": panels,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload), encoding="utf-8")
    print(f"Wrote {OUTPUT} | data through {asof.date()} (last completed US session)")
    for w in warnings:
        print("WARNING:", w)


if __name__ == "__main__":
    build()
