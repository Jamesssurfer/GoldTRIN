#!/usr/bin/env python3
"""
GoldTRIN - Arms Index (TRIN) breadth data builder for gold miners.

TRIN = (Advancers / Decliners) / (Advancing Volume / Declining Volume)
Signal rule (on the 5-day MA of TRIN):
    MA >= 1.2  -> potential bottom
    MA <= 0.8  -> potential top

Output: docs/data.json  (rendered by docs/index.html)
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
BOTTOM_LEVEL = 1.2
TOP_LEVEL = 0.8
MIN_COVERAGE = 0.70              # min share of basket with valid data on a day
DISPLAY_DAYS = 126               # ~6 months shown
DOWNLOAD_PERIOD = "1y"
OUTPUT = Path("docs/data.json")

# US-listed tickers only (avoids cross-listing volume distortion). EDIT THESE.
CORE = ["NEM", "AEM", "B", "WPM", "FNV", "GFI", "AU", "KGC", "AGI", "HMY",
        "EGO", "BTG", "IAG", "OR", "PAAS", "HL", "CDE", "EQX", "NGD", "SSRM"]
JUNIORS = ["DRD", "ORLA", "CGAU", "SA", "AUGO", "GORO", "TFPM"]

PANELS = [
    # name,                                   basket,          price ticker (None = equal-weight basket index)
    {"name": "NYSE Arca Gold Miners (GDM)",   "basket": CORE,            "price": None},
    {"name": "VanEck Gold Miners ETF (GDX)",  "basket": CORE,            "price": "GDX"},
    {"name": "Fidelity Select Gold (FSAGX)",  "basket": CORE[:16],       "price": "FSAGX"},
    {"name": "GICS Gold (15104030) universe", "basket": CORE + JUNIORS,  "price": None},
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
    raw = yf.download(sorted(set(tickers)), period=DOWNLOAD_PERIOD, interval="1d",
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
    valid = c.notna() & c.shift().notna()
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

    state = np.where(out["ma"] >= BOTTOM_LEVEL, 1, np.where(out["ma"] <= TOP_LEVEL, -1, 0))
    state = pd.Series(state, index=out.index).where(out["ma"].notna(), 0)
    out["state"] = state
    out["entry"] = (state != state.shift()) & (state != 0)
    return out


# ----------------------------------------------------------------- OUTPUT
def to_list(s: pd.Series, nd: int = 4) -> list:
    return [None if pd.isna(x) else round(float(x), nd) for x in s]


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
        if df.attrs["missing"]:
            warnings.append(f"{p['name']}: no data for {', '.join(df.attrs['missing'])}")

        price = close[p["price"]] if p["price"] in close.columns else df["composite"]
        df = df.tail(DISPLAY_DAYS)
        price = price.reindex(df.index).ffill()
        if price.notna().any():
            price = price / price.dropna().iloc[0] * 100
        last = df.dropna(subset=["ma"]).iloc[-1]

        panels.append({
            "name": p["name"],
            "members": len(df.attrs["members"]),
            "last": {
                "trin": None if pd.isna(last["trin"]) else round(float(last["trin"]), 2),
                "ma": round(float(last["ma"]), 2),
                "adv": int(last["adv"]),
                "dec": int(last["dec"]),
                "state": int(last["state"]),
            },
            "series": {
                "dates": [d.strftime("%Y-%m-%d") for d in df.index],
                "trin": to_list(df["trin"]),
                "ma": to_list(df["ma"]),
                "price": to_list(price, 2),
                "signal": [int(s) if e else 0 for s, e in zip(df["state"], df["entry"])],
            },
        })

    payload = {
        "asof": asof.strftime("%a %d %b %Y"),
        "updated": datetime.now(SG).strftime("%Y-%m-%d %H:%M SGT"),
        "params": {"ma": MA_WINDOW, "bottom": BOTTOM_LEVEL, "top": TOP_LEVEL},
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
