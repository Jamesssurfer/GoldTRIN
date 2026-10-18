#!/usr/bin/env python3
"""
GoldTRIN - Arms Index (TRIN) breadth dashboard for gold miners.

TRIN = (Advancers / Decliners) / (Advancing Volume / Declining Volume)
Signal rule (on the 5-day MA of TRIN):
    MA >= 1.2  -> potential bottom
    MA <= 0.8  -> potential top

Output: docs/index.html (static, GitHub Pages ready)
"""
from __future__ import annotations

from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from plotly.subplots import make_subplots

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
OUTPUT = Path("docs/index.html")

# US-listed tickers only (avoids cross-listing volume distortion). EDIT THESE.
CORE = ["NEM", "AEM", "B", "WPM", "FNV", "GFI", "AU", "KGC", "AGI", "HMY",
        "EGO", "BTG", "IAG", "OR", "PAAS", "HL", "CDE", "EQX", "NGD", "SSRM"]
JUNIORS = ["DRD", "ORLA", "CGAU", "SA", "AUGO", "GORO", "TFPM"]

PANELS = [
    # name,                                   basket,          price_ticker (None = equal-weight basket index)
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
    out["ma"] = out["trin"].rolling(MA_WINDOW, min_periods=MA_WINDOW).mean()
    out["composite"] = (1 + c.pct_change(fill_method=None).mean(axis=1)).cumprod()
    out.attrs["members"] = members
    out.attrs["missing"] = sorted(set(basket) - set(members))

    state = np.where(out["ma"] >= BOTTOM_LEVEL, 1, np.where(out["ma"] <= TOP_LEVEL, -1, 0))
    state = pd.Series(state, index=out.index).where(out["ma"].notna(), 0)
    out["state"] = state
    out["entry"] = (state != state.shift()) & (state != 0)
    return out


# ----------------------------------------------------------------- CHARTS
def make_chart(name: str, df: pd.DataFrame, price: pd.Series) -> str:
    df = df.tail(DISPLAY_DAYS)
    price = (price.reindex(df.index).ffill())
    price = price / price.dropna().iloc[0] * 100 if price.notna().any() else price

    top = max(2.0, float(df["ma"].max(skipna=True) or 0) * 1.2)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_hrect(y0=BOTTOM_LEVEL, y1=top, fillcolor="#16a34a", opacity=0.07, line_width=0)
    fig.add_hrect(y0=0, y1=TOP_LEVEL, fillcolor="#dc2626", opacity=0.07, line_width=0)
    fig.add_hline(y=BOTTOM_LEVEL, line=dict(color="#16a34a", width=1, dash="dash"),
                  annotation_text="1.2 potential bottom", annotation_position="top left",
                  annotation_font=dict(size=10, color="#16a34a"))
    fig.add_hline(y=TOP_LEVEL, line=dict(color="#dc2626", width=1, dash="dash"),
                  annotation_text="0.8 potential top", annotation_position="bottom left",
                  annotation_font=dict(size=10, color="#dc2626"))
    fig.add_hline(y=1.0, line=dict(color="#9ca3af", width=0.7, dash="dot"))

    fig.add_trace(go.Scatter(x=price.index, y=price, name="Price (idx 100)",
                             line=dict(color="#cbd5e1", width=1.5),
                             hovertemplate="%{y:.1f}"), secondary_y=True)
    fig.add_trace(go.Scatter(x=df.index, y=df["trin"], name="TRIN (daily)",
                             line=dict(color="#94a3b8", width=1),
                             hovertemplate="%{y:.2f}"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=df["ma"], name=f"TRIN {MA_WINDOW}d MA",
                             line=dict(color="#b45309", width=2.5),
                             hovertemplate="%{y:.2f}"), secondary_y=False)

    b = df[df["entry"] & (df["state"] == 1)]
    t = df[df["entry"] & (df["state"] == -1)]
    fig.add_trace(go.Scatter(x=b.index, y=b["ma"], mode="markers", name="Potential bottom",
                             marker=dict(symbol="triangle-up", size=13, color="#16a34a",
                                         line=dict(color="white", width=1))), secondary_y=False)
    fig.add_trace(go.Scatter(x=t.index, y=t["ma"], mode="markers", name="Potential top",
                             marker=dict(symbol="triangle-down", size=13, color="#dc2626",
                                         line=dict(color="white", width=1))), secondary_y=False)

    fig.update_layout(
        template="plotly_white", height=400, margin=dict(l=50, r=50, t=50, b=30),
        title=dict(text=f"<b>{name}</b>", x=0.01, font=dict(size=16)),
        legend=dict(orientation="h", y=1.12, x=1, xanchor="right", font=dict(size=11)),
        hovermode="x unified", font=dict(family="Inter, Segoe UI, sans-serif"),
    )
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], showgrid=False)
    fig.update_yaxes(range=[0, top], title_text="TRIN", gridcolor="#f1f5f9", secondary_y=False)
    fig.update_yaxes(showgrid=False, title_text="Price", secondary_y=True)
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"displayModeBar": False, "responsive": True})


# -------------------------------------------------------------- DASHBOARD
def status(row: pd.Series) -> tuple[str, str]:
    if row["state"] == 1:
        return "POTENTIAL BOTTOM", "#16a34a"
    if row["state"] == -1:
        return "POTENTIAL TOP", "#dc2626"
    return "NEUTRAL", "#64748b"


def build() -> None:
    all_tickers = {t for p in PANELS for t in p["basket"]} | {p["price"] for p in PANELS if p["price"]}
    close, vol = download(sorted(all_tickers))
    asof = close.index.max()

    cards, charts, warnings = [], [], []
    for p in PANELS:
        df = compute(close, vol, p["basket"])
        if df["ma"].dropna().empty:
            warnings.append(f"{p['name']}: not enough data")
            continue
        if df.attrs["missing"]:
            warnings.append(f"{p['name']}: no data for {', '.join(df.attrs['missing'])}")
        price = close[p["price"]] if p["price"] in close.columns else df["composite"]
        last = df.dropna(subset=["ma"]).iloc[-1]
        label, color = status(last)
        cards.append(f"""
        <div class="card"><div class="cn">{p['name']}</div>
        <div class="row"><span>TRIN</span><b>{last['trin']:.2f}</b></div>
        <div class="row"><span>{MA_WINDOW}d MA</span><b>{last['ma']:.2f}</b></div>
        <div class="row"><span>Adv / Dec</span><b>{int(last['adv'])} / {int(last['dec'])}</b></div>
        <div class="badge" style="background:{color}">{label}</div></div>""")
        charts.append(f'<div class="chart">{make_chart(p["name"], df, price)}</div>')

    warn_html = "".join(f"<li>{w}</li>" for w in warnings)
    updated = datetime.now(SG).strftime("%Y-%m-%d %H:%M SGT")
    html = TEMPLATE.replace("{{CARDS}}", "".join(cards)).replace("{{CHARTS}}", "".join(charts)) \
        .replace("{{ASOF}}", asof.strftime("%a %d %b %Y")).replace("{{UPDATED}}", updated) \
        .replace("{{WARN}}", f"<ul class='warn'>{warn_html}</ul>" if warnings else "")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT} | data through {asof.date()} (last completed US session)")
    for w in warnings:
        print("WARNING:", w)


TEMPLATE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GoldTRIN</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body{margin:0;background:#f8fafc;color:#0f172a;font-family:Inter,'Segoe UI',sans-serif}
.wrap{max-width:1200px;margin:0 auto;padding:28px 20px}
h1{margin:0;font-size:26px} .sub{color:#64748b;font-size:13px;margin:4px 0 22px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px;margin-bottom:22px}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:16px}
.cn{font-weight:600;font-size:13px;margin-bottom:10px}
.row{display:flex;justify-content:space-between;font-size:13px;color:#475569;padding:2px 0}
.row b{color:#0f172a}
.badge{margin-top:12px;color:#fff;font-size:11px;font-weight:700;letter-spacing:.06em;
text-align:center;border-radius:6px;padding:6px}
.chart{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:8px;margin-bottom:18px}
.note{color:#64748b;font-size:12px;line-height:1.6;margin-top:10px}
.warn{color:#b45309;font-size:12px}
</style></head><body><div class="wrap">
<h1>GoldTRIN</h1>
<div class="sub">Arms Index (TRIN) breadth for gold miners &middot; data through close of {{ASOF}} &middot; generated {{UPDATED}}</div>
<div class="cards">{{CARDS}}</div>
{{CHARTS}}
<div class="note">TRIN = (Advancers / Decliners) &divide; (Advancing Volume / Declining Volume), computed from constituent baskets.
Signals use the 5-day MA of TRIN: &ge;1.2 potential bottom (green), &le;0.8 potential top (red). Triangles mark the first day a zone is entered.
Only completed US sessions are used.</div>
{{WARN}}
</div></body></html>"""

if __name__ == "__main__":
    build()
