"""Tables, charts and the HTML report."""
from __future__ import annotations

import base64
import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.forecast import trade_direction  # noqa: E402

DEV_COLOR, VAL_COLOR = "#dbe9f6", "#fde8d0"


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def _shade(ax, periods: dict[str, tuple]):
    for name, (a, b) in periods.items():
        ax.axvspan(a, b, color=DEV_COLOR if name == "development" else VAL_COLOR, alpha=0.6, lw=0,
                   label=name, zorder=0)


# ---------------------------------------------------------------------------
# 1. Event study / decay curve
# ---------------------------------------------------------------------------

def event_study(fc: pd.DataFrame, exret: pd.Series, ks, direction_mode: str, close_mask: np.ndarray,
                horizon: int = 10) -> pd.DataFrame:
    """Mean cumulative GDX excess return on days +1..+horizon after each signal close.

    A signal is a close t in ``close_mask`` with |z_t| >= k and a valid direction.
    Rows: k, side ('long'/'short'), h, n, mean, se (se = std/sqrt(n); events can
    overlap, so the bands understate uncertainty). Also 'edge': the return in the
    trade's direction, pooled over both sides.
    """
    r = exret.reindex(fc.index).to_numpy(float)
    N = len(r)
    fwd = np.full((N, horizon), np.nan)
    for h in range(1, horizon + 1):
        fwd[: N - h, h - 1] = r[h:]
    cum = np.cumsum(fwd, axis=1)
    d = trade_direction(fc, direction_mode).to_numpy()
    z = fc["z"].abs().to_numpy()
    ok = close_mask & ~np.isnan(cum[:, -1]) & (d != 0)
    rows = []
    for k in ks:
        sig = ok & (z >= k)
        for side, sel, sgn in (("long", sig & (d > 0), 1), ("short", sig & (d < 0), 1),
                               ("edge", sig, None)):
            x = cum[sel] * (d[sel][:, None] if sgn is None else 1)
            n = len(x)
            for h in range(horizon):
                col = x[:, h]
                rows.append(dict(k=k, side=side, h=h + 1, n=n, mean=col.mean() if n else np.nan,
                                 se=col.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan))
    return pd.DataFrame(rows)


def plot_event_study(es: dict[str, pd.DataFrame], path, title: str) -> Path:
    """es: {'development': df, 'validation': df}; columns long / short / edge."""
    sides = ["long", "short", "edge"]
    fig, axes = plt.subplots(len(es), 3, figsize=(15, 3.6 * len(es)), squeeze=False, sharex=True)
    cmap = plt.get_cmap("viridis")
    for i, (period, df) in enumerate(es.items()):
        ks = sorted(df["k"].unique())
        for j, side in enumerate(sides):
            ax = axes[i, j]
            for c, k in enumerate(ks):
                s = df[(df.k == k) & (df.side == side)]
                col = cmap(c / max(len(ks) - 1, 1))
                ax.plot(s.h, s["mean"] * 1e4, marker="o", ms=3, color=col, label=f"k={k:g} (n={s.n.iloc[0]})")
                ax.fill_between(s.h, (s["mean"] - s.se) * 1e4, (s["mean"] + s.se) * 1e4, color=col, alpha=0.12)
            ax.axhline(0, color="k", lw=0.7)
            ax.set_title(f"{period}: {'return in trade direction' if side == 'edge' else side + ' signals'}",
                         fontsize=10)
            ax.set_ylabel("cum. GDX excess return (bps)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("days after signal close")
    fig.suptitle(title, y=1.0)
    return _save(fig, path)


# ---------------------------------------------------------------------------
# 2. Heatmaps
# ---------------------------------------------------------------------------

def plot_heatmaps(table: pd.DataFrame, x: str, y: str, fixed: dict, path, highlight: dict | None = None,
                  value: str = "sharpe", title: str = "") -> Path:
    """Grid of heatmaps (rows = factor sets, cols = policies) of ``value`` over (y x x),
    other params held at ``fixed``. ``table`` has config params + metrics."""
    t = table.copy()
    t["factors_s"] = t["factors"].map(lambda f: "+".join(f) if isinstance(f, tuple) else f)
    for p, v in fixed.items():
        t = t[t[p] == v]
    fsets = sorted(t["factors_s"].unique(), key=lambda s: (s.count("+"), s))
    pols = sorted(t["policy"].unique())
    fig, axes = plt.subplots(len(fsets), len(pols), figsize=(4.6 * len(pols), 3.6 * len(fsets)), squeeze=False,
                             layout="constrained")
    lim = np.nanmax(np.abs(t[value])) if len(t) else 1
    for i, fs in enumerate(fsets):
        for j, pol in enumerate(pols):
            ax = axes[i, j]
            s = t[(t.factors_s == fs) & (t.policy == pol)].pivot_table(index=y, columns=x, values=value)
            im = ax.imshow(s.to_numpy(), cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto", origin="lower")
            ax.set_xticks(range(s.shape[1]), [f"{v:g}" for v in s.columns])
            ax.set_yticks(range(s.shape[0]), [f"{v:g}" for v in s.index])
            for a in range(s.shape[0]):
                for b in range(s.shape[1]):
                    ax.text(b, a, f"{s.iat[a, b]:.2f}", ha="center", va="center", fontsize=7)
            if highlight and highlight.get("factors") == fs and highlight.get("policy") == pol:
                a, b = list(s.index).index(highlight[y]), list(s.columns).index(highlight[x])
                ax.add_patch(plt.Rectangle((b - 0.5, a - 0.5), 1, 1, fill=False, ec="k", lw=2.2))
            ax.set_title(f"{fs} | {pol}", fontsize=9)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
    fig.colorbar(im, ax=axes, shrink=0.6, label=f"development net {value}")
    fig.suptitle(title or f"Development net Sharpe over ({y} × {x}); fixed: "
                 + ", ".join(f"{k}={v}" for k, v in fixed.items()))
    return _save(fig, path)


# ---------------------------------------------------------------------------
# 3. Stage-2 slope over time
# ---------------------------------------------------------------------------

def plot_gamma(fcs: dict[str, pd.DataFrame], periods: dict, path, title: str = "") -> Path:
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    for label, fc in fcs.items():
        a1.plot(fc.index, fc["gamma"] * 1e4, lw=1, label=label)
        a2.plot(fc.index, fc["gamma_t"], lw=1, label=label)
    for ax in (a1, a2):
        _shade(ax, periods)
        ax.axhline(0, color="k", lw=0.7)
        ax.grid(alpha=0.3)
    for v in (-2, 2):
        a2.axhline(v, color="grey", ls="--", lw=0.8)
    a1.set_ylabel("γ̂ (bps per unit z)")
    a2.set_ylabel("t-stat of γ̂")
    a1.legend(fontsize=8, ncol=4)
    a1.set_title(title or "Stage-2 slope γ̂_t (expanding, no intercept): r_{t+1} = γ z_t")
    return _save(fig, path)


def plot_rolling_betas(betas: dict[str, pd.DataFrame], path: str | Path,
                       full_sample: dict[str, float] | None = None, title: str = "") -> Path:
    """One panel per coefficient; one line per lookback label.

    ``betas`` maps a label (e.g. 'L=250') to a frame from ResidualCache.betas.
    ``full_sample`` optionally draws the full-sample estimate as a dashed line.
    """
    coefs = [c for c in next(iter(betas.values())).columns if c.startswith("beta_")]
    fig, axes = plt.subplots(len(coefs), 1, figsize=(11, 2.6 * len(coefs)), sharex=True, squeeze=False)
    for ax, c in zip(axes[:, 0], coefs):
        for label, df in betas.items():
            ax.plot(df.index, df[c], lw=0.9, label=label)
        if full_sample and c in full_sample:
            ax.axhline(full_sample[c], color="k", ls="--", lw=0.8, label="full sample")
        ax.set_ylabel(c.replace("beta_", "β "))
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="upper left", ncol=len(betas) + 1, fontsize=8)
    axes[0, 0].set_title(title or "Rolling factor betas (window [t−L, t−1])")
    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------------------
# 5-6. Equity, drawdown, monthly entries
# ---------------------------------------------------------------------------

def plot_equity(nets: dict[str, pd.Series], benchmark: pd.Series, periods: dict, path) -> Path:
    """Cumulative additive P&L on 1x notional (excess of cash) and drawdowns."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    series = {**nets, "buy & hold GDX (excess)": benchmark}
    for label, s in series.items():
        cum = s.cumsum()
        dd = cum - cum.cummax().clip(lower=0)
        kw = dict(color="grey", lw=1, ls="--") if label.startswith("buy") else dict(lw=1.2)
        a1.plot(cum.index, cum, label=label, **kw)
        a2.plot(dd.index, dd, label=label, **kw)
    for ax in (a1, a2):
        _shade(ax, periods)
        ax.grid(alpha=0.3)
        ax.axhline(0, color="k", lw=0.6)
    a1.set_ylabel("cumulative net return (sum)")
    a2.set_ylabel("drawdown")
    h, lab = a1.get_legend_handles_labels()
    keep = {l: hh for hh, l in zip(h, lab)}
    a1.legend(keep.values(), keep.keys(), fontsize=8, loc="upper left")
    a1.set_title("Finalists (net of 2 bps) vs buy-and-hold GDX excess return")
    return _save(fig, path)


def plot_monthly_entries(entries: dict[str, pd.Series], periods: dict, path) -> Path:
    """entries: label -> Series of new entries per calendar month."""
    fig, axes = plt.subplots(len(entries), 1, figsize=(12, 2.4 * len(entries)), sharex=True, squeeze=False)
    for ax, (label, s) in zip(axes[:, 0], entries.items()):
        x = s.index.to_timestamp()
        zero = s == 0
        ax.bar(x[~zero], s[~zero], width=25, color="#4c72b0", zorder=3)
        if zero.any():
            ax.bar(x[zero], np.full(zero.sum(), 0.6), width=25, color="red", label="month with no entry", zorder=3)
            ax.legend(fontsize=8)
        ax.axhline(1, color="k", lw=0.8, ls="--")
        _shade(ax, periods)
        ax.set_ylabel("entries")
        ax.set_title(f"{label}: min {int(s.min())}/month, {100 * (s >= 1).mean():.0f}% of months ≥ 1",
                     fontsize=9)
    return _save(fig, path)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

CSS = """
:root { --bg:#ffffff; --fg:#1d1d1f; --muted:#5f6368; --border:#dadce0; --warn-bg:#fdecea; --warn-fg:#8a1c1c;
        --ok-bg:#e6f4ea; --th:#f1f3f4; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#1e1f22; --fg:#e8eaed;
        --muted:#9aa0a6; --border:#3c4043; --warn-bg:#3b1f1f; --warn-fg:#f5b5b5; --ok-bg:#1f3b28; --th:#2a2b2e; } }
body { background:var(--bg); color:var(--fg); font:14px/1.5 -apple-system, Segoe UI, Roboto, sans-serif;
       max-width:1200px; margin:0 auto; padding:16px; }
h1 { font-size:24px; } h2 { font-size:19px; margin-top:32px; border-bottom:1px solid var(--border); }
.muted { color:var(--muted); } img { max-width:100%; height:auto; background:#fff; }
.scroll { overflow-x:auto; } table { border-collapse:collapse; font-size:12px; }
th, td { border:1px solid var(--border); padding:3px 6px; text-align:right; white-space:nowrap; }
th { background:var(--th); } td:first-child, th:first-child { text-align:left; }
.warn { background:var(--warn-bg); color:var(--warn-fg); padding:10px 14px; border-radius:6px; }
.ok { background:var(--ok-bg); padding:10px 14px; border-radius:6px; }
"""


def img_tag(path: Path) -> str:
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{html.escape(Path(path).stem)}">'


def table_html(df: pd.DataFrame, digits: int = 3) -> str:
    return '<div class="scroll">' + df.to_html(float_format=lambda v: f"{v:.{digits}f}", na_rep="–",
                                                border=0) + "</div>"


def build_html(title: str, sections: list[tuple[str, str]], path) -> Path:
    body = "".join(f"<h2>{html.escape(h)}</h2>\n{content}\n" for h, content in sections)
    doc = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
           f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
           f"<body><h1>{html.escape(title)}</h1>{body}</body></html>")
    path = Path(path)
    path.write_text(doc)
    return path
