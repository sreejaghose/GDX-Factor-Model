"""Tables and charts."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
