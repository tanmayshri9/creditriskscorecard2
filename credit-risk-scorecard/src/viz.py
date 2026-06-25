"""
viz.py
======
Charting layer for the scorecard project. One consistent visual identity across
every figure: deep-navy ink, a teal primary accent, and a green/red good-bad
semantic pair (the universal language of credit risk).

All functions save a high-resolution PNG to outputs/figures/ and return the path.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# ---- palette --------------------------------------------------------------- #
INK       = "#0B1F33"
INK_SOFT  = "#52606D"
TEAL      = "#127475"
TEAL_SOFT = "#7FB2B2"
GOOD      = "#1B7A43"
BAD       = "#C23B22"
AMBER     = "#C68A12"
GRID      = "#E6E9ED"
PAPER     = "#FFFFFF"

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---- global style ---------------------------------------------------------- #
for _f in ["Liberation Sans"]:
    try:
        fm.findfont(_f, fallback_to_default=False)
        _SANS = _f
        break
    except Exception:
        _SANS = "DejaVu Sans"
else:
    _SANS = "DejaVu Sans"

plt.rcParams.update({
    "font.family": _SANS,
    "font.size": 11,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.8,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.labelcolor": INK_SOFT,
    "axes.labelsize": 10.5,
    "text.color": INK,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "savefig.dpi": 150,
    "figure.dpi": 110,
})


def _style(ax, grid="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    if grid in ("y", "both"):
        ax.yaxis.grid(True, color=GRID, linewidth=0.9, zorder=0)
    if grid in ("x", "both"):
        ax.xaxis.grid(True, color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    return ax


def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 1. Portfolio / target overview
# --------------------------------------------------------------------------- #
def plot_target_overview(dev):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))

    bad_rate = dev["default_flag"].mean()
    axes[0].bar(["Good\n(performing)", "Bad\n(90+ DPD)"],
                [1 - bad_rate, bad_rate],
                color=[GOOD, BAD], width=0.62, zorder=3)
    for i, v in enumerate([1 - bad_rate, bad_rate]):
        axes[0].text(i, v + 0.02, f"{v*100:.1f}%", ha="center",
                     fontweight="bold", color=INK)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Target distribution")
    axes[0].set_ylabel("Share of book")
    _style(axes[0])

    # bad rate by bureau score band
    dev2 = dev.copy()
    dev2["band"] = pd.qcut(dev2["bureau_score"], 6, duplicates="drop")
    br = dev2.groupby("band", observed=True)["default_flag"].mean()
    axes[1].plot(range(len(br)), br.values, "-o", color=TEAL, lw=2.2,
                 markersize=6, zorder=3)
    axes[1].set_xticks(range(len(br)))
    axes[1].set_xticklabels([f"{int(i.left)}-{int(i.right)}" for i in br.index],
                            rotation=35, ha="right", fontsize=8)
    axes[1].set_title("Bad rate by bureau-score band")
    axes[1].set_ylabel("Bad rate")
    _style(axes[1])

    # income distribution good vs bad
    g = dev.loc[dev.default_flag == 0, "annual_income"].clip(upper=200000)
    b = dev.loc[dev.default_flag == 1, "annual_income"].clip(upper=200000)
    axes[2].hist(g, bins=40, color=GOOD, alpha=0.55, density=True, label="Good")
    axes[2].hist(b, bins=40, color=BAD, alpha=0.55, density=True, label="Bad")
    axes[2].set_title("Income density by outcome")
    axes[2].set_xlabel("Annual income ($)")
    axes[2].legend(frameon=False, fontsize=9)
    _style(axes[2])

    fig.tight_layout()
    return _save(fig, "01_target_overview.png")


# --------------------------------------------------------------------------- #
# 2. Information Value ranking
# --------------------------------------------------------------------------- #
def plot_iv_ranking(iv_df):
    iv_df = iv_df.sort_values("IV")
    colors = []
    for v in iv_df["IV"]:
        if v < 0.02:    colors.append("#C7CED6")
        elif v < 0.1:   colors.append(TEAL_SOFT)
        elif v < 0.3:   colors.append(TEAL)
        else:           colors.append(INK)
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.barh(iv_df.index, iv_df["IV"], color=colors, zorder=3, height=0.72)
    for y, v in enumerate(iv_df["IV"]):
        ax.text(v + 0.006, y, f"{v:.3f}", va="center", fontsize=8.5, color=INK)
    for x in (0.02, 0.1, 0.3):
        ax.axvline(x, color=AMBER, ls="--", lw=1, alpha=0.7)
    ax.set_title("Information Value by predictor")
    ax.set_xlabel("IV   (dashed: weak 0.02 · medium 0.1 · strong 0.3)")
    _style(ax, grid="x")
    fig.tight_layout()
    return _save(fig, "02_iv_ranking.png")


# --------------------------------------------------------------------------- #
# 3. WOE patterns (small multiples)
# --------------------------------------------------------------------------- #
def plot_woe_grid(encoder, features):
    n = len(features)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12.5, 3.1 * nrow))
    axes = np.array(axes).reshape(-1)
    for ax, feat in zip(axes, features):
        tbl = encoder.bins_[feat]
        woe = tbl["woe"].values
        x = range(len(woe))
        colors = [GOOD if w >= 0 else BAD for w in woe]
        ax.bar(x, woe, color=colors, zorder=3, width=0.7)
        ax.axhline(0, color=INK, lw=0.9)
        ax.set_title(feat, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i + 1) for i in x], fontsize=8)
        ax.set_ylabel("WOE", fontsize=9)
        _style(ax)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Weight-of-Evidence by bin   (green = safer, red = riskier)",
                 fontsize=12.5, fontweight="bold", color=INK, y=1.005, x=0.01,
                 ha="left")
    fig.tight_layout()
    return _save(fig, "03_woe_grid.png")


# --------------------------------------------------------------------------- #
# 4. Correlation heatmap of WOE features
# --------------------------------------------------------------------------- #
def plot_corr_heatmap(woe_df, features):
    sub = woe_df[features].copy()
    sub.columns = [c[:-4] for c in sub.columns]
    corr = sub.corr()
    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticklabels(corr.columns, fontsize=8.5)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7.5, color="white" if abs(v) > 0.55 else INK)
    ax.set_title("WOE feature correlation  (multicollinearity check)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    return _save(fig, "04_corr_heatmap.png")


# --------------------------------------------------------------------------- #
# 5. KS curve
# --------------------------------------------------------------------------- #
def plot_ks_curve(y_true, score, ks_value):
    df = pd.DataFrame({"y": np.asarray(y_true), "s": np.asarray(score)})
    df = df.sort_values("s", ascending=True)
    tb, tg = (df.y == 1).sum(), (df.y == 0).sum()
    df["cum_bad"] = (df.y == 1).cumsum() / tb
    df["cum_good"] = (df.y == 0).cumsum() / tg
    pct = np.linspace(0, 1, len(df))
    gap = (df.cum_bad.values - df.cum_good.values)
    kmax = np.argmax(np.abs(gap))

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.plot(pct, df.cum_bad.values, color=BAD, lw=2.4, label="Cumulative bads")
    ax.plot(pct, df.cum_good.values, color=GOOD, lw=2.4, label="Cumulative goods")
    ax.vlines(pct[kmax], df.cum_good.values[kmax], df.cum_bad.values[kmax],
              color=INK, lw=2, ls="--")
    ax.annotate(f"KS = {ks_value*100:.1f}",
                xy=(pct[kmax], (df.cum_good.values[kmax] + df.cum_bad.values[kmax]) / 2),
                xytext=(pct[kmax] + 0.08, 0.45), fontsize=12, fontweight="bold",
                color=INK, arrowprops=dict(arrowstyle="->", color=INK))
    ax.set_xlabel("Population ordered riskiest → safest")
    ax.set_ylabel("Cumulative proportion")
    ax.set_title("Kolmogorov–Smirnov separation")
    ax.legend(frameon=False, loc="lower right", fontsize=9.5)
    _style(ax, grid="both")
    fig.tight_layout()
    return _save(fig, "05_ks_curve.png")


# --------------------------------------------------------------------------- #
# 6. ROC champion vs challengers
# --------------------------------------------------------------------------- #
def plot_roc(curves):
    """curves: list of (label, fpr, tpr, auc, color)"""
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.plot([0, 1], [0, 1], color="#B6BFC9", ls="--", lw=1.2)
    for label, fpr, tpr, auc, color in curves:
        ax.plot(fpr, tpr, lw=2.4, color=color,
                label=f"{label}  (AUC {auc:.3f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC: champion vs challengers")
    ax.legend(frameon=False, loc="lower right", fontsize=9.5)
    _style(ax, grid="both")
    fig.tight_layout()
    return _save(fig, "06_roc.png")


# --------------------------------------------------------------------------- #
# 7. Score distribution by outcome
# --------------------------------------------------------------------------- #
def plot_score_distribution(y_true, score):
    y = np.asarray(y_true)
    s = np.asarray(score)
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    bins = np.linspace(s.min(), s.max(), 45)
    ax.hist(s[y == 0], bins=bins, color=GOOD, alpha=0.55, density=True, label="Good")
    ax.hist(s[y == 1], bins=bins, color=BAD, alpha=0.55, density=True, label="Bad")
    ax.axvline(np.median(s[y == 0]), color=GOOD, ls="--", lw=1.5)
    ax.axvline(np.median(s[y == 1]), color=BAD, ls="--", lw=1.5)
    ax.set_xlabel("Credit score")
    ax.set_ylabel("Density")
    ax.set_title("Score separation of goods and bads")
    ax.legend(frameon=False, fontsize=9.5)
    _style(ax)
    fig.tight_layout()
    return _save(fig, "07_score_distribution.png")


# --------------------------------------------------------------------------- #
# 8. Rank ordering
# --------------------------------------------------------------------------- #
def plot_rank_ordering(band_table):
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    x = band_table["band"].astype(int)
    bars = ax.bar(x, band_table["bad_rate"] * 100, color=TEAL, zorder=3, width=0.62)
    # gradient: riskier bands darker red-ish
    for bar, br in zip(bars, band_table["bad_rate"]):
        bar.set_color(plt.cm.RdYlGn_r(min(br / band_table["bad_rate"].max(), 1) * 0.8 + 0.1))
    ax.set_xticks(x)
    ax.set_xlabel("Score band   (1 = riskiest · 10 = safest)")
    ax.set_ylabel("Bad rate (%)")
    ax.set_title("Rank-ordering: bad rate by score band")
    for xi, br in zip(x, band_table["bad_rate"]):
        ax.text(xi, br * 100 + 0.5, f"{br*100:.1f}", ha="center", fontsize=8, color=INK)
    _style(ax)
    fig.tight_layout()
    return _save(fig, "08_rank_ordering.png")


# --------------------------------------------------------------------------- #
# 9. Calibration
# --------------------------------------------------------------------------- #
def plot_calibration(cal_table):
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    mx = max(cal_table["pred_pd"].max(), cal_table["actual_pd"].max()) * 1.1
    ax.plot([0, mx], [0, mx], color="#B6BFC9", ls="--", lw=1.3, label="Perfect calibration")
    ax.plot(cal_table["pred_pd"], cal_table["actual_pd"], "-o", color=TEAL,
            lw=2.2, markersize=7, label="Scorecard")
    ax.set_xlabel("Predicted PD")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration (reliability)")
    ax.legend(frameon=False, fontsize=9.5)
    _style(ax, grid="both")
    fig.tight_layout()
    return _save(fig, "09_calibration.png")


# --------------------------------------------------------------------------- #
# 10. Model comparison
# --------------------------------------------------------------------------- #
def plot_model_comparison(reports):
    df = pd.DataFrame(reports).set_index("segment")
    metrics = ["ks", "gini", "auc"]
    x = np.arange(len(metrics))
    w = 0.25
    colors = [INK, TEAL, AMBER]
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for i, (seg, row) in enumerate(df.iterrows()):
        ax.bar(x + (i - 1) * w, [row[m] for m in metrics], width=w,
               color=colors[i % 3], label=seg, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(["KS", "Gini", "AUC"])
    ax.set_title("Champion vs challengers — discrimination")
    ax.legend(frameon=False, fontsize=9.5)
    for i, (seg, row) in enumerate(df.iterrows()):
        for j, m in enumerate(metrics):
            ax.text(x[j] + (i - 1) * w, row[m] + 0.008, f"{row[m]:.3f}",
                    ha="center", fontsize=7.3, color=INK)
    _style(ax)
    fig.tight_layout()
    return _save(fig, "10_model_comparison.png")


# --------------------------------------------------------------------------- #
# 11. Feature importance
# --------------------------------------------------------------------------- #
def plot_feature_importance(imp):
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.barh(imp.index, imp.values, color=TEAL, zorder=3, height=0.72)
    ax.set_title("XGBoost feature importance (gain)")
    ax.set_xlabel("Relative importance")
    _style(ax, grid="x")
    fig.tight_layout()
    return _save(fig, "11_feature_importance.png")


# --------------------------------------------------------------------------- #
# 12. PSI
# --------------------------------------------------------------------------- #
def plot_psi(psi_table, psi_value):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(psi_table))
    w = 0.4
    ax.bar(x - w / 2, psi_table["expected_pct"] * 100, width=w,
           color=INK, label="Development (expected)", zorder=3)
    ax.bar(x + w / 2, psi_table["actual_pct"] * 100, width=w,
           color=AMBER, label="Recent (actual)", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in x])
    ax.set_xlabel("Score band (low → high)")
    ax.set_ylabel("% of population")
    ax.set_title(f"Population Stability Index = {psi_value:.3f}  (investigate zone)")
    ax.legend(frameon=False, fontsize=9.5)
    _style(ax)
    fig.tight_layout()
    return _save(fig, "12_psi.png")


# --------------------------------------------------------------------------- #
# 13. CSI
# --------------------------------------------------------------------------- #
def plot_csi(csi_df):
    csi_df = csi_df.sort_values("csi")
    colors = []
    for v in csi_df["csi"]:
        if v < 0.1:    colors.append(GOOD)
        elif v < 0.25: colors.append(AMBER)
        else:          colors.append(BAD)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(csi_df["characteristic"], csi_df["csi"], color=colors, zorder=3, height=0.7)
    ax.axvline(0.1, color=AMBER, ls="--", lw=1)
    ax.axvline(0.25, color=BAD, ls="--", lw=1)
    for y, v in enumerate(csi_df["csi"]):
        ax.text(v + 0.004, y, f"{v:.3f}", va="center", fontsize=8.5, color=INK)
    ax.set_title("Characteristic Stability Index  (green stable · amber 0.1 · red 0.25)")
    ax.set_xlabel("CSI")
    _style(ax, grid="x")
    fig.tight_layout()
    return _save(fig, "13_csi.png")


# --------------------------------------------------------------------------- #
# 14. Strategy curve
# --------------------------------------------------------------------------- #
def plot_strategy_curve(strategy_df):
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.plot(strategy_df["approval_rate"] * 100,
            strategy_df["approved_bad_rate"] * 100,
            "-o", color=TEAL, lw=2.3, markersize=5, zorder=3)
    ax.set_xlabel("Approval rate (%)")
    ax.set_ylabel("Bad rate of approved book (%)")
    ax.set_title("Growth–risk frontier (strategy curve)")
    _style(ax, grid="both")
    fig.tight_layout()
    return _save(fig, "14_strategy_curve.png")


# --------------------------------------------------------------------------- #
# 15. Profit curve
# --------------------------------------------------------------------------- #
def plot_profit_curve(profit_df, best):
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.plot(profit_df["cutoff_score"], profit_df["expected_profit"] / 1e6,
            color=INK, lw=2.4, zorder=3)
    ax.axvline(best["cutoff_score"], color=BAD, ls="--", lw=1.6)
    ax.scatter([best["cutoff_score"]], [best["expected_profit"] / 1e6],
               color=BAD, zorder=5, s=60)
    ax.annotate(f"Optimal cutoff {best['cutoff_score']:.0f}\n"
                f"${best['expected_profit']/1e6:.2f}M  ·  "
                f"approve {best['approval_rate']*100:.0f}%",
                xy=(best["cutoff_score"], best["expected_profit"] / 1e6),
                xytext=(best["cutoff_score"] - 120, best["expected_profit"] / 1e6 * 0.6),
                fontsize=9.5, color=INK, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=BAD))
    ax.set_xlabel("Approval cutoff score")
    ax.set_ylabel("Expected portfolio profit ($M)")
    ax.set_title("Profit-optimised cutoff")
    _style(ax)
    fig.tight_layout()
    return _save(fig, "15_profit_curve.png")


# --------------------------------------------------------------------------- #
# 16. Swap set
# --------------------------------------------------------------------------- #
def plot_swap_set(summary):
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    colors = [INK, GOOD, BAD]
    bars = ax.bar(summary["group"], summary["bad_rate"] * 100,
                  color=colors, zorder=3, width=0.58)
    for bar, n, br in zip(bars, summary["n"], summary["bad_rate"]):
        ax.text(bar.get_x() + bar.get_width() / 2, br * 100 + 0.3,
                f"{br*100:.1f}%\n(n={n:,})", ha="center", fontsize=9, color=INK)
    ax.set_ylabel("Bad rate (%)")
    ax.set_title("Swap-set: scorecard vs flat bureau cutoff")
    ax.set_xticklabels(summary["group"], fontsize=9)
    _style(ax)
    fig.tight_layout()
    return _save(fig, "16_swap_set.png")
