"""
validation.py
=============
Discrimination, rank-ordering and calibration diagnostics for a PD model.

Metrics implemented
--------------------
* KS statistic        : max gap between cumulative good and bad distributions.
                        The headline separation metric in credit risk.
* AUC-ROC / Gini      : Gini = 2*AUC - 1. Overall ranking power.
* Score-band table    : the classic decile/band table showing bad-rate
                        rank-ordering, cumulative capture and KS by band.
* Calibration         : predicted PD vs observed default rate (reliability).
* Confusion / P-R     : operating-point classification quality.
* Brier / log-loss    : probabilistic accuracy.

A model is only deployable if it BOTH discriminates (high KS/Gini) AND
rank-orders monotonically AND is well calibrated. We check all three.

Author: Tanmay Shrivastava
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, log_loss


def ks_statistic(y_true, score, higher_is_safer=True):
    """KS = max |cum%good - cum%bad| as score moves from risky -> safe."""
    df = pd.DataFrame({"y": np.asarray(y_true), "s": np.asarray(score)})
    # sort from riskiest to safest
    df = df.sort_values("s", ascending=higher_is_safer)
    total_bad = (df.y == 1).sum()
    total_good = (df.y == 0).sum()
    df["cum_bad"] = (df.y == 1).cumsum() / total_bad
    df["cum_good"] = (df.y == 0).cumsum() / total_good
    df["gap"] = (df.cum_bad - df.cum_good).abs()
    return float(df["gap"].max())


def gini_auc(y_true, p_bad):
    auc = roc_auc_score(y_true, p_bad)
    return 2 * auc - 1, auc


def score_band_table(y_true, score, n_bands=10, higher_is_safer=True):
    """Decile-style table that risk teams use to read a scorecard.

    Bands are ordered RISKIEST -> SAFEST so bad-rate should fall down the table.
    """
    df = pd.DataFrame({"y": np.asarray(y_true).astype(int),
                       "score": np.asarray(score)})
    # rank into bands; band 1 = riskiest (lowest score)
    df["band"] = pd.qcut(df["score"].rank(method="first"),
                         q=n_bands, labels=False) + 1
    if not higher_is_safer:
        df["band"] = n_bands + 1 - df["band"]

    total_bad = (df.y == 1).sum()
    total_good = (df.y == 0).sum()

    g = df.groupby("band")
    tbl = pd.DataFrame({
        "n": g.size(),
        "min_score": g["score"].min().round(1),
        "max_score": g["score"].max().round(1),
        "bads": g["y"].sum(),
    })
    tbl["goods"] = tbl["n"] - tbl["bads"]
    tbl["bad_rate"] = (tbl["bads"] / tbl["n"]).round(4)
    # cumulative from riskiest band downward
    tbl = tbl.sort_index()
    tbl["cum_bad_pct"] = (tbl["bads"].cumsum() / total_bad).round(4)
    tbl["cum_good_pct"] = (tbl["goods"].cumsum() / total_good).round(4)
    tbl["ks"] = ((tbl["cum_bad_pct"] - tbl["cum_good_pct"]).abs()).round(4)
    # capture / lift
    overall_bad_rate = total_bad / len(df)
    tbl["lift"] = (tbl["bad_rate"] / overall_bad_rate).round(2)
    return tbl.reset_index()


def calibration_table(y_true, p_bad, n_bins=10):
    """Predicted-PD bucket vs observed default rate (reliability diagram data)."""
    df = pd.DataFrame({"y": np.asarray(y_true).astype(int),
                       "p": np.asarray(p_bad)})
    df["bucket"] = pd.qcut(df["p"].rank(method="first"), q=n_bins, labels=False)
    g = df.groupby("bucket")
    out = pd.DataFrame({
        "n": g.size(),
        "pred_pd": g["p"].mean().round(4),
        "actual_pd": g["y"].mean().round(4),
    }).reset_index(drop=True)
    return out


def confusion_at_cutoff(y_true, p_bad, threshold):
    y = np.asarray(y_true).astype(int)
    pred = (np.asarray(p_bad) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {
        "threshold": threshold, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def full_report(y_true, p_bad, score, label=""):
    """Bundle the headline metrics into a dict."""
    gini, auc = gini_auc(y_true, p_bad)
    ks = ks_statistic(y_true, score)
    return {
        "segment": label,
        "n": int(len(y_true)),
        "bad_rate": round(float(np.mean(y_true)), 4),
        "ks": round(ks, 4),
        "gini": round(gini, 4),
        "auc": round(auc, 4),
        "brier": round(brier_score_loss(y_true, p_bad), 5),
        "log_loss": round(log_loss(y_true, p_bad), 5),
    }


def roc_points(y_true, p_bad):
    fpr, tpr, _ = roc_curve(y_true, p_bad)
    return fpr, tpr


if __name__ == "__main__":
    from data_generation import generate_credit_data
    from woe_iv import WOEEncoder
    from scorecard import Scorecard, select_features, prune_multicollinearity
    from sklearn.model_selection import train_test_split

    dev, _ = generate_credit_data()
    dev = dev.reset_index(drop=True)
    y = dev["default_flag"]
    X = dev.drop(columns=["default_flag", "vintage"])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          stratify=y, random_state=42)

    enc = WOEEncoder().fit(Xtr, ytr)
    iv = enc.iv_summary()["IV"]
    feats = select_features(iv)
    woe_tr, woe_te = enc.transform(Xtr), enc.transform(Xte)
    kept, _ = prune_multicollinearity(woe_tr[[f + "_woe" for f in feats]])
    sc = Scorecard(pdo=40, base_score=600, base_odds=20).fit(woe_tr, ytr, kept)

    p_te = sc.predict_proba(woe_te)
    s_te = sc.score(woe_te)
    print(full_report(yte, p_te, s_te, "TEST"))
    print("\nScore-band table:")
    print(score_band_table(yte, s_te).to_string(index=False))
