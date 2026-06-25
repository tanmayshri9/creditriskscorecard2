"""
business_impact.py
==================
Translate model scores into lending decisions and quantify business value.
This is the layer that turns a statistics exercise into a P&L conversation with
Product, Risk and Finance.

Three analyses:

1. STRATEGY (CUTOFF) CURVE
   For each candidate approval cutoff, what is the approval rate and the bad
   rate of the approved book? This is the growth-vs-risk frontier.

2. PROFIT-OPTIMISED CUTOFF
   Using simple unit economics, find the cutoff that maximises expected
   portfolio profit:
       profit = (#approved_goods * margin_per_good)
                - (#approved_bads  * loss_per_bad)
   This makes the "balance growth, profitability and risk" mandate concrete.

3. SWAP-SET ANALYSIS
   Compare the scorecard policy against an incumbent rule-of-thumb policy
   (a flat bureau-score cutoff). Swap-ins  = approved by new, declined by old.
   Swap-outs = declined by new, approved by old. If swap-ins are better quality
   than swap-outs, the scorecard adds value at the same approval rate.

Author: Tanmay Shrivastava
"""

import numpy as np
import pandas as pd


def strategy_curve(y_true, score, n_steps=25):
    """Approval rate & approved-book bad rate across score cutoffs."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score)
    cutoffs = np.quantile(s, np.linspace(0.02, 0.98, n_steps))
    rows = []
    total = len(y)
    total_bad = y.sum()
    for c in cutoffs:
        approved = s >= c
        n_app = approved.sum()
        if n_app == 0:
            continue
        app_bad = y[approved].sum()
        rows.append({
            "cutoff_score": round(float(c), 1),
            "approval_rate": round(n_app / total, 4),
            "approved_bad_rate": round(app_bad / n_app, 4),
            "bads_avoided": int(total_bad - app_bad),
            "bad_capture_rate": round((total_bad - app_bad) / total_bad, 4),
        })
    return pd.DataFrame(rows)


def profit_optimised_cutoff(
    y_true, score,
    margin_per_good=1_200,   # lifetime margin earned on a performing loan
    loss_per_bad=9_000,      # EAD * LGD on a defaulted loan
    n_steps=60,
):
    """Find the cutoff that maximises expected portfolio profit."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score)
    cutoffs = np.quantile(s, np.linspace(0.01, 0.99, n_steps))
    rows = []
    total = len(y)
    for c in cutoffs:
        approved = s >= c
        n_app = int(approved.sum())
        if n_app == 0:
            continue
        bads = int(y[approved].sum())
        goods = n_app - bads
        profit = goods * margin_per_good - bads * loss_per_bad
        rows.append({
            "cutoff_score": round(float(c), 1),
            "approval_rate": round(n_app / total, 4),
            "approved_bad_rate": round(bads / n_app, 4),
            "expected_profit": int(profit),
            "profit_per_app": round(profit / total, 2),
        })
    df = pd.DataFrame(rows)
    best = df.loc[df["expected_profit"].idxmax()].to_dict()
    return df, best


def swap_set_analysis(y_true, new_score, incumbent_value,
                      new_cutoff, incumbent_cutoff):
    """Scorecard vs incumbent flat-cutoff policy at comparable approval rates."""
    y = np.asarray(y_true).astype(int)
    new_approve = np.asarray(new_score) >= new_cutoff
    old_approve = np.asarray(incumbent_value) >= incumbent_cutoff

    def stats(mask):
        n = int(mask.sum())
        bad = int(y[mask].sum()) if n else 0
        return n, bad, (round(bad / n, 4) if n else 0.0)

    swap_in = new_approve & ~old_approve     # gained by scorecard
    swap_out = ~new_approve & old_approve    # lost by scorecard
    kept = new_approve & old_approve

    n_in, b_in, br_in = stats(swap_in)
    n_out, b_out, br_out = stats(swap_out)
    n_keep, b_keep, br_keep = stats(kept)

    summary = pd.DataFrame([
        {"group": "Approved by both", "n": n_keep, "bads": b_keep, "bad_rate": br_keep},
        {"group": "Swap-in (scorecard only)", "n": n_in, "bads": b_in, "bad_rate": br_in},
        {"group": "Swap-out (incumbent only)", "n": n_out, "bads": b_out, "bad_rate": br_out},
    ])
    verdict = {
        "swap_in_bad_rate": br_in,
        "swap_out_bad_rate": br_out,
        "scorecard_adds_value": bool(br_in < br_out),
        "new_approval_rate": round(float(new_approve.mean()), 4),
        "incumbent_approval_rate": round(float(old_approve.mean()), 4),
    }
    return summary, verdict


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
    woe_tr = enc.transform(Xtr)
    kept, _ = prune_multicollinearity(woe_tr[[f + "_woe" for f in feats]])
    sc = Scorecard(pdo=40, base_score=600, base_odds=20).fit(woe_tr, ytr, kept)

    s_te = sc.score(enc.transform(Xte))
    df, best = profit_optimised_cutoff(yte, s_te)
    print("Profit-optimal cutoff:", best)
    summary, verdict = swap_set_analysis(
        yte, s_te, Xte["bureau_score"].values,
        new_cutoff=best["cutoff_score"], incumbent_cutoff=620,
    )
    print(summary.to_string(index=False))
    print(verdict)
