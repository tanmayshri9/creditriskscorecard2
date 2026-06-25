"""
monitoring.py
=============
Post-deployment model-health monitoring: PSI and CSI.

POPULATION STABILITY INDEX (PSI)
    Detects shift in the SCORE distribution between the development (expected)
    population and a new/recent (actual) population.

        PSI = Σ_bands ( %actual - %expected ) * ln( %actual / %expected )

    Interpretation:
        PSI < 0.10  -> stable, no action
        0.10-0.25   -> moderate shift, investigate & monitor
        PSI > 0.25  -> significant shift, model likely needs recalibration/redev

CHARACTERISTIC STABILITY INDEX (CSI)
    Same arithmetic, applied to each INPUT characteristic using the scorecard's
    bins. Tells you WHICH drivers are moving the population, so monitoring is
    actionable rather than just an alarm.

This is the quantitative core of the "model health dashboard / MIS" deliverable
that risk governance reviews each cycle.

Author: Tanmay Shrivastava
"""

import numpy as np
import pandas as pd


def _psi_from_distributions(expected_pct, actual_pct, eps=1e-4):
    expected_pct = np.clip(expected_pct, eps, None)
    actual_pct = np.clip(actual_pct, eps, None)
    return float(np.sum((actual_pct - expected_pct)
                        * np.log(actual_pct / expected_pct)))


def population_stability_index(expected_score, actual_score, n_bins=10):
    """PSI on the score distribution, with a per-band breakdown."""
    # build bands on the EXPECTED (baseline) distribution
    quantiles = np.quantile(expected_score, np.linspace(0, 1, n_bins + 1))
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    quantiles = np.unique(quantiles)

    exp_counts = pd.cut(expected_score, bins=quantiles).value_counts().sort_index()
    act_counts = pd.cut(actual_score, bins=quantiles).value_counts().sort_index()

    exp_pct = (exp_counts / exp_counts.sum()).values
    act_pct = (act_counts / act_counts.sum()).values

    contrib = (act_pct - exp_pct) * np.log(
        np.clip(act_pct, 1e-4, None) / np.clip(exp_pct, 1e-4, None)
    )
    table = pd.DataFrame({
        "band": [str(i) for i in exp_counts.index],
        "expected_pct": np.round(exp_pct, 4),
        "actual_pct": np.round(act_pct, 4),
        "psi_contribution": np.round(contrib, 4),
    })
    psi = float(contrib.sum())
    return psi, table


def characteristic_stability_index(encoder, expected_df, actual_df, features):
    """CSI per feature using the encoder's learned bins."""
    rows = []
    for woe_col in features:
        feat = woe_col[:-4]
        tbl = encoder.bins_[feat]

        if tbl.attrs.get("is_categorical", False):
            exp_pct, act_pct = _cat_distribution(tbl, expected_df[feat], actual_df[feat])
        else:
            exp_pct, act_pct = _num_distribution(tbl, expected_df[feat], actual_df[feat])

        csi = _psi_from_distributions(exp_pct, act_pct)
        rows.append({"characteristic": feat, "csi": round(csi, 4)})

    out = pd.DataFrame(rows).sort_values("csi", ascending=False).reset_index(drop=True)
    out["flag"] = pd.cut(out["csi"], bins=[-np.inf, 0.1, 0.25, np.inf],
                         labels=["stable", "moderate", "shifted"])
    return out


def _num_distribution(tbl, expected_s, actual_s):
    edges = tbl.attrs["edges"]
    exp = pd.cut(pd.to_numeric(expected_s, errors="coerce"),
                 bins=edges, include_lowest=True).value_counts().sort_index()
    act = pd.cut(pd.to_numeric(actual_s, errors="coerce"),
                 bins=edges, include_lowest=True).value_counts().sort_index()
    exp_pct = (exp / exp.sum()).values
    act_pct = (act / act.sum()).values
    return exp_pct, act_pct


def _cat_distribution(tbl, expected_s, actual_s):
    rare = tbl.attrs.get("rare_levels", [])
    levels = tbl["level"].tolist()

    def dist(s):
        s = s.astype(object).fillna("MISSING")
        s = s.where(~s.isin(rare), "OTHER")
        vc = s.value_counts()
        return np.array([vc.get(l, 0) for l in levels], dtype=float)

    exp = dist(expected_s)
    act = dist(actual_s)
    return exp / exp.sum(), act / act.sum()


if __name__ == "__main__":
    from data_generation import generate_credit_data
    from woe_iv import WOEEncoder
    from scorecard import Scorecard, select_features, prune_multicollinearity
    from sklearn.model_selection import train_test_split

    dev, oot = generate_credit_data()
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

    s_dev = sc.score(enc.transform(Xtr))
    Xoot = oot.drop(columns=["default_flag", "vintage"])
    s_oot = sc.score(enc.transform(Xoot))

    psi, psi_tbl = population_stability_index(s_dev, s_oot)
    print(f"Score PSI (dev -> OOT) = {psi:.4f}")
    print(psi_tbl.to_string(index=False))
    print("\nCSI by characteristic:")
    print(characteristic_stability_index(enc, Xtr, Xoot, kept).to_string(index=False))
