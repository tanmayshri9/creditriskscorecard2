"""
run_pipeline.py
===============
End-to-end orchestration of the credit-risk scorecard build.

Stages
------
    1. Data        : generate development + out-of-time books
    2. Split       : stratified train / in-time test
    3. WOE / IV    : fit encoder, rank predictive power
    4. Selection   : IV band -> drop endogenous -> VIF prune -> sign check
    5. Scorecard   : logistic regression, scaled to points (PDO/base odds)
    6. Validation  : KS, Gini/AUC, rank-ordering, calibration (test + OOT)
    7. Challengers : XGBoost + Random Forest benchmark
    8. Monitoring  : PSI (score) + CSI (characteristics) dev -> OOT
    9. Business    : strategy curve, profit-optimal cutoff, swap-set
   10. Artifacts   : write every figure + results.json for the report

Run:  python src/run_pipeline.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

from data_generation import generate_credit_data
from woe_iv import WOEEncoder
from scorecard import (Scorecard, select_features, prune_multicollinearity,
                       variance_inflation_factors)
import validation as val
from challengers import fit_challengers
import monitoring as mon
import business_impact as biz
import viz

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT, exist_ok=True)

R = {}  # results dict serialised to JSON for the report


def banner(msg):
    print("\n" + "=" * 70)
    print(msg)
    print("=" * 70)


# --------------------------------------------------------------------------- #
def main():
    banner("STAGE 1-2  Data + split")
    dev, oot = generate_credit_data(n_dev=60_000, n_oot=20_000)
    dev = dev.reset_index(drop=True)
    oot = oot.reset_index(drop=True)
    y = dev["default_flag"]
    X = dev.drop(columns=["default_flag", "vintage"])
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    Xoot = oot.drop(columns=["default_flag", "vintage"])
    yoot = oot["default_flag"]

    R["data"] = {
        "n_total": int(len(dev) + len(oot)),
        "n_dev": int(len(dev)),
        "n_oot": int(len(oot)),
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "dev_bad_rate": round(float(y.mean()), 4),
        "oot_bad_rate": round(float(yoot.mean()), 4),
        "n_features_raw": int(X.shape[1]),
        "feature_families": {
            "application": ["age", "annual_income", "employment_length_yrs",
                            "residential_status", "employment_status"],
            "bureau": ["bureau_score", "months_on_book", "revolving_utilization",
                       "num_delinq_24m", "num_inquiries_6m", "num_open_trades",
                       "num_public_records"],
            "behavioural_alt": ["requested_loan_amt", "dti_ratio", "interest_rate",
                                "digital_engagement_score"],
        },
    }
    print(f"Dev {len(dev):,} (bad {y.mean()*100:.2f}%) | "
          f"OOT {len(oot):,} (bad {yoot.mean()*100:.2f}%)")
    viz.plot_target_overview(dev)

    banner("STAGE 3  WOE / IV")
    enc = WOEEncoder(max_bins=6, min_bin_frac=0.05, enforce_monotonic=True)
    enc.fit(Xtr, ytr)
    iv_summary = enc.iv_summary()
    R["iv"] = {k: round(float(v), 4) for k, v in enc.iv_.items()}
    print(iv_summary.round(4).to_string())
    viz.plot_iv_ranking(iv_summary[["IV"]].copy())

    # export a couple of binning tables for the report
    R["binning_examples"] = {
        f: enc.binning_table(f).round(4).to_dict(orient="records")
        for f in ["bureau_score", "revolving_utilization", "num_delinq_24m"]
    }

    banner("STAGE 4  Feature selection")
    candidates = select_features(iv_summary["IV"], exclude=("interest_rate",))
    print("After IV band + endogenous exclusion:", candidates)
    woe_tr = enc.transform(Xtr)
    woe_te = enc.transform(Xte)
    woe_oot = enc.transform(Xoot)

    cand_cols = [f + "_woe" for f in candidates]
    kept_cols, dropped_vif = prune_multicollinearity(woe_tr[cand_cols],
                                                     vif_threshold=5.0)
    print("Dropped (VIF):", dropped_vif)

    sc = Scorecard(pdo=40, base_score=600, base_odds=20).fit(woe_tr, ytr, kept_cols)
    wrong = sc.wrong_sign_features()
    if wrong:
        print("Dropped (wrong sign):", wrong)
        kept_cols = [c for c in kept_cols if c not in wrong]
        sc = Scorecard(pdo=40, base_score=600, base_odds=20).fit(woe_tr, ytr, kept_cols)

    final_feats = [c[:-4] for c in kept_cols]
    vif_final = variance_inflation_factors(woe_tr[kept_cols])
    R["feature_selection"] = {
        "excluded_endogenous": ["interest_rate"],
        "dropped_vif": dropped_vif,
        "dropped_wrong_sign": wrong,
        "final_features": final_feats,
        "n_final": len(final_feats),
        "final_vif": {k[:-4]: round(float(v), 2) for k, v in vif_final.items()},
        "coefficients": {k[:-4]: round(float(v), 4) for k, v in sc.coef_.items()},
        "intercept": round(sc.intercept_, 4),
    }
    print("FINAL FEATURES:", final_feats)

    viz.plot_woe_grid(enc, final_feats)
    viz.plot_corr_heatmap(woe_tr, kept_cols)

    banner("STAGE 5  Scorecard scaling")
    pts = sc.build_points_table(enc)
    smin, smax = sc.score_range()
    R["scorecard"] = {
        "pdo": sc.pdo, "base_score": sc.base_score, "base_odds": sc.base_odds,
        "factor": round(sc.factor, 4), "offset": round(sc.offset, 4),
        "score_min": smin, "score_max": smax,
        "points_table": pts.to_dict(orient="records"),
    }
    print(f"Score range: {smin}–{smax}")
    print(pts[pts.feature == "bureau_score"].to_string(index=False))

    banner("STAGE 6  Validation")
    p_tr = sc.predict_proba(woe_tr); s_tr = sc.score(woe_tr)
    p_te = sc.predict_proba(woe_te); s_te = sc.score(woe_te)
    p_oot = sc.predict_proba(woe_oot); s_oot = sc.score(woe_oot)

    rep_train = val.full_report(ytr, p_tr, s_tr, "Train")
    rep_test = val.full_report(yte, p_te, s_te, "Test (in-time)")
    rep_oot = val.full_report(yoot, p_oot, s_oot, "Out-of-time")
    R["validation"] = {"champion": [rep_train, rep_test, rep_oot]}
    for r in (rep_train, rep_test, rep_oot):
        print(r)

    band_tbl = val.score_band_table(yte, s_te, n_bands=10)
    cal_tbl = val.calibration_table(yte, p_te, n_bins=10)
    R["score_band_table"] = band_tbl.round(4).to_dict(orient="records")
    R["calibration_table"] = cal_tbl.round(4).to_dict(orient="records")

    # operating point near profit-optimal region
    conf = val.confusion_at_cutoff(yte, p_te, threshold=0.15)
    R["confusion_example"] = conf
    print("Confusion @ PD>=0.15:", conf)

    viz.plot_ks_curve(yte, s_te, rep_test["ks"])
    viz.plot_score_distribution(yte, s_te)
    viz.plot_rank_ordering(band_tbl)
    viz.plot_calibration(cal_tbl)

    banner("STAGE 7  Challengers")
    ch = fit_challengers(Xtr, ytr, Xte, yte, exclude=("interest_rate",))
    all_reports = [rep_test] + ch["reports"]
    # relabel champion
    champ = dict(rep_test); champ["segment"] = "Logistic (champion)"
    all_reports = [champ] + ch["reports"]
    R["validation"]["model_comparison"] = all_reports
    R["xgb_importance"] = {k: round(float(v), 4)
                           for k, v in ch["xgb_importance"].items()}

    # ROC curves
    fpr_c, tpr_c = val.roc_points(yte, p_te)
    fpr_x, tpr_x = val.roc_points(yte, ch["pred"]["xgb"])
    fpr_r, tpr_r = val.roc_points(yte, ch["pred"]["rf"])
    viz.plot_roc([
        ("Logistic (champion)", fpr_c, tpr_c, champ["auc"], viz.INK),
        ("XGBoost", fpr_x, tpr_x, ch["reports"][0]["auc"], viz.TEAL),
        ("Random Forest", fpr_r, tpr_r, ch["reports"][1]["auc"], viz.AMBER),
    ])
    viz.plot_model_comparison(all_reports)
    viz.plot_feature_importance(ch["xgb_importance"])

    banner("STAGE 8  Monitoring (PSI / CSI)")
    psi, psi_tbl = mon.population_stability_index(s_tr, s_oot, n_bins=10)
    csi = mon.characteristic_stability_index(enc, Xtr, Xoot, kept_cols)
    R["monitoring"] = {
        "score_psi": round(psi, 4),
        "psi_table": psi_tbl.to_dict(orient="records"),
        "csi": csi.to_dict(orient="records"),
        "oot_gini_drop": round(rep_test["gini"] - rep_oot["gini"], 4),
    }
    print(f"Score PSI = {psi:.4f}")
    print(csi.to_string(index=False))
    viz.plot_psi(psi_tbl, psi)
    viz.plot_csi(csi)

    banner("STAGE 9  Business impact")
    strat = biz.strategy_curve(yte, s_te, n_steps=25)
    profit_df, best = biz.profit_optimised_cutoff(
        yte, s_te, margin_per_good=1_200, loss_per_bad=9_000)
    swap_summary, swap_verdict = biz.swap_set_analysis(
        yte, s_te, Xte["bureau_score"].values,
        new_cutoff=best["cutoff_score"], incumbent_cutoff=620)
    R["business"] = {
        "strategy_curve": strat.to_dict(orient="records"),
        "profit_best": {k: (float(v) if isinstance(v, (int, float, np.floating))
                            else v) for k, v in best.items()},
        "unit_economics": {"margin_per_good": 1200, "loss_per_bad": 9000},
        "swap_set": swap_summary.to_dict(orient="records"),
        "swap_verdict": swap_verdict,
        "through_door_bad_rate": round(float(yte.mean()), 4),
    }
    print("Profit-optimal:", best)
    print(swap_summary.to_string(index=False))
    print("Verdict:", swap_verdict)
    viz.plot_strategy_curve(strat)
    viz.plot_profit_curve(profit_df, best)
    viz.plot_swap_set(swap_summary)

    banner("STAGE 10  Persist results.json")
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(R, f, indent=2, default=str)
    print("Wrote outputs/results.json")
    print("Figures in outputs/figures/:")
    for fpng in sorted(os.listdir(os.path.join(OUT, "figures"))):
        print("   ", fpng)


if __name__ == "__main__":
    main()
