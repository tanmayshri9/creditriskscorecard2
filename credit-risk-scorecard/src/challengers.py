"""
challengers.py
==============
Champion / challenger framework.

The logistic-regression scorecard is the *champion* (interpretable, regulator-
friendly, the production decision engine). We benchmark it against two
*challenger* machine-learning models named in modern risk stacks:

    * XGBoost (gradient-boosted trees)
    * Random Forest

Trees are fit on RAW features (with one-hot encoded categoricals) because they
capture non-linearities and interactions natively and do not require WOE.

The point of this benchmark is NOT simply "pick the highest Gini". It is to
quantify how much discriminatory power, if any, we leave on the table by
choosing an explainable model -- a core Model Risk Management trade-off. If the
challenger's lift is small, the interpretable champion wins on governance,
adverse-action explainability and stability. If the lift is large, it justifies
either a model change or a hybrid (e.g. ML-derived features fed into the
scorecard).

Author: Tanmay Shrivastava
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from validation import full_report


def _prep_raw(X_train, X_test):
    """One-hot encode categoricals; align columns across train/test."""
    cat = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    Xtr = pd.get_dummies(X_train, columns=cat, drop_first=True)
    Xte = pd.get_dummies(X_test, columns=cat, drop_first=True)
    Xtr, Xte = Xtr.align(Xte, join="left", axis=1, fill_value=0)
    return Xtr, Xte


def fit_challengers(X_train, y_train, X_test, y_test, exclude=("interest_rate",)):
    """Fit XGBoost + Random Forest, return models, predictions, metrics, importances."""
    drop = [c for c in exclude if c in X_train.columns]
    X_train = X_train.drop(columns=drop)
    X_test = X_test.drop(columns=drop)

    Xtr, Xte = _prep_raw(X_train, X_test)

    # class imbalance handling for XGBoost
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        min_child_weight=5,
        scale_pos_weight=spw,
        eval_metric="auc",
        n_jobs=4,
        random_state=42,
    ).fit(Xtr, y_train)

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=50,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=4,
        random_state=42,
    ).fit(Xtr, y_train)

    p_xgb = xgb.predict_proba(Xte)[:, 1]
    p_rf = rf.predict_proba(Xte)[:, 1]

    # for tree models, use -p_bad as the "score" so higher = safer for KS
    rep_xgb = full_report(y_test, p_xgb, -p_xgb, "XGBoost")
    rep_rf = full_report(y_test, p_rf, -p_rf, "RandomForest")

    imp = (
        pd.Series(xgb.feature_importances_, index=Xtr.columns)
        .sort_values(ascending=False)
        .head(12)
    )

    return {
        "models": {"xgb": xgb, "rf": rf},
        "pred": {"xgb": p_xgb, "rf": p_rf},
        "reports": [rep_xgb, rep_rf],
        "xgb_importance": imp,
        "feature_cols": Xtr.columns.tolist(),
    }


if __name__ == "__main__":
    from data_generation import generate_credit_data
    from sklearn.model_selection import train_test_split

    dev, _ = generate_credit_data()
    dev = dev.reset_index(drop=True)
    y = dev["default_flag"]
    X = dev.drop(columns=["default_flag", "vintage"])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          stratify=y, random_state=42)
    res = fit_challengers(Xtr, ytr, Xte, yte)
    for r in res["reports"]:
        print(r)
    print("\nTop XGBoost importances:")
    print(res["xgb_importance"].round(4))
