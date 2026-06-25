"""
scorecard.py
============
Feature selection + logistic-regression scorecard, scaled to points.

Pipeline
--------
1. FEATURE SELECTION (regulatory-grade, defensible):
      * keep predictors with Information Value in a usable band
      * drop endogenous / leaky variables (e.g. offered interest rate, which is
        priced off the very risk we are predicting)
      * remove multicollinearity via Variance Inflation Factor (VIF)
      * after fitting, drop any predictor whose coefficient sign is
        counter-intuitive (positive WOE must lower the odds of default)

2. MODEL:
      * Logistic Regression on WOE-transformed inputs. Logistic regression is
        the banking standard for application scorecards: it is monotone,
        additive, fully transparent, and maps cleanly to points + reason codes,
        which Model Risk Management (SR 11-7) and fair-lending review require.

3. SCALING TO POINTS:
      Score = Offset + Factor * ln(odds_good)
        Factor = PDO / ln(2)
        Offset = BaseScore - Factor * ln(BaseOdds)
      Default anchors: 600 points @ 50:1 good:bad odds, PDO = 20
      (every 20 points doubles the odds of being good).

      Each attribute (bin) contributes:
        points = -(beta_i * WOE_i + beta_0 / n) * Factor + Offset / n

Author: Tanmay Shrivastava
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# --------------------------------------------------------------------------- #
# Feature selection helpers
# --------------------------------------------------------------------------- #
def variance_inflation_factors(woe_df):
    """Compute VIF for each WOE column (no statsmodels dependency)."""
    cols = woe_df.columns.tolist()
    vifs = {}
    X = woe_df.values
    for i, c in enumerate(cols):
        y_i = X[:, i]
        X_other = np.delete(X, i, axis=1)
        X_other = np.column_stack([np.ones(len(X_other)), X_other])
        # OLS via lstsq
        beta, *_ = np.linalg.lstsq(X_other, y_i, rcond=None)
        pred = X_other @ beta
        ss_res = np.sum((y_i - pred) ** 2)
        ss_tot = np.sum((y_i - y_i.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vifs[c] = np.inf if r2 >= 1 else 1.0 / (1.0 - r2)
    return pd.Series(vifs, name="VIF").sort_values(ascending=False)


def select_features(
    iv_series,
    exclude=("interest_rate",),
    iv_floor=0.02,
    iv_ceiling=2.0,
):
    """Return ordered list of candidate features by IV band, minus exclusions."""
    keep = []
    for feat, iv in iv_series.items():
        if feat in exclude:
            continue
        if iv_floor <= iv <= iv_ceiling:
            keep.append(feat)
    # preserve IV order
    keep = sorted(keep, key=lambda f: iv_series[f], reverse=True)
    return keep


def prune_multicollinearity(woe_df, vif_threshold=5.0):
    """Iteratively drop the highest-VIF feature until all VIF < threshold."""
    df = woe_df.copy()
    dropped = []
    while df.shape[1] > 1:
        vifs = variance_inflation_factors(df)
        worst, worst_vif = vifs.index[0], vifs.iloc[0]
        if worst_vif < vif_threshold:
            break
        dropped.append((worst, round(float(worst_vif), 2)))
        df = df.drop(columns=[worst])
    return df.columns.tolist(), dropped


# --------------------------------------------------------------------------- #
# Scorecard model
# --------------------------------------------------------------------------- #
class Scorecard:
    def __init__(self, pdo=20, base_score=600, base_odds=50, C=1.0):
        self.pdo = pdo
        self.base_score = base_score
        self.base_odds = base_odds
        self.C = C
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)

        self.model_ = None
        self.features_ = None          # WOE column names used
        self.coef_ = None
        self.intercept_ = None
        self.points_table_ = None

    def fit(self, woe_df, y, features):
        self.features_ = features
        X = woe_df[features].values
        self.model_ = LogisticRegression(
            C=self.C, solver="lbfgs", max_iter=1000
        ).fit(X, y)
        self.coef_ = dict(zip(features, self.model_.coef_[0]))
        self.intercept_ = float(self.model_.intercept_[0])
        return self

    def wrong_sign_features(self):
        """WOE coef should be NEGATIVE (higher WOE -> safer -> lower P(bad))."""
        return [f for f, b in self.coef_.items() if b > 0]

    def predict_proba(self, woe_df):
        X = woe_df[self.features_].values
        return self.model_.predict_proba(X)[:, 1]

    def score(self, woe_df):
        """Return scaled credit scores (higher = safer)."""
        X = woe_df[self.features_].values
        n = len(self.features_)
        betas = np.array([self.coef_[f] for f in self.features_])
        # per-row contribution
        contrib = -(X * betas + self.intercept_ / n) * self.factor + self.offset / n
        return contrib.sum(axis=1)

    # ---- scorecard points table (the artefact underwriters read) --------- #
    def build_points_table(self, encoder):
        """Translate WOE bins -> points using the scaling decomposition."""
        n = len(self.features_)
        rows = []
        for woe_col in self.features_:
            raw_feat = woe_col[:-4]  # strip '_woe'
            beta = self.coef_[woe_col]
            tbl = encoder.bins_[raw_feat]
            for _, b in tbl.iterrows():
                pts = -(beta * b["woe"] + self.intercept_ / n) * self.factor \
                      + self.offset / n
                rows.append({
                    "feature": raw_feat,
                    "bin": b["bin"],
                    "n": int(b["n"]),
                    "event_rate": round(float(b["event_rate"]), 4),
                    "woe": round(float(b["woe"]), 4),
                    "beta": round(float(beta), 4),
                    "points": int(round(pts)),
                })
        self.points_table_ = pd.DataFrame(rows)
        return self.points_table_

    def score_range(self):
        """Theoretical min/max achievable score from the points table."""
        pt = self.points_table_
        mn = pt.groupby("feature")["points"].min().sum()
        mx = pt.groupby("feature")["points"].max().sum()
        return int(mn), int(mx)


if __name__ == "__main__":
    from data_generation import generate_credit_data
    from woe_iv import WOEEncoder
    from sklearn.model_selection import train_test_split

    dev, _ = generate_credit_data()
    dev = dev.reset_index(drop=True)
    y = dev["default_flag"]
    X = dev.drop(columns=["default_flag", "vintage"])

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    enc = WOEEncoder(max_bins=6, min_bin_frac=0.05).fit(Xtr, ytr)
    iv = enc.iv_summary()["IV"]

    feats_raw = select_features(iv, exclude=("interest_rate",))
    woe_tr = enc.transform(Xtr)
    woe_cols = [f + "_woe" for f in feats_raw]
    kept, dropped = prune_multicollinearity(woe_tr[woe_cols], vif_threshold=5.0)
    print("Dropped for multicollinearity:", dropped)

    sc = Scorecard().fit(woe_tr, ytr, kept)
    wrong = sc.wrong_sign_features()
    if wrong:
        print("Dropping wrong-sign features:", wrong)
        kept = [f for f in kept if f not in wrong]
        sc = Scorecard().fit(woe_tr, ytr, kept)

    print("\nFinal features:", kept)
    print("Intercept:", round(sc.intercept_, 4))
    pts = sc.build_points_table(enc)
    print("Score range:", sc.score_range())
    print("\nPoints table (bureau_score):")
    print(pts[pts.feature == "bureau_score"].to_string(index=False))
