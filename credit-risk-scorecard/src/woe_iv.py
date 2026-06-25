"""
woe_iv.py
=========
Weight-of-Evidence (WOE) transformation and Information-Value (IV) engine.

This is the backbone of regulatory-grade credit scorecards. Rather than feeding
raw predictors into a model, each variable is binned and replaced by the WOE of
its bin:

        WOE_bin = ln( %Good_in_bin / %Bad_in_bin )

so a *higher* WOE always means *lower* risk. WOE has three properties risk teams
care about:

    * it linearises the relationship between predictor and log-odds, which is
      exactly what logistic regression assumes;
    * it handles outliers and missing values gracefully (they become a bin);
    * it produces a monotone, explainable transform that supports adverse-action
      reason codes and Model Risk Management review (SR 11-7).

Information Value ranks predictive power:

        IV = Σ_bins ( %Good - %Bad ) * WOE

    Rule of thumb:  <0.02 useless | 0.02-0.1 weak | 0.1-0.3 medium
                    0.3-0.5 strong | >0.5 suspiciously strong (check leakage)

Binning strategy:
    Numeric    -> supervised splits from a shallow decision tree (monotone,
                  respects a minimum bin population), with optional merging of
                  adjacent bins that violate WOE monotonicity.
    Categorical-> one bin per level (rare levels merged into 'OTHER').
    Missing    -> always its own bin.

Author: Tanmay Shrivastava
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


class WOEEncoder:
    def __init__(
        self,
        max_bins=6,
        min_bin_frac=0.05,
        enforce_monotonic=True,
        rare_level_frac=0.02,
        smooth=0.5,
    ):
        self.max_bins = max_bins
        self.min_bin_frac = min_bin_frac
        self.enforce_monotonic = enforce_monotonic
        self.rare_level_frac = rare_level_frac
        self.smooth = smooth  # Laplace smoothing on bin counts

        self.numeric_cols_ = []
        self.categorical_cols_ = []
        self.bins_ = {}          # feature -> bin definition / WOE table
        self.iv_ = {}            # feature -> IV
        self.total_good_ = None
        self.total_bad_ = None

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #
    def fit(self, X, y, numeric_cols=None, categorical_cols=None):
        X = X.copy()
        y = np.asarray(y).astype(int)
        self.total_good_ = (y == 0).sum()
        self.total_bad_ = (y == 1).sum()

        if numeric_cols is None:
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if categorical_cols is None:
            categorical_cols = [c for c in X.columns if c not in numeric_cols]

        self.numeric_cols_ = numeric_cols
        self.categorical_cols_ = categorical_cols

        for col in numeric_cols:
            self.bins_[col] = self._fit_numeric(X[col], y)
            self.iv_[col] = self.bins_[col]["iv"].sum()

        for col in categorical_cols:
            self.bins_[col] = self._fit_categorical(X[col], y)
            self.iv_[col] = self.bins_[col]["iv"].sum()

        return self

    # ------------------------------------------------------------------ #
    # Numeric binning
    # ------------------------------------------------------------------ #
    def _fit_numeric(self, s, y):
        s = pd.to_numeric(s, errors="coerce")
        notna = s.notna().values
        x = s[notna].values.reshape(-1, 1)
        yy = y[notna]

        n = len(s)
        min_leaf = max(int(self.min_bin_frac * n), 50)

        # supervised split points via a shallow tree
        tree = DecisionTreeClassifier(
            max_leaf_nodes=self.max_bins,
            min_samples_leaf=min_leaf,
            random_state=42,
        )
        tree.fit(x, yy)
        thr = np.sort(
            tree.tree_.threshold[tree.tree_.threshold != -2]
        )
        edges = np.concatenate([[-np.inf], thr, [np.inf]])
        # de-duplicate edges (can happen on discrete vars)
        edges = np.unique(edges)

        table = self._woe_table_from_edges(s, y, edges)

        if self.enforce_monotonic:
            table, edges = self._merge_for_monotonicity(s, y, edges)

        table.attrs["edges"] = edges
        return table

    def _woe_table_from_edges(self, s, y, edges):
        rows = []
        # finite bins
        binned = pd.cut(s, bins=edges, include_lowest=True, duplicates="drop")
        for interval, idx in binned.groupby(binned, observed=True).groups.items():
            mask = s.index.isin(idx)
            rows.append(self._bin_stats(str(interval), y[mask.nonzero()[0]], mask.sum(),
                                        lo=interval.left, hi=interval.right))
        # missing bin
        miss_mask = s.isna().values
        if miss_mask.sum() > 0:
            rows.append(self._bin_stats("MISSING", y[miss_mask], miss_mask.sum(),
                                        lo=np.nan, hi=np.nan, is_missing=True))

        table = pd.DataFrame(rows)
        return self._finalize_table(table)

    def _merge_for_monotonicity(self, s, y, edges):
        """Greedily merge adjacent bins until WOE is monotone in the bin index."""
        edges = list(edges)
        while True:
            table = self._woe_table_from_edges(
                s, y, np.array(edges)
            )
            woe = table.loc[table["bin"] != "MISSING", "woe"].values
            if len(woe) <= 2:
                break
            diffs = np.diff(woe)
            # find first sign change (non-monotone point)
            sign = np.sign(diffs)
            nonmono = np.where(sign[:-1] * sign[1:] < 0)[0]
            if len(nonmono) == 0:
                break
            # merge the offending interior edge
            edge_to_drop = nonmono[0] + 1  # index into interior edges
            interior = edges[1:-1]
            if not interior:
                break
            interior.pop(min(edge_to_drop, len(interior) - 1))
            edges = [edges[0]] + interior + [edges[-1]]
            if len(edges) <= 3:
                break
        return self._woe_table_from_edges(s, y, np.array(edges)), np.array(edges)

    # ------------------------------------------------------------------ #
    # Categorical binning
    # ------------------------------------------------------------------ #
    def _fit_categorical(self, s, y):
        s = s.astype(object).fillna("MISSING")
        counts = s.value_counts(normalize=True)
        rare = counts[counts < self.rare_level_frac].index.tolist()
        s2 = s.where(~s.isin(rare), other="OTHER")

        rows = []
        for level in pd.unique(s2):
            mask = (s2 == level).values
            rows.append(self._bin_stats(str(level), y[mask], mask.sum(),
                                        lo=np.nan, hi=np.nan, level=str(level)))
        table = pd.DataFrame(rows)
        table = self._finalize_table(table)
        table.attrs["rare_levels"] = rare
        table.attrs["is_categorical"] = True
        return table

    # ------------------------------------------------------------------ #
    # Shared stats
    # ------------------------------------------------------------------ #
    def _bin_stats(self, label, y_bin, n, lo=np.nan, hi=np.nan,
                   is_missing=False, level=None):
        n_bad = int(np.sum(y_bin))
        n_good = int(n - n_bad)
        return {
            "bin": label,
            "lo": lo,
            "hi": hi,
            "level": level,
            "is_missing": is_missing,
            "n": int(n),
            "n_good": n_good,
            "n_bad": n_bad,
        }

    def _finalize_table(self, table):
        s = self.smooth
        tg, tb = self.total_good_, self.total_bad_
        table["event_rate"] = table["n_bad"] / table["n"]
        table["dist_good"] = (table["n_good"] + s) / (tg + s * len(table))
        table["dist_bad"] = (table["n_bad"] + s) / (tb + s * len(table))
        table["woe"] = np.log(table["dist_good"] / table["dist_bad"])
        table["iv"] = (table["dist_good"] - table["dist_bad"]) * table["woe"]
        table["pct_total"] = table["n"] / table["n"].sum()
        return table.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Transform
    # ------------------------------------------------------------------ #
    def transform(self, X):
        out = pd.DataFrame(index=X.index)
        for col in self.numeric_cols_:
            out[col + "_woe"] = self._transform_numeric(X[col], self.bins_[col])
        for col in self.categorical_cols_:
            out[col + "_woe"] = self._transform_categorical(X[col], self.bins_[col])
        return out

    def _transform_numeric(self, s, table):
        s = pd.to_numeric(s, errors="coerce")
        edges = table.attrs["edges"]
        result = pd.Series(np.nan, index=s.index, dtype=float)

        finite = table[table["bin"] != "MISSING"]
        binned = pd.cut(s, bins=edges, include_lowest=True)
        # map interval -> woe by left edge order
        woe_by_bin = {row["bin"]: row["woe"] for _, row in finite.iterrows()}
        result = binned.astype(str).map(woe_by_bin)

        # missing
        miss_woe = table.loc[table["bin"] == "MISSING", "woe"]
        if len(miss_woe):
            result = result.where(s.notna(), miss_woe.iloc[0])
        else:
            # if no missing bin learned, impute with neutral (0) WOE
            result = result.where(s.notna(), 0.0)
        return result.astype(float).values

    def _transform_categorical(self, s, table):
        s = s.astype(object).fillna("MISSING")
        rare = table.attrs.get("rare_levels", [])
        s = s.where(~s.isin(rare), other="OTHER")
        woe_map = {row["level"]: row["woe"] for _, row in table.iterrows()}
        default_woe = 0.0
        return s.map(woe_map).fillna(default_woe).astype(float).values

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def iv_summary(self):
        df = (
            pd.Series(self.iv_, name="IV")
            .sort_values(ascending=False)
            .to_frame()
        )
        df["strength"] = pd.cut(
            df["IV"],
            bins=[-np.inf, 0.02, 0.1, 0.3, 0.5, np.inf],
            labels=["useless", "weak", "medium", "strong", "very_strong"],
        )
        return df

    def binning_table(self, feature):
        cols = ["bin", "n", "pct_total", "n_good", "n_bad",
                "event_rate", "woe", "iv"]
        return self.bins_[feature][cols].copy()


if __name__ == "__main__":
    from data_generation import generate_credit_data

    dev, _ = generate_credit_data()
    y = dev["default_flag"]
    X = dev.drop(columns=["default_flag", "vintage"])

    enc = WOEEncoder(max_bins=6, min_bin_frac=0.05).fit(X, y)
    print("=== Information Value ranking ===")
    print(enc.iv_summary().round(4))
    print("\n=== Binning table: bureau_score ===")
    print(enc.binning_table("bureau_score").round(4).to_string(index=False))
    print("\n=== Binning table: residential_status ===")
    print(enc.binning_table("residential_status").round(4).to_string(index=False))
