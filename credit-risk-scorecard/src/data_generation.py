"""
data_generation.py
===================
Generates a realistic, synthetic consumer-lending dataset for Probability of
Default (PD) scorecard development.

The data emulates an unsecured personal-loan / credit-card book and combines
three data families that mirror what a real lender works with:

    1. Application / demographic data   (age, income, employment, residence)
    2. Credit-bureau attributes         (bureau score, utilisation, delinquencies,
                                          inquiries, trades, public records)
    3. Behavioural / alternative data    (DTI, digital engagement, tenure)

A latent log-odds of default is built from these drivers so that downstream
Weight-of-Evidence, Information-Value and model results are economically
sensible (monotone risk, realistic Gini/KS, ~9-11% bad rate).

A second *out-of-time* (OOT) population is generated with deliberate
distribution drift + macro deterioration so that PSI / CSI monitoring has
something real to detect.

Author: Tanmay Shrivastava
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20240517)


# --------------------------------------------------------------------------- #
# Helper distributions
# --------------------------------------------------------------------------- #
def _truncated_normal(mean, sd, low, high, size, rng):
    out = rng.normal(mean, sd, size)
    return np.clip(out, low, high)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------- #
# Core generator
# --------------------------------------------------------------------------- #
def _generate_population(n, rng, regime="development"):
    """Generate one population of `n` applicants.

    `regime` controls macro / mix shifts:
        development -> base economy, base customer mix
        recent      -> out-of-time book: thinner files, higher utilisation,
                       macro stress -> higher realised bad rate
    """
    # ---- macro / mix levers ------------------------------------------------ #
    if regime == "development":
        util_shift, score_shift, macro_logodds = 0.0, 0.0, 0.0
        thin_file_p = 0.18
    else:  # recent / out-of-time
        util_shift, score_shift, macro_logodds = 12.0, -22.0, 0.18
        thin_file_p = 0.34

    # ---- 1. Application / demographic ------------------------------------- #
    age = _truncated_normal(41, 12, 19, 84, n, rng).round().astype(int)

    # income correlated with age, lognormal-ish
    income = np.exp(rng.normal(10.85, 0.45, n)) * (1 + (age - 41) / 220)
    income = np.clip(income, 14_000, 400_000).round(-2)

    employment_length = np.clip(
        rng.gamma(shape=2.0, scale=3.2, size=n) + (age - 25) / 18, 0, 45
    ).round(1)

    residential_status = rng.choice(
        ["OWN", "MORTGAGE", "RENT", "OTHER"],
        size=n,
        p=[0.18, 0.41, 0.36, 0.05],
    )

    employment_status = rng.choice(
        ["SALARIED", "SELF_EMPLOYED", "CONTRACT", "RETIRED", "UNEMPLOYED"],
        size=n,
        p=[0.58, 0.20, 0.12, 0.07, 0.03],
    )

    # ---- 2. Bureau attributes --------------------------------------------- #
    thin_file = rng.random(n) < thin_file_p
    months_on_book = np.where(
        thin_file,
        _truncated_normal(14, 8, 0, 60, n, rng),
        _truncated_normal(96, 48, 6, 360, n, rng),
    ).round().astype(int)

    # FICO-like bureau score, anchored to age/income/tenure with noise
    base_score = (
        640
        + (age - 41) * 0.9
        + (np.log(income) - 10.85) * 22
        + employment_length * 1.4
        + (months_on_book - 60) * 0.10
        + score_shift
    )
    bureau_score = _truncated_normal(base_score, 55, 300, 850, n, rng).round().astype(int)

    # revolving utilisation %, inversely related to score
    util_mean = 55 - (bureau_score - 640) * 0.16 + util_shift
    revolving_utilization = np.clip(
        rng.normal(util_mean, 22, n), 0, 150
    ).round(1)

    # delinquencies in last 24m: more likely at low score / high utilisation
    delinq_lambda = np.exp(
        -1.4 - (bureau_score - 640) / 90 + revolving_utilization / 120
    )
    num_delinq_24m = rng.poisson(delinq_lambda).clip(0, 12)

    # credit inquiries last 6m
    inq_lambda = np.exp(-0.5 - (bureau_score - 640) / 160 + revolving_utilization / 200)
    num_inquiries_6m = rng.poisson(inq_lambda).clip(0, 15)

    # open trade lines
    num_open_trades = np.clip(
        rng.poisson(6 + (income / 60_000)) + (months_on_book // 36), 0, 35
    )

    # public-record derogatories (rare)
    pub_rec_lambda = np.exp(-3.2 - (bureau_score - 640) / 70)
    num_public_records = rng.poisson(pub_rec_lambda).clip(0, 6)

    # ---- 3. Behavioural / alternative ------------------------------------- #
    requested_loan = np.clip(
        rng.normal(income * 0.32, income * 0.18), 1_000, 75_000
    ).round(-2)

    # debt-to-income: higher loan / lower income / higher utilisation
    dti = np.clip(
        (requested_loan / income) * 100 * 0.55
        + revolving_utilization * 0.25
        + rng.normal(8, 6, n),
        0,
        90,
    ).round(1)

    interest_rate = np.clip(
        9.5 + (700 - bureau_score) * 0.035 + dti * 0.05 + rng.normal(0, 1.4, n),
        4.0,
        34.0,
    ).round(2)

    # alternative data: digital engagement score 0-100 (proxy for app usage,
    # on-time digital payments, verified data depth)
    digital_engagement = np.clip(
        rng.normal(55, 18, n) + (bureau_score - 640) * 0.05, 0, 100
    ).round().astype(int)

    # ---- Latent log-odds of DEFAULT --------------------------------------- #
    # Centred / scaled drivers so coefficients are interpretable in log-odds.
    z = (
        -2.95  # intercept -> base rate
        - 0.0090 * (bureau_score - 640)
        + 0.0140 * (revolving_utilization - 45)
        + 0.2600 * num_delinq_24m
        + 0.1500 * num_inquiries_6m
        + 0.0180 * (dti - 30)
        + 0.4200 * num_public_records
        - 0.0150 * (age - 41)
        - 0.0090 * (digital_engagement - 55)
        - 0.0060 * employment_length
        - 0.0008 * (months_on_book - 60)
        + np.where(employment_status == "UNEMPLOYED", 0.55, 0.0)
        + np.where(employment_status == "CONTRACT", 0.18, 0.0)
        + np.where(residential_status == "RENT", 0.20, 0.0)
        + np.where(residential_status == "OWN", -0.12, 0.0)
        + macro_logodds
        + rng.normal(0, 0.45, n)  # idiosyncratic noise -> realistic, imperfect signal
    )

    pd_true = _sigmoid(z)
    default_flag = (rng.random(n) < pd_true).astype(int)

    df = pd.DataFrame(
        {
            # application
            "age": age,
            "annual_income": income,
            "employment_length_yrs": employment_length,
            "residential_status": residential_status,
            "employment_status": employment_status,
            # bureau
            "bureau_score": bureau_score,
            "months_on_book": months_on_book,
            "revolving_utilization": revolving_utilization,
            "num_delinq_24m": num_delinq_24m,
            "num_inquiries_6m": num_inquiries_6m,
            "num_open_trades": num_open_trades,
            "num_public_records": num_public_records,
            # behavioural / alternative
            "requested_loan_amt": requested_loan,
            "dti_ratio": dti,
            "interest_rate": interest_rate,
            "digital_engagement_score": digital_engagement,
            # target
            "default_flag": default_flag,
        }
    )
    return df


def generate_credit_data(n_dev=60_000, n_oot=20_000, seed=20240517):
    """Return (development_df, out_of_time_df).

    development_df : in-time book used for build + in-time validation (split later)
    out_of_time_df : a later booking window with drift, used for OOT validation
                     and PSI/CSI monitoring demonstrations.
    """
    rng = np.random.default_rng(seed)
    dev = _generate_population(n_dev, rng, regime="development")
    oot = _generate_population(n_oot, rng, regime="recent")

    # add a vintage label
    dev["vintage"] = "2023H1_DEV"
    oot["vintage"] = "2024H1_OOT"
    return dev, oot


if __name__ == "__main__":
    dev, oot = generate_credit_data()
    print("Development sample :", dev.shape, "| bad rate =",
          round(dev.default_flag.mean() * 100, 2), "%")
    print("Out-of-time sample :", oot.shape, "| bad rate =",
          round(oot.default_flag.mean() * 100, 2), "%")
    print("\nDev dtypes:\n", dev.dtypes)
    print("\nHead:\n", dev.head())
