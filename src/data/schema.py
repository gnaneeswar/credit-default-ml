"""Canonical schema for UCI dataset 350 (Default of Credit Card Clients, Taiwan 2005).

The UCI variables table names features X1..X23 and the target Y; their descriptions carry
the conventional domain names. Note: repayment status uses PAY_0, PAY_2..PAY_6 (there is
no PAY_1) — this follows the original dataset and every public reference to it.
"""

from __future__ import annotations

UCI_ID = 350
ID_COL = "ID"
RAW_TARGET = "Y"
TARGET = "default_payment_next_month"

PAY_STATUS_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_AMT_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
DEMOGRAPHIC_COLS = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]

FEATURES = ["LIMIT_BAL", *DEMOGRAPHIC_COLS, *PAY_STATUS_COLS, *BILL_AMT_COLS, *PAY_AMT_COLS]
CATEGORICAL_CODE_COLS = ["SEX", "EDUCATION", "MARRIAGE", *PAY_STATUS_COLS]

RAW_TO_CANONICAL: dict[str, str] = {f"X{i}": name for i, name in enumerate(FEATURES, start=1)}
RAW_TO_CANONICAL[RAW_TARGET] = TARGET

# Position 1 = most recent month. PAY_0 / BILL_AMT1 / PAY_AMT1 all refer to September 2005.
MONTHS = ["2005-09", "2005-08", "2005-07", "2005-06", "2005-05", "2005-04"]
TIMELINE = [
    {"month": m, "pay_status": p, "bill_amt": b, "pay_amt": a}
    for m, p, b, a in zip(MONTHS, PAY_STATUS_COLS, BILL_AMT_COLS, PAY_AMT_COLS, strict=True)
]

# Codes as documented by UCI.
DOCUMENTED_CODES: dict[str, dict[int, str]] = {
    "SEX": {1: "male", 2: "female"},
    "EDUCATION": {1: "graduate school", 2: "university", 3: "high school", 4: "others"},
    "MARRIAGE": {1: "married", 2: "single", 3: "others"},
    "PAY_n": {
        -1: "pay duly (paid in full)",
        **{k: f"payment delay for {k} month{'s' if k > 1 else ''}" for k in range(1, 9)},
        9: "payment delay for nine months and above",
    },
}

# Codes present in the data but absent from the documentation, and how the pipeline treats them.
UNDOCUMENTED_CODES: dict[str, dict] = {
    "EDUCATION": {
        "codes": [0, 5, 6],
        "interpretation": "unknown / unlabelled education category",
        "handling": "remap to 4 ('others'): rare and semantically 'not one of the named levels'",
    },
    "MARRIAGE": {
        "codes": [0],
        "interpretation": "unknown marital status",
        "handling": "remap to 3 ('others')",
    },
    "PAY_n": {
        "codes": [-2, 0],
        "interpretation": (
            "-2 = no consumption (card not used that month); "
            "0 = revolving credit (minimum paid on time, balance carried over)"
        ),
        "handling": (
            "preserve as distinct ordinal codes; they are behaviourally different from -1 "
            "(paid in full) and carry different default risk"
        ),
    },
}

# Domain clarification of the low PAY_n codes (not part of the UCI documentation; provided
# as a domain definition for this project). The EDA checks each against same-month bills.
PAY_STATUS_DEFINITIONS: list[dict] = [
    {
        "code": -2,
        "meaning": "No consumption",
        "financial_behavior": "Card was not used that month; no bill was generated.",
    },
    {
        "code": -1,
        "meaning": "Paid in full",
        "financial_behavior": "Statement balance was paid completely on time.",
    },
    {
        "code": 0,
        "meaning": "Revolving credit",
        "financial_behavior": "Minimum payment was made on time, but a balance was carried over.",
    },
]
PAY_STATUS_DEFINITIONS_SOURCE = "project domain clarification, supplementing UCI variable_info"

BUSINESS_PROBLEM = (
    "Predict whether a credit-card holder will default on next month's payment (Oct 2005) "
    "using 6 months of repayment status, bill and payment history plus limit and demographics. "
    "The bank uses the score to flag high-risk cardholders for intervention (limit reduction, "
    "collections outreach). Trade-off: false positives add credit friction for good customers "
    "(lost revenue, churn); false negatives are missed defaults (charge-off losses). With a "
    "~22% default rate, accuracy is misleading, so the pipeline optimises F1 and reports "
    "ROC-AUC, PR-AUC, precision and recall, then picks an explicit probability cutoff."
)
