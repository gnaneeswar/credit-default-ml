"""Canonical schema for UCI dataset 350 (Default of Credit Card Clients, Taiwan 2005).

The UCI variables table names features X1..X23 and the target Y; their descriptions carry
the conventional domain names. The raw source names September's repayment status ``PAY_0``
(and has no ``PAY_1``), while bills and payments use ``BILL_AMT1`` / ``PAY_AMT1``. On load,
X6 is mapped straight to ``PAY_1`` so every monthly family shares 1..6 indexing
(1 = Sep 2005 ... 6 = Apr 2005). ``UCI_ORIGINAL_NAMES`` records the source name.
"""

from __future__ import annotations

UCI_ID = 350
ID_COL = "ID"
RAW_TARGET = "Y"
TARGET = "default_payment_next_month"

PAY_STATUS_COLS = [f"PAY_{i}" for i in range(1, 7)]  # PAY_1 = Sep 2005 (UCI: PAY_0)
BILL_AMT_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
DEMOGRAPHIC_COLS = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]

FEATURES = ["LIMIT_BAL", *DEMOGRAPHIC_COLS, *PAY_STATUS_COLS, *BILL_AMT_COLS, *PAY_AMT_COLS]
CATEGORICAL_CODE_COLS = ["SEX", "EDUCATION", "MARRIAGE", *PAY_STATUS_COLS]

RAW_TO_CANONICAL: dict[str, str] = {f"X{i}": name for i, name in enumerate(FEATURES, start=1)}
RAW_TO_CANONICAL[RAW_TARGET] = TARGET

# Canonical name -> name used in the UCI source / literature, where they differ.
UCI_ORIGINAL_NAMES: dict[str, str] = {"PAY_1": "PAY_0"}

# Position 1 = most recent month. PAY_1 / BILL_AMT1 / PAY_AMT1 all refer to September 2005.
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
            "-2 = inactive cycle (no balance owed, no new charges); "
            "0 = revolving / minimum met (minimum paid on time, balance carried forward)"
        ),
        "handling": (
            "preserve as distinct ordinal codes; they are behaviourally different from -1 "
            "(paid in full) and carry different default risk"
        ),
    },
}

# --------------------------------------------------------------------------------------
# Billing-cycle mechanics and accounting rules (project domain definition, supplementing
# the UCI documentation). Kept verbatim; `eda.domain_rule_checks` tests them against the data.
# --------------------------------------------------------------------------------------
DOMAIN_RULES_SOURCE = "project domain definition, supplementing UCI"

TARGET_MONTH = "2005-10"  # default_payment_next_month refers to October 2005

BILLING_CYCLE: dict = {
    "rules": [
        "Statements run on rolling cycles that start on a date in one month and close on a "
        "date in the next; a statement is always named after the month in which it closes.",
        "PAY_n counts consecutive payment due dates where at least the minimum payment due was "
        "missed, not billing cycles elapsed since a purchase. Paying at least the minimum by "
        "the deadline keeps the account current (PAY_n = 0); the window between a statement's "
        "closing date and its due date is a normal grace period (0 cycles late).",
    ],
    "example_cycle": "16th-to-15th cycle, payment due on the 5th of the following month",
    "example": [
        {
            "statement": "August (n=2: BILL_AMT2, PAY_2)",
            "covers": "Jul 16 - Aug 15",
            "closes": "Aug 15",
            "due": "Sep 5",
            "note": "A swipe on Aug 14 lands on the August statement (BILL_AMT2).",
        },
        {
            "statement": "September (n=1: BILL_AMT1, PAY_1)",
            "covers": "Aug 16 - Sep 15",
            "closes": "Sep 15",
            "due": "Oct 5",
            "note": (
                "A swipe on Aug 16 lands on the September statement (BILL_AMT1). On Sep 15 the "
                "bank evaluates whether the Sep 5 due date for the August bill was met and "
                "records that status in PAY_1."
            ),
        },
    ],
    "cash_flow": [
        "PAY_AMT1 sums all cash received Sep 1 - Sep 30 (calendar month).",
        "A payment on Sep 4 (before the Sep 15 close) counts in PAY_AMT1 and is deducted from "
        "BILL_AMT1.",
        "A payment on Sep 22 (after the close) counts in PAY_AMT1 but is NOT deducted from "
        "BILL_AMT1; it carries to October's statement, outside the 6-month window.",
    ],
}

DELINQUENCY_PROGRESSION: list[dict] = [
    # Purchase on Aug 14 (Jul 16 - Aug 15 cycle), zero payments thereafter.
    {
        "statement_date": "2005-08-15",
        "cycle": "Jul 16 - Aug 15",
        "due": "2005-09-05",
        "missed_due_dates": 0,
        "status": "PAY_2 = -1 or 0",
        "explanation": "BILL_AMT2 newly generated; grace period active, not yet late.",
    },
    {
        "statement_date": "2005-09-15",
        "cycle": "Aug 16 - Sep 15",
        "due": "2005-10-05",
        "missed_due_dates": 1,
        "status": "PAY_1 = 1",
        "explanation": "Sep 5 due date missed: 1 deadline passed without the minimum.",
    },
    {
        "statement_date": "2005-10-15",
        "cycle": "Sep 16 - Oct 15",
        "due": "2005-11-05",
        "missed_due_dates": 2,
        "status": "PAY = 2 (target window)",
        "explanation": "Sep 5 and Oct 5 missed: 2 consecutive delinquent due dates.",
    },
    {
        "statement_date": "2005-11-15",
        "cycle": "Oct 16 - Nov 15",
        "due": "2005-12-05",
        "missed_due_dates": 3,
        "status": "PAY = 3 (beyond dataset)",
        "explanation": "Sep 5, Oct 5 and Nov 5 missed: 3 consecutive delinquent due dates.",
    },
]

ACCOUNTING_RULES: dict[str, str] = {
    "BILL_AMT_n": (
        "Statement balance, a snapshot on month n's statement closing date: previous unpaid "
        "balance + new cycle purchases + interest/fees - payments posted on or before the "
        "closing date."
    ),
    "PAY_AMT_n": (
        "Cash received within calendar month n (1st to month end); may include payments made "
        "before or after month n's closing date, so BILL_AMT_n = BILL_AMT_(n+1) - PAY_AMT_n + "
        "new spend does not hold strictly within a single month."
    ),
    "PAY_n": (
        "Repayment status, a snapshot on month n's closing date of how the customer handled the "
        "previous statement (n+1) by its due date."
    ),
}

# Code-level meanings of PAY_n (the brief's section 4 wording).
PAY_STATUS_DEFINITIONS: list[dict] = [
    {
        "code": -2,
        "meaning": "Inactive cycle",
        "financial_behavior": (
            "No balance owed on the previous statement (n+1) and no new charges through month "
            "n's closing date (BILL_AMT_n <= 0)."
        ),
    },
    {
        "code": -1,
        "meaning": "Paid in full",
        "financial_behavior": (
            "Cleared the entire previous statement (n+1) on or before its due date; any positive "
            "BILL_AMT_n reflects new purchases in cycle n."
        ),
    },
    {
        "code": 0,
        "meaning": "Revolving / minimum met",
        "financial_behavior": (
            "Paid at least the minimum due on the previous statement (n+1) by its due date, "
            "carrying the remaining balance forward with interest."
        ),
    },
    {
        "code": "1..9",
        "meaning": "Delinquent",
        "financial_behavior": (
            "Number of consecutive due dates without meeting the minimum payment (9 = 9 or more)."
        ),
    },
]
PAY_STATUS_DEFINITIONS_SOURCE = DOMAIN_RULES_SOURCE

EMPIRICAL_CHECK_NOTE = (
    "The rules above are the project's domain definition. The EDA tests them against the data "
    "(artifacts/eda_summary.json -> domain_rule_checks). Two measured deviations: (1) a "
    "'paid in full' status in month k is confirmed by statement k being covered by the "
    "following month's payment (PAY_AMT_(k-1) >= BILL_AMT_k) in ~98-99% of rows, versus "
    "~73-80% for the section-4 pairing (PAY_AMT_k >= BILL_AMT_(k+1)), i.e. the data aligns "
    "status codes one month later than the rule; (2) '-2 => BILL_AMT_n <= 0' holds for only "
    "~33-71% of -2 rows. Neither affects the engineered features, which use the status "
    "sequence and per-month amounts directly. Timing caveat: under the rule PAY_1 holds no "
    "October information; under the data's alignment it may partly reflect October payments "
    "(an ablation without PAY_1's information costs about 0.04 CV F1)."
)


BUSINESS_PROBLEM = (
    "Predict whether a credit-card holder will default on next month's payment (Oct 2005) "
    "using 6 months of repayment status, bill and payment history plus limit and demographics. "
    "The bank uses the score to flag high-risk cardholders for intervention (limit reduction, "
    "collections outreach). Trade-off: false positives add credit friction for good customers "
    "(lost revenue, churn); false negatives are missed defaults (charge-off losses). With a "
    "~22% default rate, accuracy is misleading, so the pipeline optimises F1 and reports "
    "ROC-AUC, PR-AUC, precision and recall, then picks an explicit probability cutoff."
)
