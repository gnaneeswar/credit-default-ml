# Data Dictionary — Default of Credit Card Clients (UCI 350)

Source: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients · DOI 10.24432/C55S3H · 30000 rows

## Business problem

Predict whether a credit-card holder will default on next month's payment (Oct 2005) using 6 months of repayment status, bill and payment history plus limit and demographics. The bank uses the score to flag high-risk cardholders for intervention (limit reduction, collections outreach). Trade-off: false positives add credit friction for good customers (lost revenue, churn); false negatives are missed defaults (charge-off losses). With a ~22% default rate, accuracy is misleading, so the pipeline optimises F1 and reports ROC-AUC, PR-AUC, precision and recall, then picks an explicit probability cutoff.

## Target

`Y` → `default_payment_next_month`: default payment next month (1 = default, 0 = no default)

## Variables

| Raw | Canonical | Type | Role | Demographic |
|---|---|---|---|---|
| ID | `ID` | Integer | ID |  |
| X1 | `LIMIT_BAL` | Integer | Feature |  |
| X2 | `SEX` | Integer | Feature | Sex |
| X3 | `EDUCATION` | Integer | Feature | Education Level |
| X4 | `MARRIAGE` | Integer | Feature | Marital Status |
| X5 | `AGE` | Integer | Feature | Age |
| X6 | `PAY_1` | Integer | Feature |  |
| X7 | `PAY_2` | Integer | Feature |  |
| X8 | `PAY_3` | Integer | Feature |  |
| X9 | `PAY_4` | Integer | Feature |  |
| X10 | `PAY_5` | Integer | Feature |  |
| X11 | `PAY_6` | Integer | Feature |  |
| X12 | `BILL_AMT1` | Integer | Feature |  |
| X13 | `BILL_AMT2` | Integer | Feature |  |
| X14 | `BILL_AMT3` | Integer | Feature |  |
| X15 | `BILL_AMT4` | Integer | Feature |  |
| X16 | `BILL_AMT5` | Integer | Feature |  |
| X17 | `BILL_AMT6` | Integer | Feature |  |
| X18 | `PAY_AMT1` | Integer | Feature |  |
| X19 | `PAY_AMT2` | Integer | Feature |  |
| X20 | `PAY_AMT3` | Integer | Feature |  |
| X21 | `PAY_AMT4` | Integer | Feature |  |
| X22 | `PAY_AMT5` | Integer | Feature |  |
| X23 | `PAY_AMT6` | Integer | Feature |  |
| Y | `default_payment_next_month` | Binary | Target |  |

## Monthly timeline (position 1 = most recent)

| Month | Repayment status | Bill amount | Amount paid |
|---|---|---|---|
| 2005-09 | `PAY_1` | `BILL_AMT1` | `PAY_AMT1` |
| 2005-08 | `PAY_2` | `BILL_AMT2` | `PAY_AMT2` |
| 2005-07 | `PAY_3` | `BILL_AMT3` | `PAY_AMT3` |
| 2005-06 | `PAY_4` | `BILL_AMT4` | `PAY_AMT4` |
| 2005-05 | `PAY_5` | `BILL_AMT5` | `PAY_AMT5` |
| 2005-04 | `PAY_6` | `BILL_AMT6` | `PAY_AMT6` |

Target `default_payment_next_month` refers to **2005-10**. Renamed on load for consistent 1-6 indexing: `PAY_1` (UCI: `PAY_0`). `PAY_AMTn` is cash received *in* calendar month n; it largely settles the prior statement (`BILL_AMT(n+1)`), which is how payment-coverage ratios are built.

## Documented codes

- **SEX**: `1` = male; `2` = female
- **EDUCATION**: `1` = graduate school; `2` = university; `3` = high school; `4` = others
- **MARRIAGE**: `1` = married; `2` = single; `3` = others
- **PAY_n**: `-1` = pay duly (paid in full); `1` = payment delay for 1 month; `2` = payment delay for 2 months; `3` = payment delay for 3 months; `4` = payment delay for 4 months; `5` = payment delay for 5 months; `6` = payment delay for 6 months; `7` = payment delay for 7 months; `8` = payment delay for 8 months; `9` = payment delay for nine months and above

## Undocumented codes found in the data

| Column | Codes | Interpretation | Pipeline handling |
|---|---|---|---|
| EDUCATION | `0`, `5`, `6` | unknown / unlabelled education category | remap to 4 ('others'): rare and semantically 'not one of the named levels' |
| MARRIAGE | `0` | unknown marital status | remap to 3 ('others') |
| PAY_n | `-2`, `0` | -2 = inactive cycle (no balance owed, no new charges); 0 = revolving / minimum met (minimum paid on time, balance carried forward) | preserve as distinct ordinal codes; they are behaviourally different from -1 (paid in full) and carry different default risk |

## Low repayment-status codes — definitions (project domain definition, supplementing UCI)

| Code | Meaning | Financial behaviour |
|---|---|---|
| `-2` | Inactive cycle | No balance owed on the previous statement (n+1) and no new charges through month n's closing date (BILL_AMT_n <= 0). |
| `-1` | Paid in full | Cleared the entire previous statement (n+1) on or before its due date; any positive BILL_AMT_n reflects new purchases in cycle n. |
| `0` | Revolving / minimum met | Paid at least the minimum due on the previous statement (n+1) by its due date, carrying the remaining balance forward with interest. |
| `1..9` | Delinquent | Number of consecutive due dates without meeting the minimum payment (9 = 9 or more). |

## Billing-cycle mechanics (project domain definition, supplementing UCI)

- Statements run on rolling cycles that start on a date in one month and close on a date in the next; a statement is always named after the month in which it closes.
- PAY_n counts consecutive payment due dates where at least the minimum payment due was missed, not billing cycles elapsed since a purchase. Paying at least the minimum by the deadline keeps the account current (PAY_n = 0); the window between a statement's closing date and its due date is a normal grace period (0 cycles late).

Illustrative cycle: 16th-to-15th cycle, payment due on the 5th of the following month.

| Statement | Covers | Closes | Due | Note |
|---|---|---|---|---|
| August (n=2: BILL_AMT2, PAY_2) | Jul 16 - Aug 15 | Aug 15 | Sep 5 | A swipe on Aug 14 lands on the August statement (BILL_AMT2). |
| September (n=1: BILL_AMT1, PAY_1) | Aug 16 - Sep 15 | Sep 15 | Oct 5 | A swipe on Aug 16 lands on the September statement (BILL_AMT1). On Sep 15 the bank evaluates whether the Sep 5 due date for the August bill was met and records that status in PAY_1. |

Calendar-month cash flow:

- PAY_AMT1 sums all cash received Sep 1 - Sep 30 (calendar month).
- A payment on Sep 4 (before the Sep 15 close) counts in PAY_AMT1 and is deducted from BILL_AMT1.
- A payment on Sep 22 (after the close) counts in PAY_AMT1 but is NOT deducted from BILL_AMT1; it carries to October's statement, outside the 6-month window.

### Multi-month delinquency progression (purchase Aug 14, no payments afterwards)

| Statement date | Cycle | Due | Missed due dates | Status | Explanation |
|---|---|---|---|---|---|
| 2005-08-15 | Jul 16 - Aug 15 | 2005-09-05 | 0 | `PAY_2 = -1 or 0` | BILL_AMT2 newly generated; grace period active, not yet late. |
| 2005-09-15 | Aug 16 - Sep 15 | 2005-10-05 | 1 | `PAY_1 = 1` | Sep 5 due date missed: 1 deadline passed without the minimum. |
| 2005-10-15 | Sep 16 - Oct 15 | 2005-11-05 | 2 | `PAY = 2 (target window)` | Sep 5 and Oct 5 missed: 2 consecutive delinquent due dates. |
| 2005-11-15 | Oct 16 - Nov 15 | 2005-12-05 | 3 | `PAY = 3 (beyond dataset)` | Sep 5, Oct 5 and Nov 5 missed: 3 consecutive delinquent due dates. |

### Accounting rules (n = 1..6)

- **`BILL_AMT_n`**: Statement balance, a snapshot on month n's statement closing date: previous unpaid balance + new cycle purchases + interest/fees - payments posted on or before the closing date.
- **`PAY_AMT_n`**: Cash received within calendar month n (1st to month end); may include payments made before or after month n's closing date, so BILL_AMT_n = BILL_AMT_(n+1) - PAY_AMT_n + new spend does not hold strictly within a single month.
- **`PAY_n`**: Repayment status, a snapshot on month n's closing date of how the customer handled the previous statement (n+1) by its due date.

### Empirical check

The rules above are the project's domain definition. The EDA tests them against the data (artifacts/eda_summary.json -> domain_rule_checks). Two measured deviations: (1) a 'paid in full' status in month k is confirmed by statement k being covered by the following month's payment (PAY_AMT_(k-1) >= BILL_AMT_k) in ~98-99% of rows, versus ~75-80% for the section-4 pairing (PAY_AMT_k >= BILL_AMT_(k+1)), i.e. the data aligns status codes one month later than the rule; (2) '-2 => BILL_AMT_n <= 0' holds for only ~33-71% of -2 rows. Neither affects the engineered features, which use the status sequence and per-month amounts directly. Timing caveat: under the rule PAY_1 holds no October information; under the data's alignment it may partly reflect October payments (an ablation without PAY_1's information costs about 0.04 CV F1).

## UCI variable information (verbatim)

```
This research employed a binary variable, default payment (Yes = 1, No = 0), as the response variable. This study reviewed the literature and used the following 23 variables as explanatory variables:
X1: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.
X2: Gender (1 = male; 2 = female).
X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).
X4: Marital status (1 = married; 2 = single; 3 = others).
X5: Age (year).
X6 - X11: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: X6 = the repayment status in September, 2005; X7 = the repayment status in August, 2005; . . .;X11 = the repayment status in April, 2005. The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.
X12-X17: Amount of bill statement (NT dollar). X12 = amount of bill statement in September, 2005; X13 = amount of bill statement in August, 2005; . . .; X17 = amount of bill statement in April, 2005. 
X18-X23: Amount of previous payment (NT dollar). X18 = amount paid in September, 2005; X19 = amount paid in August, 2005; . . .;X23 = amount paid in April, 2005.

```
