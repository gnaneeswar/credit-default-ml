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
| X6 | `PAY_0` | Integer | Feature |  |
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
| 2005-09 | `PAY_0` | `BILL_AMT1` | `PAY_AMT1` |
| 2005-08 | `PAY_2` | `BILL_AMT2` | `PAY_AMT2` |
| 2005-07 | `PAY_3` | `BILL_AMT3` | `PAY_AMT3` |
| 2005-06 | `PAY_4` | `BILL_AMT4` | `PAY_AMT4` |
| 2005-05 | `PAY_5` | `BILL_AMT5` | `PAY_AMT5` |
| 2005-04 | `PAY_6` | `BILL_AMT6` | `PAY_AMT6` |

Note: `PAY_AMTn` is the amount paid *in* month n; it is conventionally read as settling the prior statement (`BILL_AMT(n+1)`), which is how payment-coverage ratios are built. There is no `PAY_1` in the original data.

## Documented codes

- **SEX**: `1` = male; `2` = female
- **EDUCATION**: `1` = graduate school; `2` = university; `3` = high school; `4` = others
- **MARRIAGE**: `1` = married; `2` = single; `3` = others
- **PAY_n**: `-1` = pay duly; `1` = payment delay for 1 month; `2` = payment delay for 2 months; `3` = payment delay for 3 months; `4` = payment delay for 4 months; `5` = payment delay for 5 months; `6` = payment delay for 6 months; `7` = payment delay for 7 months; `8` = payment delay for 8 months; `9` = payment delay for nine months and above

## Undocumented codes found in the data

| Column | Codes | Interpretation | Pipeline handling |
|---|---|---|---|
| EDUCATION | `0`, `5`, `6` | unknown / unlabelled education category | remap to 4 ('others'): rare and semantically 'not one of the named levels' |
| MARRIAGE | `0` | unknown marital status | remap to 3 ('others') |
| PAY_n | `-2`, `0` | -2 = no consumption / inactive (zero bill); 0 = revolving credit — minimum payment made, balance carried | preserve as distinct ordinal codes; they are behaviourally different from -1 (paid in full) and carry different default risk |

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
