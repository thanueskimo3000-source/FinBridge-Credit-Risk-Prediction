# Feature choices at application submission

The assignment asks for a decision at submission. The allowlist uses borrower-reported attributes and bureau balances, counts and utilization described as application or credit-file information. Historical records have no field-level timestamps. This selection assumes the bureau values are the application snapshot. Confirm that assumption against the source system before operational use.

## Retained raw columns

| Field | Interpretation and representation |
|---|---|
| requested_amount | Requested principal in USD. Captures absolute exposure. |
| annual_income | Reported gross annual USD earnings. Supplies the denominator for affordability ratios. |
| debt_to_income_ratio | Supplied debt-obligation measure. Preserve the source scale. |
| delinquencies_24m | Count of prior delinquencies in the past two years. |
| credit_inquiries_6m | Count of recent hard bureau inquiries. |
| active_credit_lines | Number of currently open credit lines. |
| derogatory_public_records | Count of adverse public records. |
| revolving_credit_balance | Outstanding revolving balance in USD. |
| revolving_utilization_pct | Share of available revolving credit used, in percentage points. Parse 76.7% as 76.7. |
| total_credit_accounts | Total open and closed credit accounts. Overlaps with active lines but describes cumulative history. |
| employment_tenure | Reported employment duration. Preserve the labeled categories, including open-ended values. |
| housing_status | Reported housing arrangement. Encode observed labels, including the literal category NONE. |
| loan_purpose | Applicant's stated use of funds. Encode as a nominal category. |

## Engineered columns in the pipeline

| Name | Inputs and formula | Meaning and rationale | Redundancy |
|---|---|---|---|
| loan_to_income | requested_amount / annual_income | Requested principal as a fraction of one year's gross income. A loan of the same size imposes a different burden at different incomes. | Determined by principal and income. Adds their nonlinear relationship explicitly. Differs from a monthly payment share. |
| revolving_balance_to_income | revolving_credit_balance / annual_income | Existing revolving debt relative to earnings. Makes balances comparable across income levels. | Derived from balance and income. Related to utilization, whose denominator is credit limit. |
| has_prior_delinquency | 1 if delinquencies_24m > 0, otherwise 0, preserving missing values | Distinguishes any missed-payment history from a clean record. Can capture a step change even for a linear estimator. | Determined by the count. The count retains severity information. |

Ratios are unitless. Nonpositive or missing income makes a ratio undefined. Infinite numbers become missing before imputation. Missing delinquency counts yield a missing flag. The pipeline retains the row and imputes each numerical column from its training fold.

## Exclusions

| Column | Reason |
|---|---|
| application_id | Unique record key. Keep separately and in test order for submission matching. |
| applicant_ref_id | Borrower reference key with no financial measurement. |
| default_flag | Outcome used as the supervised label and for descriptive comparisons only. |
| recent_payment_amount | The PDF describes a payment received on the account. The files cannot establish that the payment predates the application. |
| interest_rate_pct | Assigned rate with no evidence of when pricing became available. |
| monthly_emi | Scheduled installment if funded. Depends on loan terms whose submission-time availability is unverified. |
| repayment_tenure | Described as the scheduled term. The files do not distinguish the applicant's requested term from the final offer. |
| risk_tier, risk_subtier | Internally assigned grades lack assignment timestamps. Subtier also contains the broad tier. |
| income_verification | Verification outcome could be determined during underwriting. The CSV does not identify when the check completed. |

## Alternatives examined

Monthly income = annual_income / 12 is an exact unit conversion. EMI-to-income = monthly_emi / monthly_income is a contractual repayment-burden measure and inherits the EMI timing limitation. Combining it with the supplied DTI would also require knowing which obligations DTI includes. Use loan_to_income to represent the requested borrowing burden. Illustrative utilization bands at 30% and 70% coarsen the continuous measurement with no validated project cutoffs. Credit-history length in years is unavailable.

The training median annual income is USD 59,000 and the maximum is USD 6,000,000. An IQR rule flags 1,411 incomes. The files provide no error labels or defensible universal cap. Keep observed values and rows. Inspect sensitivity to extreme values during model development. Standard scaling changes units and remains sensitive to extremes.
