# FinBridge training data findings

Missing training values: employment_tenure 829 (2.67%), revolving_utilization_pct 38 (0.12%). Preserve missing employment as a category and impute utilization using the training-fold median.

4,507 of 31,016 applications defaulted (14.53%). The majority class contains 26,509 repaid loans. Use stratified validation.

Annual income has a long right tail: median USD 59,000, maximum USD 6,000,000. The raw axis compresses typical incomes.

The income boxplot flags 1,411 observations. Keep all applications: the files provide no evidence for a universal error cutoff.

Most frequent levels by field: repayment_tenure = 36 months (75.0%), risk_tier = B (30.0%), risk_subtier = A4 (7.4%), employment_tenure = 10+ years (22.0%), housing_status = RENT (47.9%), income_verification = Not Verified (43.3%), loan_purpose = debt_consolidation (46.9%).

The strongest numerical pair is requested_amount and monthly_emi (r = 0.931). Requested principal and the scheduled installment capture overlapping loan size information.

annual_income: observed rates range from 10.6% in (90000.0, 6000000.0] (n=6,036) to 18.6% in (3999.999, 37000.0] (n=6,226).

requested_amount: observed rates range from 12.2% in (5000.0, 7750.0] (n=4,821) to 18.6% in (16000.0, 35000.0] (n=6,130).

interest_rate_pct: observed rates range from 5.4% in (5.419, 7.9] (n=6,415) to 25.9% in (15.23, 24.59] (n=6,073).

debt_to_income_ratio: observed rates range from 12.2% in (-0.001, 6.88] (n=6,205) to 16.9% in (19.68, 29.99] (n=6,197).

revolving_utilization_pct: observed rates range from 9.7% in (-0.001, 19.9] (n=6,211) to 34.2% in (missing) (n=38).

Across the five measured utilization groups, default rates rise from 9.7% to 19.8%. The missing group has only 38 applications.

delinquencies_24m: observed rates range from 14.3% in 0 (n=27,635) to 19.6% in 2 (n=540).

credit_inquiries_6m: observed rates range from 12.1% in 0 (n=15,028) to 20.2% in 3+ (n=2,826).

risk_tier: observed rates range from 6.0% in A (n=8,037) to 32.6% in F (n=800).

risk_subtier: observed rates range from 2.7% in A1 (n=912) to 41.9% in F5 (n=93).

employment_tenure: observed rates range from 12.9% in 9 years (n=978) to 22.8% in (missing) (n=829).

housing_status: observed rates range from 0.0% in NONE (n=3) to 15.9% in OTHER (n=82).

The two largest housing groups, RENT and MORTGAGE, have default rates of 15.4% and 13.6%. The three NONE applications are too few to estimate a reliable group rate.

loan_purpose: observed rates range from 10.3% in wedding (n=755) to 25.9% in small_business (n=1,434).

The assignment describes recent_payment_amount as a payment received on the account. Its availability at submission is unverified. Exclude it from prediction. The charts summarize historical associations, and estimates in small groups are unstable.
