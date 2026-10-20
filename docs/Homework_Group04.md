# Homework Group 04: Repeat purchase prediction

## Objective and data

We predict repeat_purchase_90d: whether a customer buys again within 90 days of a purchase occasion. Each row represents one occasion. train.csv has 6,534 labelled rows and 39 columns; 55.9% of targets are positive. test.csv has 986 later unlabelled occasions. clientes.csv contains 1,143 customers and 13 fields. This binary classification task can support customer follow-up.

## Exploration and manual feature decisions

Notebook 01 examines distributions, missing values, invalid history and relationships with the target. Notebook 02 audits every client field. ID and customer_id are excluded from predictors; purchase_date defines chronology. A validated left join adds country_or_region, commercial_region, island and categorical customer_since_year. Empty and nearly unique client fields are excluded; finer geography is deferred. Missing island is retained as a category because missingness may carry information.

## Pipeline schematic

1. 01 + 02 | Explore and define rules: Transaction audit; client audit; safe join; documented feature decisions.

2. 03 | Clean and split chronologically: Fixed corrections; outer assessment split; separate inner selection split.

3. 03 | Fit preparation, selection and classifier: Training medians and scaling; categorical encoding; optional L1 selection.

4. 03 | Assess, choose, refit and export: Compare feature sets; refit the chosen pipeline; predict the 986 test rows.

## Cleaning and preparation

Both experiments exclude the same 65 almost empty purchase rows without valid dates, leaving 6,469 occasions. Audited negative historical intervals, suspected negative monetary errors and exact 1,000,000,000 sentinels become missing. Treating negative monetary history as invalid is an explicit assumption; ordinary high values remain. Numeric features use median imputation, missing indicators and standardisation. Categorical features use a Missing category and one-hot encoding, with unseen categories ignored. Learned transformations are fitted only on the relevant training set.

## Temporal assessment

The outer training set contains 4,424 occasions from 11 July 2015 to 10 April 2022. A 90-day gap, defined from 14 April to 12 July 2022, excludes 103 occasions from assessment. Validation contains 1,942 occasions from 13 July 2022 to 15 February 2025. This gap lets training target windows finish before validation starts. Both feature sets use the same rows. Shared customers are allowed to assess future occasions by known customers. Gap rows return when fitting the final model.

## Feature selection within training data

L1 logistic regression with SelectFromModel complements the manual exclusions. L1 can set coefficients to zero, allowing a smaller predictor set without fixing its size in advance. Selection follows imputation, encoding and scaling. For each feature set, eight candidates compare no selection or L1 C in {0.01, 0.1, 1}, each with and without random_noise. An inner temporal split uses only the outer training data: 2,971 training occasions, a 90-day gap of 111, and 1,342 validation occasions starting 4 July 2020. The best inner F1 determines the configuration.

## Chosen configuration and classifier

Both feature sets choose L1 C=0.01 without random_noise. The best candidates tie with and without noise, so the stated preference for fewer inputs removes it; this is not evidence that noise alone caused the improvement. The selector (liblinear solver; coefficient threshold 1e-8) and preprocessing are refitted on the outer training set. The classifier is L2 logistic regression, C=1, lbfgs, maximum 10,000 iterations. The probability threshold stays fixed at 0.5.

## Outer validation results

| Feature set | Selection | F1 | Precision | Recall | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Always positive | None | 0.7211 | 0.5639 | 1.0000 | 0.5000 |
| Transactions | None | 0.7267 | 0.7987 | 0.6667 | 0.7764 |
| Transactions + clients | None | 0.7056 | 0.7959 | 0.6338 | 0.7722 |
| Transactions | L1 | 0.7357 | 0.7627 | 0.7105 | 0.7811 |
| Transactions + clients | L1 | 0.7330 | 0.7777 | 0.6932 | 0.7815 |

We prioritise F1 to balance precision and recall for repeat purchases. Scores use the same outer validation rows. Selection improves F1 by 0.90 percentage points for transactions and 2.73 for transactions plus clients. The chosen transaction model exceeds the constant-positive reference by 1.46 points. Client enrichment does not improve F1 here; the difference after selection is only 0.27 points, so it does not establish a general disadvantage of client information.

## Final feature set and predictions

After choosing transactions without clients, the complete pipeline is refitted on all 6,469 eligible labelled occasions. Its selector retains 12 encoded columns from 11 original variables: Total Imp_, Total Desc_ Linha, prior_total_spend, prior_std_purchase_value, prior_mean_gap_days, prior_std_gap_days, purchase_count_90d, spend_180d, purchase_count_365d, salesperson and commercial_zone. Two retained columns are missing indicators and two are categorical levels. Refitting on more data can change the selected columns. The pipeline exports 986 binary predictions, checking IDs, order and CSV format. Test spans 24 May 2025 to 6 October 2026, after all training label windows finish.

## Scope of the conclusions

EDA used all labelled data, and outer scores informed the final feature-set choice. These scores describe development performance, not an independent final estimate. One holdout cannot establish stability over time. We assume client attributes were available at purchase; consistent entry years do not prove historical availability.
