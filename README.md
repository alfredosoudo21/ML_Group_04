# ML Group 04 - Repeat Purchase Prediction

Course group project for predicting whether a customer makes another purchase within 90 days of a purchase occasion.

## Repository layout

```text
data/                         # Official project files
  train.csv                   # Labelled purchase occasions; contains the target
  test.csv                    # Purchase occasions to predict; no target
  clientes.csv                # Optional customer-level data, joined on customer_id
  sample_submission.csv       # Required Kaggle submission layout
notebooks/
  01_data_exploration.ipynb   # Transaction EDA and cleaning decisions
  02_client_data_integration.ipynb # Client EDA, feature decisions and safe joins
  03_temporal_baseline_experiments.ipynb # Temporal evaluation, L1 selection and predictions
  04_model_comparison.ipynb   # Final project model comparisons and random forest submissions
src/history_cleaning.py      # Shared, deterministic same-row audit and corrections
scripts/                     # Rebuild PDF, verify deliverables and create the ZIP
Homework_Group04.pdf          # Two-page intermediate report
Homework_Group04.zip          # Verified course submission package
outputs/notebook03/           # Generated metrics, feature lists and submission CSV
outputs/notebook04/           # Two reproducible random forest submission CSVs
```

`data/` contains the original CSV files from the supplied project archive. Do not overwrite these files. Any derived datasets, charts or submissions should go in a local `outputs/` folder, which is intentionally ignored by Git.

## First-time setup

Use Python 3.11 or later. NumPy 2.3.1 or later is required to include the fix for false floating-point warnings with Accelerate on newer Apple Silicon Macs.

Clone the repository and open the project folder:

```powershell
git clone https://github.com/alfredosoudo21/ML_Group_04.git
cd ML_Group_04
```

Create and activate a virtual environment, then install the shared dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m jupyter lab
```

Run each notebook from top to bottom, starting in the repository root or the `notebooks` directory. Keep `src/` beside `data/` and `notebooks/` so Notebooks 01, 03 and 04 can import the shared cleaning rules.

Notebook 04 compares eight classifiers with and without customer attributes in three expanding temporal windows. Section 4.6 examines the forest's errors and the purchase rhythm. It tests four further features individually on the forest and compares the full set and a training-only feature filter on the forest, logistic regression and gradient boosting. Section 4.7 reproduces both Kaggle submissions. The forest with customer attributes scored 0.80798 public F1 on 20 November 2026. The transaction-only forest with purchase rhythm scored 0.79487 on 21 November 2026.

The original CSVs are never overwritten. The shared rules reconstruct invalid or missing historical means from total/count when count is positive; invalidate impossible maxima and recover the exact one/two-purchase cases; and reconstruct annual spend only through exact equal-count identities. Unrecoverable invalid values remain missing for training-fold imputation. Structural missing means for customers without purchase history remain undefined. Counts, totals, minima, spread and shorter-window spend are trusted after the cross-column checks in Notebook 01.

Notebook 03 compares transaction-only and client-enriched logistic regression using the same chronological split and a 90-day gap. An inner chronological split with its own 90-day gap selects the L1 strength and whether to remove `random_noise`. Preprocessing and selection are fitted only on the corresponding training rows. Main-validation scores select the final feature set, so they are not an independent final test of the complete decision process.

The final pipeline is refitted on all eligible labelled purchases and writes six files to `outputs/notebook03/`:

- `submission_logistic_temporal.csv`: predictions for all 986 test IDs.
- `validation_metrics.csv`: scores for the two selected procedures and constant-positive reference.
- `feature_selection_search.csv`: the 16 inner candidates and their scores.
- `selected_features.csv`: the final retained columns, original variables and coefficients.
- `cleaning_audit.csv`: action counts and residual-violation checks for train and test.
- `report_summary.json`: the execution results consumed by the PDF builder.

Running the notebook does not upload a submission to Kaggle. The first two notebooks retain the shared EDA and cleaning decisions; Notebook 03 contains the executable preprocessing, feature selection and model comparison. The two-page report is `Homework_Group04.pdf`.

The intermediate delivery consists of `Homework_Group04.pdf` and `Homework_Group04.zip`. The ZIP includes Notebooks 01 to 03, original data, shared source, environment requirements, the report and the six generated files listed above.

The corrected workflow was executed on Python 3.11.17, NumPy 2.4.6, pandas 3.0.6 and scikit-learn 1.8.0. Current validation F1 is 0.7350 for transactions and 0.7338 for transactions plus clients; these are development scores, not independent final test estimates. The chosen L1 rule retains 12 encoded columns from 12 original variables after final refitting. Raw `random_noise` enters the chosen candidate but is discarded by its fitted selector. Do not reuse predictions from an earlier cleaning version.

## Rebuild and verify the delivery

After running all notebooks, install the small reporting-only dependency list and run:

```powershell
python -m pip install -r requirements-report.txt
python scripts/build_homework_report.py
python scripts/verify_homework.py
python scripts/package_homework.py
```

The report reads the latest Notebook 03 outputs instead of copying numerical results by hand. Verification checks execution, index links, fixed split/search sizes, cleaning counts, report consistency and all test IDs. Packaging includes a SHA-256 manifest and verifies the archive contents. Rebuild the PDF and ZIP after any change to notebooks or results. Uploading the ZIP/PDF to the course platform and the new prediction CSV to Kaggle remains a separate manual action.

## Collaboration rules

1. Pull the latest changes before starting: `git pull`.
2. Do not edit the raw CSVs in `data/`.
3. Make one focused change at a time, describe it in the commit message, and push it when it runs.
4. Keep modelling decisions and observations in Markdown cells beside the code that supports them.
5. Never use `test.csv` to fit imputers, encoders, scalers, feature selection or model parameters.

## What the files mean

- `train.csv`: labelled data used to train and validate models. Its target is `repeat_purchase_90d`.
- `test.csv`: unlabelled future data. It is used only for final predictions.
- `clientes.csv`: optional extra customer information; test the impact of joining it through `customer_id`.
- `sample_submission.csv`: template for the Kaggle file, with `ID` and `repeat_purchase_90d` columns.
