# ML Group 04 - Repeat Purchase Prediction

Course group project for predicting whether a customer makes another purchase within 90 days of a purchase occasion.

## Final project

Start with [Notebook 01](notebooks/01_data_exploration.ipynb) for the abstract, the business question and the reading order. Section II continues in [Notebook 02](notebooks/02_client_data_integration.ipynb). Section III starts with the logistic baseline in [Notebook 03](notebooks/03_temporal_baseline_experiments.ipynb). [Notebook 04](notebooks/04_model_comparison.ipynb) completes model assessment and feature selection, compares errors by purchase frequency in Section IV and generates the selected CSV in Section V. The group will supply its contribution breakdown before hand-in.

The selected submission uses the random forest with four client attributes, 100 trees, a minimum of five purchases per leaf and threshold 0.45. Its public F1 was **0.80798**. From the repository root, create a fresh environment with Python **3.14.6**, the version used to generate the submitted file. With that environment active, run:

```sh
python -m pip install -r requirements-final.txt
python -m src.final_forest
```

The command reads the four official files, refits the model and writes `outputs/notebook04/submission_rf_clients_leaf5_t045.csv`. The submitted version contains 986 predictions in sample order, 726 of them positive. Its SHA-256 is `419f972f04b97a4d135cf6fd179a4a0ada29b8a984c2cf9410442a61486eae93`. To reproduce the analysis as well, start `python -m jupyter lab` in the same environment and run the notebooks in order.

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
src/final_forest.py           # Train the selected forest and write its submission
requirements-final.txt       # Versions used for the final notebooks and prediction command
scripts/                     # Rebuild PDF, verify deliverables and create the ZIP
Homework_Group04.pdf          # Two-page intermediate report
Homework_Group04.zip          # Verified course submission package
outputs/notebook03/           # Generated metrics, feature lists and submission CSV
outputs/notebook04/           # Four random forest submission CSVs
```

`data/` contains the original CSV files from the supplied project archive. Do not overwrite these files. Any derived datasets, charts or submissions should go in a local `outputs/` folder, which is intentionally ignored by Git.

## Intermediate homework setup

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

Notebook 04 compares eight classifiers on three expanding temporal windows. It tests purchase rhythm, four further relationships and groups of original inputs. Section IV compares the selected forest with simple frequency rules. It also selects thresholds and training lengths on earlier purchases and checks them with three later refits. Lowering the threshold for purchases with one to five prior-year purchases raised later mean F1 from 0.7623 to 0.7651 across four seeds, but scored slightly lower on the earlier periods and with submission seed 42. The four-year training window lost in all three later periods. The original forest remains selected.

Section V reproduces four public Kaggle submissions. The client forest scored 0.80798, adding purchase rhythm scored 0.80299, the transaction-only rhythm forest with different settings scored 0.79487 and removing the five count inputs from the client forest scored 0.80000.

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

## Rebuild and verify the intermediate delivery

After running all notebooks, install the small reporting-only dependency list and run:

```powershell
python -m pip install -r requirements-report.txt
python scripts/build_homework_report.py
python scripts/verify_homework.py
python scripts/package_homework.py
```

The intermediate report reads the Notebook 03 outputs. Verification checks execution, index links, fixed split/search sizes, cleaning counts, report consistency and all test IDs. The packaging script rebuilds the intermediate archive with a SHA-256 manifest. The final project uses all four notebooks and the submission described above.

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
