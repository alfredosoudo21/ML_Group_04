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
Midterm_Group04.pdf           # Two-page intermediate report
outputs/notebook03/           # Generated metrics, feature lists and submission CSV
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

Open the notebooks in numerical order and run each from top to bottom. Both the repository root and the `notebooks` directory are supported as working directories. The first two notebooks include a table of contents; there are no separate copies for the index.

Notebook 03 compares transaction-only and client-enriched logistic regression using the same chronological split and a 90-day gap. An inner chronological split with its own 90-day gap selects the L1 strength and whether to remove `random_noise`. Preprocessing and selection are fitted only on the corresponding training rows. Main-validation scores select the final feature set, so they are not an independent final test of the complete decision process.

The final pipeline is refitted on all eligible labelled purchases and writes four files to `outputs/notebook03/`:

- `submission_logistic_temporal.csv`: predictions for all 986 test IDs.
- `validation_metrics.csv`: scores for the two selected procedures and constant-positive reference.
- `feature_selection_search.csv`: the 16 inner candidates and their scores.
- `selected_features.csv`: the final retained columns, original variables and coefficients.

Running the notebook does not upload a submission to Kaggle. The first two notebooks retain the shared EDA and cleaning decisions; Notebook 03 contains the executable preprocessing, feature selection and model comparison. The two-page report is `Midterm_Group04.pdf`.

For the course submission, use the required `Homework_Group04` naming convention for the PDF and notebook ZIP.

The complete modelling notebook was also verified with the existing Anaconda environment (Python 3.14.6, NumPy 2.4.6, pandas 3.0.3 and scikit-learn 1.9.0). It produced the same selected configuration and test predictions without numerical warnings. Older environments should be updated using `requirements.txt`; do not suppress numerical warnings to obtain a clean run.

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
