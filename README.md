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
  01_data_exploration.ipynb   # Shared starting point: data audit and EDA
```

`data/` contains the original CSV files from the supplied project archive. Do not overwrite these files. Any derived datasets, charts or submissions should go in a local `outputs/` folder, which is intentionally ignored by Git.

## First-time setup

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

Open `notebooks/01_data_exploration.ipynb` from JupyterLab and run it from top to bottom. The notebook is written with paths relative to the repository root.

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
