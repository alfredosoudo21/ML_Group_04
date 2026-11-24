"""Write the optimized logistic regression submission from the repository root with python -m scripts.logistic_max.

Configuration chosen on the three early validation periods only: client inputs + purchase month and weekday
+ days to the next peak month; numeric inputs median-imputed with missing indicators, signed log1p and scaled;
eight quantile bins of the frequency and timing inputs; rare categories grouped; L2 penalty with C=0.03;
threshold 0.44.
"""
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer, OneHotEncoder, StandardScaler

from scripts.peak_forest import peak_months, prepare_rows
from src.final_forest import CLIENT_FEATURES, TARGET, transaction_columns

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs' / 'notebook04' / 'submission_logistic_max_t044.csv'
BINNED = ['purchase_count_30d', 'purchase_count_90d', 'purchase_count_180d', 'purchase_count_365d',
          'prior_purchase_count', 'days_since_previous_purchase', 'prior_mean_gap_days', 'prior_std_gap_days',
          'customer_tenure_days', 'spend_365d', 'Total a Pagar', 'days_to_peak']


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def logistic_model():
    numbers = make_pipeline(SimpleImputer(strategy='median', add_indicator=True),
                            FunctionTransformer(signed_log), StandardScaler())
    categories = make_pipeline(SimpleImputer(strategy='constant', fill_value='Missing'),
                               OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=20, sparse_output=False))
    bins = make_pipeline(SimpleImputer(strategy='median'),
                         KBinsDiscretizer(n_bins=8, encode='onehot-dense', strategy='quantile'))
    preparation = ColumnTransformer([
        ('numbers', numbers, make_column_selector(dtype_include='number')),
        ('categories', categories, make_column_selector(dtype_exclude='number')),
        ('bins', bins, BINNED),
    ])
    return make_pipeline(preparation, LogisticRegression(C=0.03, max_iter=5000))


def main():
    data = ROOT / 'data'
    train = pd.read_csv(data / 'train.csv')
    test = pd.read_csv(data / 'test.csv')
    clients = pd.read_csv(data / 'clientes.csv')
    sample = pd.read_csv(data / 'sample_submission.csv')
    dated = train.loc[train['purchase_date'].notna()]
    peaks = peak_months(dated['purchase_date'])
    train_rows, test_rows = prepare_rows(dated, clients, peaks), prepare_rows(test, clients, peaks)
    columns = transaction_columns(train) + CLIENT_FEATURES + ['purchase_month', 'purchase_weekday', 'days_to_peak']
    model = logistic_model().fit(train_rows[columns], train_rows[TARGET])
    probability = model.predict_proba(test_rows[columns])[:, 1]
    labels = pd.Series((probability >= 0.44).astype(int), index=test_rows['ID'])
    submission = sample[['ID']].copy()
    submission[TARGET] = submission['ID'].map(labels)
    assert submission[TARGET].isin([0, 1]).all() and len(submission) == len(sample)
    submission.to_csv(OUTPUT, index=False, lineterminator='\n')
    pd.testing.assert_frame_equal(pd.read_csv(OUTPUT), submission)
    print(f'{len(submission)} predictions, {int(submission[TARGET].sum())} positive, written to {OUTPUT}')
    print('SHA-256:', sha256(OUTPUT.read_bytes()).hexdigest()[:16])
    new = submission.set_index('ID')[TARGET]
    for name in ['submission_rf_clients_leaf5_t045.csv', 'submission_rf_clients_calendar_peak_500_t045.csv']:
        old = pd.read_csv(OUTPUT.parent / name).set_index('ID')[TARGET].loc[new.index]
        print(f'vs {name}: {int(new.ne(old).sum())} predictions differ')


if __name__ == '__main__':
    main()
