"""Write the calendar forest submission from the repository root with python -m scripts.calendar_forest.

Same cleaning, client attributes, forest settings and threshold as src/final_forest.py, plus the purchase
month and weekday as categories and 500 trees. The selected 0.80798 file is left untouched.
"""
from hashlib import sha256
from pathlib import Path

import pandas as pd

from src.final_forest import CLIENT_FEATURES, TARGET, add_client_features, client_forest, export_forest, transaction_columns
from src.history_cleaning import recover_time_intervals, repair_history

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / 'outputs' / 'notebook04'
OUTPUT = OUTPUT_DIR / 'submission_rf_clients_calendar_500_t045.csv'
CALENDAR = ['purchase_month', 'purchase_weekday']


def prepare_rows(frame, clients):
    cleaned, _ = repair_history(frame)
    intervals = recover_time_intervals(cleaned)
    cleaned[intervals.columns] = intervals
    joined = add_client_features(cleaned, clients)
    date = pd.to_datetime(joined['purchase_date'])
    joined['purchase_month'] = date.dt.month.astype(str)
    joined['purchase_weekday'] = date.dt.dayofweek.astype(str)
    return joined


def main():
    data = ROOT / 'data'
    train = pd.read_csv(data / 'train.csv')
    test = pd.read_csv(data / 'test.csv')
    clients = pd.read_csv(data / 'clientes.csv')
    sample = pd.read_csv(data / 'sample_submission.csv')
    assert pd.to_datetime(test['purchase_date']).notna().all()
    columns = transaction_columns(train) + CLIENT_FEATURES + CALENDAR
    dated = train.loc[train['purchase_date'].notna()]
    forest = client_forest().set_params(n_estimators=500)
    submission = export_forest(forest, prepare_rows(dated, clients), prepare_rows(test, clients),
                               columns, 0.45, sample, OUTPUT)
    print(f'{len(submission)} predictions, {int(submission[TARGET].sum())} positive, written to {OUTPUT}')
    print('SHA-256:', sha256(OUTPUT.read_bytes()).hexdigest())
    new = submission.set_index('ID')[TARGET]
    for name in ['submission_rf_clients_leaf5_t045.csv', 'ML_Group_04_Version08.csv']:
        path = OUTPUT_DIR / name
        if path.exists():
            old = pd.read_csv(path).set_index('ID')[TARGET].loc[new.index]
            changed = new.ne(old)
            print(f'vs {name}: {int(changed.sum())} predictions differ '
                  f'({int((changed & new.eq(1)).sum())} new returns, {int((changed & new.eq(0)).sum())} removed)')


if __name__ == '__main__':
    main()
