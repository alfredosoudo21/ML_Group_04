"""Write the 'days until the next peak month' forests from the repository root with python -m scripts.peak_forest.

Peak months are the three months with the most purchases in the labelled data (volume only, no target).
For each purchase, days_to_peak counts the days to the first day of the next peak month after the purchase
month; the calendar of the coming 90 days is known when the purchase happens.
Writes two files, each with 500 trees, seed 42 and threshold 0.45:
  submission_rf_clients_calendar_peak_500_t045.csv  (client forest + month + weekday + days_to_peak)
  submission_rf_clients_peak_500_t045.csv           (client forest + days_to_peak)
The selected 0.80798 file is left untouched.
"""
from hashlib import sha256
from pathlib import Path

import pandas as pd

from src.final_forest import CLIENT_FEATURES, TARGET, add_client_features, client_forest, export_forest, transaction_columns
from src.history_cleaning import recover_time_intervals, repair_history

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / 'outputs' / 'notebook04'


def peak_months(purchase_dates):
    return sorted(pd.to_datetime(purchase_dates).dt.month.value_counts().nlargest(3).index.tolist())


def days_to_next_peak(purchase_dates, peaks):
    values = []
    for day in pd.to_datetime(purchase_dates):
        for step in range(1, 13):
            month = (day.month - 1 + step) % 12 + 1
            year = day.year + (day.month - 1 + step) // 12
            if month in peaks:
                values.append((pd.Timestamp(year, month, 1) - day).days)
                break
    return values


def prepare_rows(frame, clients, peaks):
    cleaned, _ = repair_history(frame)
    intervals = recover_time_intervals(cleaned)
    cleaned[intervals.columns] = intervals
    joined = add_client_features(cleaned, clients)
    date = pd.to_datetime(joined['purchase_date'])
    joined['purchase_month'] = date.dt.month.astype(str)
    joined['purchase_weekday'] = date.dt.dayofweek.astype(str)
    joined['days_to_peak'] = days_to_next_peak(date, peaks)
    return joined


def main():
    data = ROOT / 'data'
    train = pd.read_csv(data / 'train.csv')
    test = pd.read_csv(data / 'test.csv')
    clients = pd.read_csv(data / 'clientes.csv')
    sample = pd.read_csv(data / 'sample_submission.csv')
    dated = train.loc[train['purchase_date'].notna()]
    peaks = peak_months(dated['purchase_date'])
    print('Peak months by purchase volume in the labelled data:', peaks)
    train_rows, test_rows = prepare_rows(dated, clients, peaks), prepare_rows(test, clients, peaks)
    base = transaction_columns(train) + CLIENT_FEATURES
    selected = pd.read_csv(OUTPUT_DIR / 'submission_rf_clients_leaf5_t045.csv').set_index('ID')[TARGET]
    for filename, columns in [
        ('submission_rf_clients_calendar_peak_500_t045.csv', base + ['purchase_month', 'purchase_weekday', 'days_to_peak']),
        ('submission_rf_clients_peak_500_t045.csv', base + ['days_to_peak']),
    ]:
        path = OUTPUT_DIR / filename
        submission = export_forest(client_forest().set_params(n_estimators=500), train_rows, test_rows,
                                   columns, 0.45, sample, path)
        new = submission.set_index('ID')[TARGET]
        changed = new.ne(selected.loc[new.index])
        print(f'{filename}: {int(new.sum())} positive | {int(changed.sum())} differ from the 0.80798 file '
              f'({int((changed & new.eq(1)).sum())} new returns, {int((changed & new.eq(0)).sum())} removed) '
              f'| SHA-256 {sha256(path.read_bytes()).hexdigest()[:16]}')


if __name__ == '__main__':
    main()
