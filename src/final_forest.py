"""Refit the selected customer forest and generate its Kaggle CSV."""
from hashlib import sha256
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .history_cleaning import recover_time_intervals, repair_history

TARGET = 'repeat_purchase_90d'
CLIENT_FEATURES = ['country_or_region', 'commercial_region', 'island', 'customer_since_year']
SELECTED_FILE = 'submission_rf_clients_leaf5_t045.csv'


def transaction_columns(frame):
    return frame.columns.drop(['ID', 'customer_id', 'purchase_date', TARGET, 'random_noise']).tolist()


def add_client_features(transactions, clients):
    assert transactions['customer_id'].isin(clients['customer_id']).all()
    joined = transactions.merge(clients[['customer_id'] + CLIENT_FEATURES], on='customer_id',
                                how='left', validate='many_to_one', sort=False)
    joined.index = transactions.index
    assert joined['ID'].equals(transactions['ID'])
    joined['customer_since_year'] = joined['customer_since_year'].astype('string')
    return joined


def make_preparation():
    numbers = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler())
    categories = make_pipeline(SimpleImputer(strategy='constant', fill_value='Missing'),
                               OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    return ColumnTransformer([
        ('numbers', numbers, make_column_selector(dtype_include='number')),
        ('categories', categories, make_column_selector(dtype_exclude='number')),
    ])


def client_forest():
    return RandomForestClassifier(n_estimators=100, min_samples_leaf=5, max_features='sqrt',
                                  random_state=42)


def export_forest(forest, train, test, columns, threshold, sample, path):
    model = make_pipeline(make_preparation(), clone(forest))
    model.fit(train[columns], train[TARGET])
    probability = model.predict_proba(test[columns])[:, 1]
    assert pd.Series(probability).between(0, 1).all()
    assert test['ID'].is_unique and sample['ID'].is_unique and set(test['ID']) == set(sample['ID'])
    labels = pd.Series((probability >= threshold).astype(int), index=test['ID'])
    submission = sample[['ID']].copy()
    submission[TARGET] = submission['ID'].map(labels)
    assert submission.columns.equals(sample.columns) and submission[TARGET].isin([0, 1]).all()
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False, lineterminator='\n')
    pd.testing.assert_frame_equal(pd.read_csv(path), submission)
    return submission


def write_selected_submission(data_dir, output_dir):
    train = pd.read_csv(data_dir / 'train.csv')
    test = pd.read_csv(data_dir / 'test.csv')
    clients = pd.read_csv(data_dir / 'clientes.csv')
    sample = pd.read_csv(data_dir / 'sample_submission.csv')
    columns = transaction_columns(train)

    dates = pd.to_datetime(train['purchase_date'])
    assert dates.isna().equals(train[columns + ['random_noise']].isna().all(axis=1))
    assert train['ID'].is_unique and set(train['ID']).isdisjoint(test['ID'])
    assert dates.max() + pd.Timedelta(days=90) < pd.to_datetime(test['purchase_date']).min()

    def prepare_rows(frame):
        cleaned, _ = repair_history(frame)
        intervals = recover_time_intervals(cleaned)
        cleaned[intervals.columns] = intervals
        return add_client_features(cleaned, clients)

    return export_forest(client_forest(), prepare_rows(train.loc[dates.notna()]), prepare_rows(test),
                         columns + CLIENT_FEATURES, 0.45, sample, output_dir / SELECTED_FILE)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    output_dir = root / 'outputs' / 'notebook04'
    result = write_selected_submission(root / 'data', output_dir)
    path = output_dir / SELECTED_FILE
    print(f'{len(result)} predictions written to {path}')
    print('SHA-256:', sha256(path.read_bytes()).hexdigest())
