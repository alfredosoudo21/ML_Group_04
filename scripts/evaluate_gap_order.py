"""Run the gap-order experiment from the repository root with python -m scripts.evaluate_gap_order."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.final_forest import (
    CLIENT_FEATURES, TARGET, add_client_features, client_forest, make_preparation, transaction_columns,
)
from src.history_cleaning import recover_time_intervals, repair_history


DATA = Path(__file__).resolve().parents[1] / 'data'
SEEDS = [0, 1, 2, 42]
UNKNOWN = 'insufficient history'


def completed_gap_order(purchases):
    """Pattern as of the end of each purchase occasion, using only its last three gaps."""
    history = purchases.sort_values(['customer_id', 'purchase_date'])
    assert not history.duplicated(['customer_id', 'purchase_date']).any()
    assert history['customer_id'].notna().all()
    customer = history['customer_id']
    gaps = history['purchase_date'].groupby(customer).diff().dt.days
    consecutive = history['prior_purchase_count'].groupby(customer).diff().eq(1)
    last = gaps.where(consecutive & gaps.gt(0))
    middle = last.groupby(customer).shift(1)
    first = last.groupby(customer).shift(2)
    order = pd.Series(UNKNOWN, index=history.index)
    order.loc[first.notna() & middle.notna() & last.notna()] = 'mixed or tied'
    order.loc[(first > middle) & (middle > last)] = 'shortening'
    order.loc[(first < middle) & (middle < last)] = 'lengthening'
    return order


def frozen_order(training, completed, customers):
    last_rows = training.sort_values('purchase_date').drop_duplicates('customer_id', keep='last')
    by_customer = pd.Series(completed.loc[last_rows.index].to_numpy(), index=last_rows['customer_id'])
    return customers.map(by_customer).fillna(UNKNOWN)


def temporal_windows(date):
    gap = pd.Timedelta(days=90)
    start = date.sort_values().iloc[int(len(date) * 0.70)]
    early = [pd.Timestamp('2019-07-01'), pd.Timestamp('2020-07-01'), pd.Timestamp('2021-07-01'), start - gap]
    recent = [start + pd.DateOffset(years=i) for i in range(3)] + [date.max() + pd.Timedelta(days=1)]
    for boundaries in [early, recent]:
        for left, right in zip(boundaries, boundaries[1:]):
            training = date < left - gap
            validation = (date >= left) & (date < right)
            assert date[training].max() + gap < date[validation].min()
            yield training, validation


def paired_bootstrap(predictions, reference):
    """Resample whole customers jointly for every period, seed and model."""
    customers = pd.Index(sorted(predictions['customer_id'].unique()))
    rng = np.random.default_rng(0)
    weights = rng.multinomial(len(customers), np.full(len(customers), 1 / len(customers)), size=2000)
    gains = []
    for _, rows in predictions.groupby(['period', 'seed']):
        index = customers.get_indexer(rows['customer_id'])
        scores = []
        for model in [reference, 'calendar_order']:
            true_positives = np.bincount(index, weights=rows[model] & rows[TARGET], minlength=len(customers))
            denominator = np.bincount(index, weights=rows[model].astype(int) + rows[TARGET], minlength=len(customers))
            scores.append(2 * (weights @ true_positives) / (weights @ denominator))
        gains.append(scores[1] - scores[0])
    return np.mean(gains, axis=0)


def run():
    raw = pd.read_csv(DATA / 'train.csv')
    base = transaction_columns(raw) + CLIENT_FEATURES
    calendar = base + ['purchase_month', 'purchase_weekday']
    data, _ = repair_history(raw.loc[raw['purchase_date'].notna()].copy())
    intervals = recover_time_intervals(data)
    data[intervals.columns] = intervals
    data = add_client_features(data, pd.read_csv(DATA / 'clientes.csv'))
    date = data['purchase_date'] = pd.to_datetime(data['purchase_date'])
    data['purchase_month'] = date.dt.month.astype(str)
    data['purchase_weekday'] = date.dt.dayofweek.astype(str)
    completed = completed_gap_order(data)
    # The state after the previous observed purchase is available strictly before this purchase.
    data['gap_order'] = completed.groupby(data.loc[completed.index, 'customer_id']).shift(fill_value=UNKNOWN)
    test_customers = pd.read_csv(DATA / 'test.csv', usecols=['customer_id'])['customer_id']
    print('Test coverage:', frozen_order(data, completed, test_customers).ne(UNKNOWN).sum(), '/', len(test_customers))

    models = {'base': base, 'calendar': calendar, 'calendar_order': calendar + ['gap_order']}
    results, decisions = [], []
    for period, (train_mask, valid_mask) in enumerate(temporal_windows(date), start=1):
        training = data.loc[train_mask]
        validation = data.loc[valid_mask].copy()
        validation['gap_order'] = frozen_order(training, completed, validation['customer_id'])
        rows = validation[['customer_id', TARGET]].astype({TARGET: bool}).copy()
        predictions = {seed: rows.assign(period=period, seed=seed) for seed in SEEDS}
        for name, columns in models.items():
            preparation = make_preparation()
            x_train = preparation.fit_transform(training[columns])
            x_valid = preparation.transform(validation[columns])
            for seed in SEEDS:
                forest = client_forest().set_params(n_estimators=500, random_state=seed, n_jobs=-1)
                forest.fit(x_train, training[TARGET])
                predictions[seed][name] = forest.predict_proba(x_valid)[:, 1] >= 0.45
            if name == 'base':
                submitted = client_forest().set_params(n_jobs=-1)
                submitted.fit(x_train, training[TARGET])
                predictions[42]['submitted'] = submitted.predict_proba(x_valid)[:, 1] >= 0.45
        for seed, frame in predictions.items():
            scores = {m: f1_score(frame[TARGET], frame[m]) for m in models}
            if seed == 42:
                scores['submitted'] = f1_score(frame[TARGET], frame['submitted'])
            results.append(dict(period=period, seed=seed, **scores))
            decisions.append(frame)
        print('Period', period, 'complete. Validation coverage:', validation['gap_order'].ne(UNKNOWN).sum(),
              '/', len(validation), flush=True)

    results = pd.DataFrame(results)
    decisions = pd.concat(decisions, ignore_index=True)
    means = results.groupby('period')[list(models)].mean()
    print('Mean F1 per period over four seeds:\n', means.round(5))
    positive_blocks = True
    for reference in ['calendar', 'base']:
        delta = means['calendar_order'] - means[reference]
        early, recent = delta.iloc[:3].mean(), delta.iloc[3:].mean()
        boot = paired_bootstrap(decisions, reference)
        print(f'Gain over {reference}: early {early:+.6f}, recent {recent:+.6f}, all {delta.mean():+.6f}.')
        print('95% bootstrap interval:', np.quantile(boot, [.025, .975]), 'fraction positive:', (boot > 0).mean())
        positive_blocks = positive_blocks and early > 0 and recent > 0
        if reference == 'base':
            promote = positive_blocks and delta.mean() >= .005 and (boot > 0).mean() >= .90
    print('Meets agreed promotion rule:', bool(promote))
    exact = results.loc[results['seed'].eq(42)].set_index('period')[['submitted', 'calendar_order']]
    print('Submission configurations, seed 42:\n', exact.round(5))
    for name in models:
        fp = (decisions[name] & ~decisions[TARGET]).sum() / len(SEEDS)
        fn = (~decisions[name] & decisions[TARGET]).sum() / len(SEEDS)
        print(f'{name}: {fp:.2f} false positives, {fn:.2f} false negatives per seed over the six periods.')


if __name__ == '__main__':
    run()
