"""Audit and repair purchase-history fields using only the current row.

These rules do not fit statistics, inspect the target, combine customers or use
future purchases. Unknown values remain NaN for training-fold imputation.
"""
import numpy as np
import pandas as pd

TOLERANCE = 1e-6


def _different(left, right):
    """Ignore absent operands; tolerate floating-point arithmetic noise."""
    return left.notna() & right.notna() & ~np.isclose(
        left, right, rtol=0, atol=TOLERANCE
    )


def _rules(frame):
    n = frame['prior_purchase_count']
    total = frame['prior_total_spend']
    mean = total.div(n.where(n > 0))
    minimum = frame['prior_min_purchase_value']
    maximum = frame['prior_max_purchase_value']
    sd = frame['prior_std_purchase_value']

    # A conservative upper bound valid for either population or sample SD.
    # For n=1 the sole purchase is both the total and the maximum.
    max_limit = mean + sd * np.sqrt((n - 1).clip(lower=0))
    invalid_max = (
        (maximum > total + TOLERANCE)
        | ((n == 1) & _different(maximum, total))
        | ((n >= 2) & (maximum > max_limit + TOLERANCE))
        | (maximum < minimum - TOLERANCE)
        | (maximum < mean - TOLERANCE)
    )
    valid_max = maximum.mask(invalid_max)

    count365 = frame['purchase_count_365d']
    spend365 = frame['spend_365d']
    # Nested windows with equal counts contain exactly the same purchases.
    candidates = {
        'zero': pd.Series(0., index=frame.index).where(count365 == 0),
        'all_history': total.where(count365 == n),
    }
    for window in [180, 90, 30]:
        candidates[str(window)] = frame[f'spend_{window}d'].where(
            count365 == frame[f'purchase_count_{window}d']
        )
    candidates = pd.DataFrame(candidates)
    if ((candidates.max(axis=1) - candidates.min(axis=1)) > TOLERANCE).any():
        raise ValueError('Conflicting same-row sources for annual spend; inspect these rows.')
    annual_value = candidates.bfill(axis=1).iloc[:, 0]

    invalid_annual = (
        (spend365 < 0) | spend365.eq(1_000_000_000)
        | (spend365 > total + TOLERANCE)
        | (spend365 < frame['spend_180d'] - TOLERANCE)
        | (spend365 > count365 * valid_max + TOLERANCE)
        | (spend365 < count365 * minimum - TOLERANCE)
        | _different(spend365, annual_value)
    )
    flags = pd.DataFrame({
        'days_since_previous_purchase': frame['days_since_previous_purchase'] < 0,
        'prior_mean_gap_days': frame['prior_mean_gap_days'] < 0,
        'prior_mean_purchase_value': _different(frame['prior_mean_purchase_value'], mean),
        'prior_max_purchase_value': invalid_max,
        'spend_365d': invalid_annual,
    }, index=frame.index)

    max_value = pd.Series(np.nan, index=frame.index)
    max_value.loc[n == 1] = total
    max_value.loc[n == 2] = total - minimum
    return flags, mean, max_value, annual_value


def history_violations(frame):
    """Return per-column logical inconsistencies, preserving input row order."""
    return _rules(frame)[0]


def repair_history(frame):
    """Return a repaired copy and counts; never overwrite the supplied frame.

    Trusted operands are prior counts, totals, minima, standard deviations and
    the 30/90/180-day windows, checked separately in Notebook 01. Reconstruction
    restores identities; it is not a fitted estimate of an unknown value.
    """
    flags, mean, max_value, annual_value = _rules(frame)
    cleaned = frame.copy()
    missing_mean = frame['prior_mean_purchase_value'].isna() & mean.notna()
    mean_repair = flags['prior_mean_purchase_value'] | missing_mean
    cleaned.loc[mean_repair, 'prior_mean_purchase_value'] = mean

    invalid_max = flags['prior_max_purchase_value']
    max_repair = invalid_max & max_value.notna()
    cleaned.loc[invalid_max, 'prior_max_purchase_value'] = np.nan
    cleaned.loc[max_repair, 'prior_max_purchase_value'] = max_value

    invalid_annual = flags['spend_365d']
    missing_annual = frame['spend_365d'].isna()
    annual_repair = (invalid_annual | missing_annual) & annual_value.notna()
    cleaned.loc[invalid_annual, 'spend_365d'] = np.nan
    cleaned.loc[annual_repair, 'spend_365d'] = annual_value
    for column in ['days_since_previous_purchase', 'prior_mean_gap_days']:
        cleaned.loc[flags[column], column] = np.nan

    # Check the actual returned copy, not only the proposed replacement values.
    if history_violations(cleaned).any().any():
        raise AssertionError('Historical identities remain inconsistent after repair.')
    if not cleaned.loc[frame['prior_purchase_count'].eq(0), 'prior_mean_purchase_value'].isna().all():
        raise AssertionError('A mean must stay undefined for customers without history.')
    summary = flags.sum().astype(int).rename(index=lambda c: 'Invalid: ' + c)
    extra = pd.Series({
        'Distinct rows with invalid values': int(flags.any(axis=1).sum()),
        'Missing mean reconstructed': int(missing_mean.sum()),
        'Invalid maximum reconstructed': int(max_repair.sum()),
        'Invalid maximum left missing': int((invalid_max & ~max_repair).sum()),
        'Invalid annual spend reconstructed': int((invalid_annual & annual_repair).sum()),
        'Missing annual spend reconstructed': int((missing_annual & annual_repair).sum()),
        'Invalid annual spend left missing': int((invalid_annual & ~annual_repair).sum()),
        'Remaining audited violations': 0,
    }, dtype=int)
    return cleaned, pd.concat([summary, extra])
