"""Check the executed notebooks, report, metrics and prediction delivery.

Run with Python and the optional requirements-report.txt installed. Uses only
the standard library plus pypdf; it does not refit or modify anything.
"""
import csv
import json
import math
import re
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'notebook03'


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def verify_project():
    total_code = 0
    for path in sorted((ROOT / 'notebooks').glob('*.ipynb')):
        nb = json.loads(path.read_text(encoding='utf-8'))
        cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
        assert cells and all(c.get('execution_count') is not None for c in cells), path
        assert not [o for c in cells for o in c.get('outputs', []) if o['output_type'] == 'error'], path
        warnings = '\n'.join(''.join(o.get('text', [])) for c in cells
                             for o in c.get('outputs', []) if o['output_type'] == 'stream')
        assert not any(w in warnings for w in ['ConvergenceWarning', 'RuntimeWarning', 'overflow encountered']), path
        markdown = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'markdown')
        anchors = set(re.findall(r'<a\s+id="([^"]+)"', markdown))
        links = set(re.findall(r'href="#([^"]+)"', markdown))
        assert links <= anchors, (path.name, 'Broken internal links', links - anchors)
        total_code += len(cells)
        print(path.name + ':', len(cells), 'executed code cells, no errors; index links valid.')

    fields, submission = read_csv(OUT / 'submission_logistic_temporal.csv')
    _, test = read_csv(ROOT / 'data' / 'test.csv')
    sample_fields, sample = read_csv(ROOT / 'data' / 'sample_submission.csv')
    assert fields == sample_fields == ['ID', 'repeat_purchase_90d']
    assert len(submission) == len(test) == 986
    assert [row['ID'] for row in submission] == [row['ID'] for row in test] == [row['ID'] for row in sample]
    assert len({row['ID'] for row in submission}) == len(submission)
    assert all(row['repeat_purchase_90d'] in ['0', '1'] for row in submission)

    run = json.loads((OUT / 'report_summary.json').read_text(encoding='utf-8'))
    assert [run['split'][p] for p in ['train', 'gap', 'validation']] == [4424, 103, 1942]
    assert [run['inner'][p] for p in ['train', 'gap', 'validation']] == [2971, 111, 1342]
    assert run['threshold'] == 0.5
    assert sum(int(row['repeat_purchase_90d']) for row in submission) == run['test_positive_predictions']

    _, metrics = read_csv(OUT / 'validation_metrics.csv')
    assert len(metrics) == len(run['selected_metrics']) == 3
    for a, b in zip(metrics, run['selected_metrics']):
        assert a['Model'] == b['Model']
        for key in ['F1', 'Precision', 'Recall', 'Accuracy', 'ROC-AUC']:
            assert math.isclose(float(a[key]), b[key], rel_tol=0, abs_tol=1e-12)
    _, features = read_csv(OUT / 'selected_features.csv')
    assert len(features) == run['final_encoded_retained']
    assert len({row['Original variable'] for row in features}) == run['final_raw_retained']
    assert {row['Original variable'] for row in features} == set(run['final_raw_names'])
    assert all(math.isfinite(float(row['Coefficient'])) for row in features)
    _, candidates = read_csv(OUT / 'feature_selection_search.csv')
    assert len(candidates) == 16
    for name, chosen in run['settings_by_feature_set'].items():
        arm = [row for row in candidates if row['Feature set'] == name]
        assert len(arm) == 8
        assert {(row['Selection'], row['Keep noise']) for row in arm} == {
            (selection, noise) for selection in ['No selection', 'L1 C=0.01', 'L1 C=0.1', 'L1 C=1.0']
            for noise in ['True', 'False']
        }
        ranked = sorted([row for row in arm if row['Status'] == 'OK'], key=lambda row: (
            -float(row['Inner F1']), int(row['Encoded kept']), row['Keep noise'] == 'True', row['Candidate']
        ))
        assert ranked[0]['Selection'] == chosen['selection']
        assert (ranked[0]['Keep noise'] == 'True') == chosen['keep_noise']

    _, audit_rows = read_csv(OUT / 'cleaning_audit.csv')
    audit = {row['']: row for row in audit_rows}
    expected = {
        'Invalid: prior_mean_purchase_value': (78, 11),
        'Missing mean reconstructed': (194, 30),
        'Invalid: prior_max_purchase_value': (48, 7),
        'Invalid: spend_365d': (70, 11),
        'Distinct rows with invalid values': (227, 32),
        'Remaining audited violations': (0, 0),
    }
    for key, values in expected.items():
        assert tuple(int(audit[key][c]) for c in ['Train (eligible rows)', 'Test (all rows)']) == values

    pdf = PdfReader(ROOT / 'Homework_Group04.pdf')
    assert len(pdf.pages) == 2, 'Homework report must have exactly two pages.'
    text = '\n'.join(page.extract_text() or '' for page in pdf.pages)
    for page in pdf.pages:
        assert abs(float(page.mediabox.width) - 595.28) < 1
        assert abs(float(page.mediabox.height) - 841.89) < 1
    for row in run['selected_metrics']:
        assert f"{row['F1']:.4f}" in text
    assert str(run['test_positive_predictions']) in text
    assert all(str(n) in text for n in [78, 194, 48, 70, 227, 32])
    print('PASS:', total_code, 'executed cells; two-page A4 PDF; coherent metrics/features;')
    print('      unchanged splits/search grid; 986 ordered binary test predictions; no residual audited violations.')
    return run


if __name__ == '__main__':
    verify_project()
