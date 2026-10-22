"""Build and verify Homework_Group04.zip from an explicit delivery allow-list."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from verify_homework import verify_project

ROOT = Path(__file__).resolve().parents[1]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    verify_project()
    files = [
        'README.md', '.gitattributes', 'requirements.txt', 'requirements-report.txt', 'Homework_Group04.pdf',
        'src/__init__.py', 'src/history_cleaning.py',
        'scripts/build_homework_report.py', 'scripts/verify_homework.py', 'scripts/package_homework.py',
        'data/train.csv', 'data/test.csv', 'data/clientes.csv', 'data/sample_submission.csv',
        'notebooks/01_data_exploration.ipynb',
        'notebooks/02_client_data_integration.ipynb',
        'notebooks/03_temporal_baseline_experiments.ipynb',
    ] + ['outputs/notebook03/' + name for name in [
        'submission_logistic_temporal.csv', 'validation_metrics.csv',
        'feature_selection_search.csv', 'selected_features.csv',
        'cleaning_audit.csv', 'report_summary.json',
    ]]
    blobs = {name: (ROOT / name).read_bytes() for name in files}
    manifest = {'purpose': 'Homework Group 04 submission; original data plus executed workflow',
                'sha256': {name: sha256(data) for name, data in blobs.items()}}
    target = ROOT / 'Homework_Group04.zip'
    with ZipFile(target, 'w', compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in blobs.items():
            archive.writestr(name, data)
        archive.writestr('delivery_manifest.json', json.dumps(manifest, indent=2))
    with ZipFile(target) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == set(files) | {'delivery_manifest.json'}
        for name, expected in manifest['sha256'].items():
            assert sha256(archive.read(name)) == expected
        assert not any('outputs/review/' in name for name in archive.namelist())
    print('Verified delivery ZIP:', target, f'({target.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
