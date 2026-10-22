"""Build the two-page report from the latest executed Notebook 03 outputs.

Optional reporting dependencies: pip install -r requirements-report.txt
"""
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    KeepTogether,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'notebook03'
BLUE = colors.HexColor('#145c9e')
INK = colors.HexColor('#182d41')
LIGHT = colors.HexColor('#edf4fa')
GREY = colors.HexColor('#536474')
WIDTH = A4[0] - 88
BODY = ParagraphStyle('body', fontName='Helvetica', fontSize=9,
                      leading=12, textColor=INK, spaceAfter=6,
                      splitLongWords=True, alignment=TA_LEFT)
SMALL = ParagraphStyle('small', parent=BODY, fontSize=8, leading=10.4, spaceAfter=4)
HEAD = ParagraphStyle('head', parent=BODY, fontName='Helvetica-Bold',
                      fontSize=11, leading=14, textColor=BLUE,
                      spaceBefore=8, spaceAfter=5)
TITLE = ParagraphStyle('title', parent=HEAD, fontSize=17, leading=21,
                       spaceBefore=0, spaceAfter=9)


def p(text, style=BODY):
    return Paragraph(text, style)


def table(rows, widths, header=True, font_size=8):
    style = ParagraphStyle('table_text', parent=SMALL, fontSize=font_size,
                           leading=font_size + 2.3, spaceAfter=0)
    cells = [[p(escape(str(value)), style) for value in row] for row in rows]
    t = Table(cells, colWidths=widths, hAlign='LEFT', repeatRows=int(header))
    settings = [
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LINEBELOW', (0, -1), (-1, -1), .4, colors.HexColor('#cbd9e4')),
    ]
    if header:
        settings += [('BACKGROUND', (0, 0), (-1, 0), LIGHT),
                     ('LINEBELOW', (0, 0), (-1, 0), .6, BLUE)]
    t.setStyle(TableStyle(settings))
    return t


def furniture(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(BLUE)
    canvas.setFont('Helvetica-Bold', 8)
    canvas.drawString(44, A4[1] - 29, 'MACHINE LEARNING  |  HOMEWORK  |  GROUP 04')
    canvas.setStrokeColor(colors.HexColor('#cbd9e4'))
    canvas.line(44, 35, A4[0] - 44, 35)
    canvas.setFillColor(GREY)
    canvas.setFont('Helvetica', 8)
    canvas.drawString(44, 23, 'Repeat purchase prediction | 2026/2027')
    canvas.drawRightString(A4[0] - 44, 23, f'{doc.page} / 2')
    canvas.restoreState()


def main():
    run = json.loads((OUT / 'report_summary.json').read_text(encoding='utf-8'))
    with (OUT / 'cleaning_audit.csv').open(newline='', encoding='utf-8') as f:
        audit = {row['']: row for row in csv.DictReader(f)}
    rows = run['rows']
    split = run['split']
    inner = run['inner']
    metrics = {row['Model']: row for row in run['selected_metrics']}
    baseline = {row['Model']: row for row in run['baseline_metrics']}
    winner = metrics[run['selected_version']]
    story = [
        p('Data, corrections and assessment', TITLE),
        p('Objective and inputs', HEAD),
        p(f"Predict <b>repeat_purchase_90d</b>: whether the customer purchases again within "
          f"90 days of the current purchase occasion. Each row is an occasion, not a customer. "
          f"Train has {rows['raw_train']:,} labelled rows and {rows['raw_train_columns']} columns "
          f"({run['train_positive_rate']:.1%} positive); test has {rows['test']:,} later unlabelled rows. "
          f"The optional client table has {rows['clients']:,} customers and {rows['client_columns']} fields."),
        p('Pipeline and fit boundaries', HEAD),
        table([
            ['Stage / notebook', 'Technique and rationale', 'Data boundary'],
            ['1. Audit / 01 + 02', 'Describe purchases and clients; inspect missingness, target relationships and cross-column identities.', 'Raw files preserved.'],
            ['2. Prepare / 03', 'Repair exact same-row identities; otherwise invalidate impossible history. Exclude 65 undated train rows.', 'No fitted statistics or target used.'],
            ['3. Feature sets / 02 + 03', 'Transactions only versus + country, commercial region, island and categorical entry year; validated customer join.', 'Same purchase IDs in both branches.'],
            ['4. Fit / 03', 'Training medians + residual missing indicators; StandardScaler; Missing category; one-hot; optional L1; logistic classifier.', 'Fit on each training partition only.'],
            ['5. Select and assess / 03', 'Inner F1 chooses L1/noise settings; main validation compares the two feature sets.', 'Chronological splits; both have 90-day gaps.'],
            ['6. Refit and export / 03', 'Refit the complete chosen rule on all eligible labelled rows; predict every test ID.', 'Test transformed only; no test fitting.'],
        ], [105, 280, WIDTH - 385], font_size=7.8),
        p('Audited corrections', HEAD),
    ]
    audit_rows = [['Issue / exact recovery', 'Train', 'Test']]
    for label, key in [
        ('Inconsistent historical mean', 'Invalid: prior_mean_purchase_value'),
        ('Missing mean recovered with positive count', 'Missing mean reconstructed'),
        ('Impossible historical maximum', 'Invalid: prior_max_purchase_value'),
        ('Impossible annual spend', 'Invalid: spend_365d'),
        ('Distinct purchases with invalid cells', 'Distinct rows with invalid values'),
    ]:
        audit_rows.append([label, audit[key]['Train (eligible rows)'], audit[key]['Test (all rows)']])
    story += [
        table(audit_rows, [WIDTH - 100, 50, 50]),
        p('Rebuild mean as total/count only when count is positive. Recover four train maxima exactly '
          '(one prior purchase: total; two: total minus minimum); leave other invalid maxima missing. '
          'Equal-count nested windows recover 108/5 annual amounts, including missing values. '
          'Unrecoverable values and negative time intervals stay missing for fold-fitted imputation. '
          'Counts/totals/minima/spread and shorter windows are trusted after consistency checks. '
          'No rounding-by-default, future events or target-derived repairs are used.', SMALL),
        p('Temporal design', HEAD),
        p(f"Main training: <b>{split['train']:,}</b> purchases, {split['train_first']} to {split['train_last']}. "
          f"A 90-day window starting {split['training_limit']} excludes <b>{split['gap']}</b> purchases. "
          f"Validation: <b>{split['validation']:,}</b>, {split['validation_first']} to {split['validation_last']}. "
          'Complete training target windows finish before validation starts. Shared customers are allowed '
          'because the task includes their future purchases. Dates are anonymised but order and intervals are preserved.', SMALL),
        PageBreak(),
        p('Selection, evidence and final solution', TITLE),
        p('Selection and modelling choices', HEAD),
        p(f"Inside the {split['train']:,} main-training rows, an inner temporal split uses "
          f"{inner['train']:,} fitting rows, a 90-day gap of {inner['gap']}, and {inner['validation']:,} "
          f"validation rows starting {inner['validation_first']}. Compare eight settings per feature set: "
          'no selection or L1 C in {0.01, 0.1, 1}, each with/without random_noise. '
          'Preprocessing and selection are refitted inside each candidate.'),
        p('Embedded L1 is suitable for the chosen logistic family and related historical predictors. '
          'SelectFromModel retains coefficients with magnitude at least 1e-8. Choose the highest inner F1; '
          'exact ties prefer fewer encoded columns, then no noise. Smaller C means stronger selection. '
          'The final classifier stays L2 logistic regression (C=1, lbfgs, max_iter=10,000), '
          'with threshold 0.5. Standardisation prevents numeric units dominating regularisation; '
          'medians provide a simple outlier-resistant fill for residual numeric gaps.'),
        p('Main validation results', HEAD),
    ]
    metric_rows = [['Feature set', 'Selection', 'F1', 'Precision', 'Recall', 'ROC-AUC']]
    for row in run['baseline_metrics']:
        metric_rows.append([row['Model'], 'None', *[f"{row[k]:.4f}" for k in ['F1', 'Precision', 'Recall', 'ROC-AUC']]])
    for row in run['selected_metrics']:
        if row['Model'] != 'Always positive':
            metric_rows.append([row['Model'], run['settings_by_feature_set'][row['Model']]['selection'],
                                *[f"{row[k]:.4f}" for k in ['F1', 'Precision', 'Recall', 'ROC-AUC']]])
    story += [table(metric_rows, [147, 78, 61, 66, 61, WIDTH - 413], font_size=7.6)]
    change = metrics['Transactions + clients']['F1'] - metrics['Transactions']['F1']
    gain = winner['F1'] - metrics['Always positive']['F1']
    story += [p(
        f"F1 is the supplied competition metric; precision, recall and ROC-AUC give context. "
        f"Selected feature set: <b>{escape(run['selected_version'])}</b>, F1 <b>{winner['F1']:.4f}</b>. "
        f"The margin over the always-positive reference is {gain*100:.2f} percentage points. "
        f"Adding clients changes selected-procedure F1 by {change*100:+.2f} points on this holdout. "
        'This small difference does not establish a general advantage or disadvantage of client data.'),
        p('Noise, final features and predictions', HEAD)]
    noise_status = 'retained' if 'random_noise' in run['final_raw_names'] else 'discarded'
    noise_input = 'included as a candidate' if run['keep_noise'] else 'removed before encoding'
    story += [p(
        f"The chosen rule is {escape(run['selection'])}, with raw noise {noise_input}; "
        f"random_noise is {noise_status} by the final selection. This empirical decision is not a leakage test. "
        f"Refit the complete pipeline on all {rows['eligible_train']:,} eligible labelled purchases, "
        f"including the old gap and validation. The final selector retains {run['final_encoded_retained']} "
        f"of {run['final_encoded_before']} encoded columns, representing {run['final_raw_retained']} original variables."),
        p('<b>Final original variables:</b> ' + escape(', '.join(run['final_raw_names'])) + '.', SMALL),
        p(f"Refitting can change selected columns. The export contains {rows['test']} binary predictions "
          f"({run['test_positive_predictions']} positive), with original IDs, order and read-back checks. "
          f"Test runs from {split['test_first']} to {split['test_last']}; all training label windows finish before it starts. "
          'The predicted positive fraction is not test accuracy.', SMALL),
        p('Limits and reproducibility', HEAD),
        p('EDA used all labelled data and main scores informed the final feature-set choice: these are '
          'development results, not an independent final estimate. One inner/outer holdout cannot establish '
          'stability or statistical significance. Current client attributes are assumed historically available. '
          'The amount identities and nonnegative history support the correction rules but remain measurement assumptions. '
          'The fixed threshold was not optimised. All three notebooks, shared source, original CSVs, environment '
          'requirements and generated evidence accompany the submission.', SMALL),
        p('Class basis: Feature Selection slides (embedded L1), Week 04 classification (SelectFromModel), '
          'and Week 05 model selection (fit within partitions; refit the chosen pipeline).', SMALL),
    ]
    target = ROOT / 'Homework_Group04.pdf'
    doc = SimpleDocTemplate(str(target), pagesize=A4, rightMargin=44, leftMargin=44,
                            topMargin=49, bottomMargin=45, title='Homework Group 04',
                            author='ML Group 04')
    doc.build(story, onFirstPage=furniture, onLaterPages=furniture)
    print(target)


if __name__ == '__main__':
    main()
