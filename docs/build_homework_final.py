"""Build the two-page Group 04 homework report and its editable text."""
from pathlib import Path
from tempfile import gettempdir
import textwrap

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / 'Homework_Group04.pdf'
PREVIEW_DIR = Path(gettempdir()) / 'ml_group04_report_preview'
PREVIEW_DIR.mkdir(exist_ok=True)
BLUE = '#153B56'
BODY = '#263746'
LIGHT = '#EEF4F8'
RULE = '#CBDCE6'
SOURCE = ['# Homework Group 04: Repeat purchase prediction']


def page(number, subtitle):
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(.075, .953, 'MACHINE LEARNING | HOMEWORK | GROUP 04',
            fontsize=9.4, color=BLUE, weight='bold')
    ax.text(.075, .918, subtitle, fontsize=18, color=BLUE, weight='bold')
    ax.plot([.075, .925], [.898, .898], color=BLUE, linewidth=1)
    ax.text(.075, .035, 'Repeat purchase prediction | 20 October 2026',
            fontsize=8, color='#607080')
    ax.text(.925, .035, f'{number} / 2', fontsize=8,
            ha='right', color='#607080')
    return fig, ax


def paragraph(ax, y, text):
    assert not any(char in text for char in '\u2013\u2014')
    lines = textwrap.wrap(text, width=101)
    ax.text(.075, y, '\n'.join(lines), va='top', fontsize=9.7,
            color=BODY, linespacing=1.35)
    return y - len(lines) * .01555 - .013


def section(ax, y, heading, text):
    SOURCE.extend(['## ' + heading, text])
    ax.text(.075, y, heading, va='top', fontsize=11.4,
            color=BLUE, weight='bold')
    return paragraph(ax, y - .026, text)


with PdfPages(PDF) as pdf:
    pdf.infodict().update({
        'Title': 'Homework Group 04: Repeat purchase prediction',
        'Author': 'Group 04',
        'Subject': 'Exploration, preparation, feature selection and temporal assessment',
    })
    fig, ax = page(1, 'Data, decisions and validation design')
    y = section(ax, .876, 'Objective and data',
        'We predict repeat_purchase_90d: whether a customer buys again within 90 days of a '
        'purchase occasion. Each row represents one occasion. train.csv has 6,534 labelled '
        'rows and 39 columns; 55.9% of targets are positive. test.csv has 986 later unlabelled '
        'occasions. clientes.csv contains 1,143 customers and 13 fields. This binary '
        'classification task can support customer follow-up.')

    y = section(ax, y, 'Exploration and manual feature decisions',
        'Notebook 01 examines distributions, missing values, invalid history and relationships '
        'with the target. Notebook 02 audits every client field. ID and customer_id are excluded '
        'from predictors; purchase_date defines chronology. A validated left join adds '
        'country_or_region, commercial_region, island and categorical customer_since_year. '
        'Empty and nearly unique client fields are excluded; finer geography is deferred. '
        'Missing island is retained as a category because missingness may carry information.')

    SOURCE.append('## Pipeline schematic')
    ax.text(.075, y, 'Pipeline schematic', va='top', fontsize=11.4,
            color=BLUE, weight='bold')
    y -= .034
    stages = [
        ('01 + 02 | Explore and define rules',
         'Transaction audit; client audit; safe join; documented feature decisions'),
        ('03 | Clean and split chronologically',
         'Fixed corrections; outer assessment split; separate inner selection split'),
        ('03 | Fit preparation, selection and classifier',
         'Training medians and scaling; categorical encoding; optional L1 selection'),
        ('03 | Assess, choose, refit and export',
         'Compare feature sets; refit the chosen pipeline; predict the 986 test rows'),
    ]
    for i, (title, detail) in enumerate(stages):
        SOURCE.append(f'{i + 1}. {title}: {detail}.')
        ax.add_patch(FancyBboxPatch((.075, y - .041), .85, .043,
                     boxstyle='round,pad=0.005', facecolor=LIGHT, edgecolor=RULE))
        ax.text(.090, y - .001, title, va='top', fontsize=9.6,
                color=BLUE, weight='bold')
        ax.text(.090, y - .021, detail, va='top', fontsize=8.7, color=BODY)
        if i < len(stages) - 1:
            ax.annotate('', xy=(.50, y - .058), xytext=(.50, y - .047),
                        arrowprops=dict(arrowstyle='->', color=BLUE, linewidth=.9))
        y -= .063
    y += .004

    y = section(ax, y, 'Cleaning and preparation',
        'Both experiments exclude the same 65 almost empty purchase rows without valid dates, '
        'leaving 6,469 occasions. Audited negative historical intervals, suspected negative '
        'monetary errors and exact 1,000,000,000 sentinels become missing. Treating negative '
        'monetary history as invalid is an explicit assumption; ordinary high values remain. '
        'Numeric features use median imputation, missing indicators and standardisation. '
        'Categorical features use a Missing category and one-hot encoding, with unseen '
        'categories ignored. Learned transformations are fitted only on the relevant training set.')

    y = section(ax, y, 'Temporal assessment',
        'The outer training set contains 4,424 occasions from 11 July 2015 to 10 April 2022. '
        'A 90-day gap, defined from 14 April to 12 July 2022, excludes 103 occasions from '
        'assessment. Validation contains 1,942 occasions from 13 July 2022 to 15 February 2025. '
        'This gap lets training target windows finish before validation starts. Both feature '
        'sets use the same rows. Shared customers are allowed to assess future occasions '
        'by known customers. Gap rows return when fitting the final model.')
    assert y > .061, f'Page 1 content extends too low: {y}'
    pdf.savefig(fig)
    fig.savefig(PREVIEW_DIR / 'page1.png', dpi=130)
    plt.close(fig)

    fig, ax = page(2, 'Feature selection, results and final model')
    y = section(ax, .876, 'Feature selection within training data',
        'L1 logistic regression with SelectFromModel complements the manual exclusions. '
        'L1 can set coefficients to zero, allowing a smaller predictor set without fixing '
        'its size in advance. Selection follows imputation, encoding and scaling. For each '
        'feature set, eight candidates compare no selection or L1 C in {0.01, 0.1, 1}, '
        'each with and without random_noise. An inner temporal split uses only the outer '
        'training data: 2,971 training occasions, a 90-day gap of 111, and 1,342 validation '
        'occasions starting 4 July 2020. The best inner F1 determines the configuration.')

    y = section(ax, y, 'Chosen configuration and classifier',
        'Both feature sets choose L1 C=0.01 without random_noise. The best candidates tie '
        'with and without noise, so the stated preference for fewer inputs removes it; '
        'this is not evidence that noise alone caused the improvement. The selector '
        '(liblinear solver; coefficient threshold 1e-8) and preprocessing are refitted '
        'on the outer training set. The classifier is L2 logistic regression, C=1, '
        'lbfgs, maximum 10,000 iterations. The probability threshold stays fixed at 0.5.')

    SOURCE.append('## Outer validation results')
    ax.text(.075, y, 'Outer validation results', va='top', fontsize=11.4,
            color=BLUE, weight='bold')
    y -= .028
    table_rows = [
        ['Always positive', 'None', '0.7211', '0.5639', '1.0000', '0.5000'],
        ['Transactions', 'None', '0.7267', '0.7987', '0.6667', '0.7764'],
        ['Transactions + clients', 'None', '0.7056', '0.7959', '0.6338', '0.7722'],
        ['Transactions', 'L1', '0.7357', '0.7627', '0.7105', '0.7811'],
        ['Transactions + clients', 'L1', '0.7330', '0.7777', '0.6932', '0.7815'],
    ]
    table_height = .154
    table = ax.table(cellText=table_rows,
                     colLabels=['Feature set', 'Selection', 'F1', 'Precision', 'Recall', 'ROC-AUC'],
                     colWidths=[.30, .14, .14, .14, .14, .14],
                     cellLoc='center', bbox=[.075, y - table_height, .85, table_height])
    table.auto_set_font_size(False)
    table.set_fontsize(8.4)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(RULE)
        cell.set_facecolor(BLUE if row == 0 else (LIGHT if row == 4 else 'white'))
        if row == 0:
            cell.set_text_props(color='white', weight='bold')
        elif row == 4:
            cell.set_text_props(weight='bold')
    SOURCE.append('\n'.join([
        '| Feature set | Selection | F1 | Precision | Recall | ROC-AUC |',
        '|---|---|---:|---:|---:|---:|',
        *['| ' + ' | '.join(row) + ' |' for row in table_rows],
    ]))
    y -= table_height + .016
    interpretation = (
        'We prioritise F1 to balance precision and recall for repeat purchases. Scores use the same outer validation '
        'rows. Selection improves F1 by '
        '0.90 percentage points for transactions and 2.73 for transactions plus clients. '
        'The chosen transaction model exceeds the constant-positive reference by 1.46 points. '
        'Client enrichment does not improve F1 here; the difference after selection is only '
        '0.27 points, so it does not establish a general disadvantage of client information.')
    y = paragraph(ax, y, interpretation)
    SOURCE.append(interpretation)

    y = section(ax, y, 'Final feature set and predictions',
        'After choosing transactions without clients, the complete pipeline is refitted '
        'on all 6,469 eligible labelled occasions. Its selector retains 12 encoded columns '
        'from 11 original variables: Total Imp_, Total Desc_ Linha, prior_total_spend, '
        'prior_std_purchase_value, prior_mean_gap_days, prior_std_gap_days, purchase_count_90d, '
        'spend_180d, purchase_count_365d, salesperson and commercial_zone. Two retained columns '
        'are missing indicators and two are categorical levels. Refitting on more data can '
        'change the selected columns. The pipeline exports 986 binary predictions, checking '
        'IDs, order and CSV format. Test spans 24 May 2025 to 6 October 2026, after all '
        'training label windows finish.')

    y = section(ax, y, 'Scope of the conclusions',
        'EDA used all labelled data, and outer scores informed the final feature-set choice. '
        'These scores describe development performance, not an independent final estimate. '
        'One holdout cannot establish stability over time. We assume client attributes were '
        'available at purchase; consistent entry years do not prove historical availability.')
    assert y > .061, f'Page 2 content extends too low: {y}'
    pdf.savefig(fig)
    fig.savefig(PREVIEW_DIR / 'page2.png', dpi=130)
    plt.close(fig)

(ROOT / 'docs/Homework_Group04.md').write_text('\n\n'.join(SOURCE) + '\n')
print(PDF)
