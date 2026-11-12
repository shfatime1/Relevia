"""Compare saved sentiment predictions against the 45 supplied AI reference labels."""
import collections
import csv
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / 'reports' / 'sentiment' / 'manual'
COMPARISON_REPORTS = ROOT / 'reports' / 'sentiment' / 'comparison'
DATA = ROOT / 'data'
LABELS = ('negative', 'neutral', 'positive')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique_by_id(rows, name):
    result = {row['review_id']: row for row in rows}
    if len(result) != len(rows) or '' in result or None in result:
        raise ValueError(f'{name} contains duplicate or empty review IDs')
    return result


def main():
    reference_path = REPORTS / 'manual_review_sample_labeled.csv'
    sample_path = REPORTS / 'manual_review_sample.csv'
    paths = {'baseline': DATA / 'review_sentiment_tweet_baseline.parquet',
             'product_review_model': DATA / 'review_sentiment_product_comparison.parquet'}
    references = read_csv(reference_path)
    sample_rows = read_csv(sample_path)
    if len(references) != 45 or len(sample_rows) != 45:
        raise ValueError('The reference and original sample must each contain 45 rows')
    references_by_id = unique_by_id(references, 'Reference CSV')
    sample = unique_by_id(sample_rows, 'Original sample')
    if set(references_by_id) != set(sample):
        raise ValueError('Reference IDs do not match the original sample')
    predictions = {name: unique_by_id(pq.read_table(path).to_pylist(), name) for name, path in paths.items()}
    truth = []
    guesses = {name: [] for name in paths}
    for row in references:
        review_id = row['review_id']
        label = row['human_label'].strip().lower()
        if label not in LABELS:
            raise ValueError(f'Invalid or empty AI reference label for {review_id}: {row["human_label"]!r}')
        for key in ('product_id', 'rating', 'review_text', 'baseline_prediction', 'product_review_prediction'):
            if row[key] != sample[review_id][key]:
                raise ValueError(f'Labeled sample changed {key} for {review_id}')
        truth.append(label)
        for name, embedded in (('baseline', 'baseline_prediction'), ('product_review_model', 'product_review_prediction')):
            prediction = predictions[name].get(review_id)
            if (prediction is None or prediction['parent_asin'] != row['product_id'] or
                    prediction['rating'] != float(row['rating']) or prediction['sentiment'] != row[embedded] or
                    prediction['sentiment'] not in LABELS):
                raise ValueError(f'{name} prediction mismatch for {review_id}')
            guesses[name].append(prediction['sentiment'])
    models = json.loads((COMPARISON_REPORTS / 'sentiment_model_comparison.json').read_text(encoding='utf-8'))
    metrics = {'reference_labels': 'AI-generated, not independent human ground truth',
               'sample_description': '45 reviews, nine per star rating, disagreement-enriched',
               'review_count': len(references), 'labels_order': list(LABELS),
               'reference_label_distribution': dict(collections.Counter(truth)),
               'rating_distribution': dict(collections.Counter(row['rating'] for row in references)),
               'model_disagreements': sum(a != b for a, b in zip(guesses['baseline'], guesses['product_review_model'])),
               'source_sha256': {'labeled_csv': sha256(reference_path), 'original_sample_csv': sha256(sample_path),
                                 **{name: sha256(path) for name, path in paths.items()}}, 'models': {}}
    for name in paths:
        report = classification_report(truth, guesses[name], labels=LABELS, output_dict=True, zero_division=0)
        metadata = models['baseline' if name == 'baseline' else 'product_review_model']
        metrics['models'][name] = {'model': metadata['model'], 'revision': metadata['revision'],
                                   'accuracy': report['accuracy'], 'macro_f1': report['macro avg']['f1-score'],
                                   'per_class': {label: report[label] for label in LABELS},
                                   'confusion_matrix': confusion_matrix(truth, guesses[name], labels=LABELS).tolist()}
    (REPORTS / 'manual_review_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    baseline, candidate = metrics['models']['baseline'], metrics['models']['product_review_model']
    if candidate['macro_f1'] > baseline['macro_f1']:
        recommendation = 'The product-review model has the higher macro F1 on this sample.'
    elif candidate['macro_f1'] < baseline['macro_f1']:
        recommendation = 'The tweet-trained baseline has the higher macro F1 on this sample.'
    else:
        recommendation = 'The models have equal macro F1 on this sample; neither is preferred on that metric.'
    lines = ['# AI-labeled review sample evaluation', '',
             '**The reference labels are AI-generated, not independent human ground truth.** '
             'The sample is small and not representative of all reviews: it contains nine reviews per star rating '
             'and deliberately oversamples model disagreements. These metrics describe agreement with these labels, '
             'not unbiased population performance.', '',
             f"Reference labels: {metrics['reference_label_distribution']}; model disagreements: {metrics['model_disagreements']} of 45.", '',
             '| Metric | Tweet-trained baseline | Product-review model |', '|---|---:|---:|',
             f"| Accuracy | {baseline['accuracy']:.2%} | {candidate['accuracy']:.2%} |",
             f"| Macro F1 | {baseline['macro_f1']:.3f} | {candidate['macro_f1']:.3f} |"]
    for label in LABELS:
        for metric in ('precision', 'recall', 'f1-score'):
            lines.append(f"| {label.title()} {metric} | {baseline['per_class'][label][metric]:.3f} | {candidate['per_class'][label][metric]:.3f} |")
    lines += ['', 'Confusion matrices use AI-generated reference labels as rows and model predictions as columns.']
    for name, model in (('Tweet-trained baseline', baseline), ('Product-review model', candidate)):
        lines += ['', f'**{name}**', '', '| Reference | Negative | Neutral | Positive |', '|---|---:|---:|---:|']
        for label, row in zip(LABELS, model['confusion_matrix']):
            lines.append(f'| {label.title()} | {row[0]} | {row[1]} | {row[2]} |')
    lines += ['', f'**Recommendation:** {recommendation} This is provisional: the AI labeling process may share '
              'biases with the models, the sample has only 45 items, and the Neutral class has few examples. '
              'The labeling CSV contains ratings and model predictions; whether those fields were hidden during '
              'AI labeling is unknown, so label contamination cannot be ruled out. '
              'The product-review model is now the default; obtain independently human-labeled, representative '
              'reviews to validate its use. Running this evaluation does not change the configured pipeline.', '',
              'Full metrics, model revisions, and input file hashes are in [manual_review_metrics.json](manual_review_metrics.json).']
    (REPORTS / 'manual_review_evaluation.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Validated {len(references)} AI labels and joined both prediction sets by review ID.')
    print(f"Baseline: accuracy {baseline['accuracy']:.3f}, macro F1 {baseline['macro_f1']:.3f}; "
          f"product-review model: accuracy {candidate['accuracy']:.3f}, macro F1 {candidate['macro_f1']:.3f}.")


if __name__ == '__main__':
    main()
