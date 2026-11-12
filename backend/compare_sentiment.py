"""Compare the saved Phase 3 baseline with a product-review sentiment model."""
import argparse
import collections
import json

import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import classification_report, confusion_matrix

from backend import sentiment

CANDIDATE = 'MilindSingh316/distilbert-clothing-review-sentiment'
REVISION = '1afdfec74e1df72dabf0b82e011d89ecbee9e614'
REPORTS = sentiment.REPORTS / 'comparison'
BASELINE_REPORTS = sentiment.REPORTS / 'baseline'
DATA = sentiment.DATA
LABELS = sentiment.LABELS


def evaluate(truth, predictions):
    report = classification_report(truth, predictions, labels=LABELS, output_dict=True, zero_division=0)
    return {'agreement_rate': report['accuracy'], 'macro_f1': report['macro avg']['f1-score'],
            'per_class': {label: report[label] for label in LABELS},
            'confusion_matrix': confusion_matrix(truth, predictions, labels=LABELS).tolist(),
            'prediction_distribution': dict(collections.Counter(predictions))}


def run(batch_size):
    if batch_size < 1:
        raise ValueError('Batch size must be positive')
    reviews = pq.read_table(DATA / 'reviews.parquet').to_pylist()
    baseline_rows = pq.read_table(DATA / 'review_sentiment_tweet_baseline.parquet').to_pylist()
    baseline_report = json.loads((BASELINE_REPORTS / 'sentiment_metrics.json').read_text(encoding='utf-8'))
    ids = sentiment.review_ids(reviews)
    if len(reviews) != 8662 or len(baseline_rows) != len(reviews) or len(set(ids)) != len(ids):
        raise ValueError('Unexpected source or baseline review count')
    if baseline_report['reviews_sha256'] != sentiment.file_hash(DATA / 'reviews.parquet'):
        raise ValueError('Baseline report refers to a different reviews file')
    for review, review_id, row in zip(reviews, ids, baseline_rows):
        if (row['review_id'] != review_id or row['parent_asin'] != review['parent_asin'] or
                row['rating'] != review['rating'] or row['sentiment'] not in LABELS):
            raise ValueError(f'Baseline prediction does not match source review {review_id}')
    truth = ['negative' if r['rating'] <= 2 else 'neutral' if r['rating'] == 3 else 'positive' for r in reviews]
    baseline = evaluate(truth, [row['sentiment'] for row in baseline_rows])
    if baseline['confusion_matrix'] != baseline_report['confusion_matrix']:
        raise ValueError('Baseline predictions do not match the saved baseline report')
    baseline['model'] = baseline_report['model']
    baseline['revision'] = baseline_report['revision']
    baseline['max_length_tokens'] = baseline_report['max_length_tokens']
    if baseline['max_length_tokens'] != sentiment.MAX_LENGTH:
        raise ValueError('Baseline and candidate token limits differ')

    # Reuse the existing batch inference/checkpoint logic with isolated model and cache settings.
    sentiment.MODEL = CANDIDATE
    sentiment.REVISION = REVISION
    sentiment.CACHE = sentiment.ROOT / '.cache' / 'sentiment_product_comparison'
    predictions, confidence, fingerprint = sentiment.predict(reviews, batch_size)
    labels = [LABELS[int(i)] for i in predictions]
    candidate_rows = [{'review_id': review_id, 'parent_asin': review['parent_asin'],
                       'rating': review['rating'], 'sentiment': label, 'confidence': float(score)}
                      for review_id, review, label, score in zip(ids, reviews, labels, confidence)]
    pq.write_table(pa.Table.from_pylist(candidate_rows), DATA / 'review_sentiment_product_comparison.parquet', compression='zstd')
    candidate = evaluate(truth, labels)
    candidate.update(model=CANDIDATE, revision=REVISION, max_length_tokens=sentiment.MAX_LENGTH)
    metrics = {'review_count': len(reviews), 'reviews_sha256': fingerprint['reviews_sha256'],
               'input': 'review text only', 'weak_label_rule': '1-2 negative, 3 neutral, 4-5 positive',
               'labels_order': list(LABELS), 'star_label_distribution': dict(collections.Counter(truth)),
               'baseline': baseline, 'product_review_model': candidate}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / 'sentiment_model_comparison.json').write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')

    b_neutral = baseline['per_class']['neutral']['f1-score']
    c_neutral = candidate['per_class']['neutral']['f1-score']
    if c_neutral > b_neutral:
        choice = 'product-review model'
        reason = 'It has the higher Neutral F1 on this weak-label comparison.'
    elif c_neutral < b_neutral:
        choice = 'tweet-trained baseline'
        reason = 'It has the higher Neutral F1 on this weak-label comparison.'
    else:
        choice = 'product-review model' if candidate['macro_f1'] > baseline['macro_f1'] else 'tweet-trained baseline'
        reason = 'Neutral F1 is tied; this model has the higher macro F1.'
    lines = ['# Sentiment model comparison', '',
             f"Both models classified the same {len(reviews):,} review texts using at most {sentiment.MAX_LENGTH} tokens. "
             'Ratings were used only to form weak comparison labels: 1–2 Negative, 3 Neutral, 4–5 Positive. '
             f"Source SHA-256: `{fingerprint['reviews_sha256']}`.", '',
             f"Baseline: [{baseline['model']}](https://huggingface.co/{baseline['model']}) at `{baseline['revision']}`. "
             f"Product-review model: [{CANDIDATE}](https://huggingface.co/{CANDIDATE}) at `{REVISION}`; "
             'its model card describes training on clothing reviews.', '',
             '| Metric | Tweet-trained baseline | Product-review model |', '|---|---:|---:|',
             f"| Agreement | {baseline['agreement_rate']:.2%} | {candidate['agreement_rate']:.2%} |",
             f"| Macro F1 | {baseline['macro_f1']:.3f} | {candidate['macro_f1']:.3f} |"]
    for label in LABELS:
        lines.append(f"| {label.title()} F1 | {baseline['per_class'][label]['f1-score']:.3f} | {candidate['per_class'][label]['f1-score']:.3f} |")
    lines += ['', f"Neutral precision / recall: baseline {baseline['per_class']['neutral']['precision']:.1%} / "
              f"{baseline['per_class']['neutral']['recall']:.1%}; product-review model "
              f"{candidate['per_class']['neutral']['precision']:.1%} / {candidate['per_class']['neutral']['recall']:.1%}."]
    lines += ['', 'Confusion matrices use star-derived labels as rows and predictions as columns, ordered Negative, Neutral, Positive.']
    for name, item in (('Tweet-trained baseline', baseline), ('Product-review model', candidate)):
        lines += ['', f'**{name}**', '', '| Weak label | Negative | Neutral | Positive |', '|---|---:|---:|---:|']
        for label, row in zip(LABELS, item['confusion_matrix']):
            lines.append(f'| {label.title()} | {row[0]:,} | {row[1]:,} | {row[2]:,} |')
    lines += ['', f'**Recommendation:** Prefer the {choice} for a follow-up review. {reason} '
              'This recommendation prioritizes Neutral F1; also consider agreement and macro F1 above. '
              'The product-review model is the default; this comparison does not change the configured pipeline.',
              '', 'These metrics measure agreement with ratings rather than human-verified text sentiment. '
              'The 3-star Neutral proxy can contain mixed or strongly polarized text, and class imbalance affects agreement. '
              'The candidate was trained on clothing, not electronics; its product-review training may use related rating-derived labels. '
              'Overlap with the evaluation data has not been independently verified. Both models truncate longer reviews. '
              'The selected reviews are not representative of the full Amazon catalog. A blinded, human-labeled '
              'electronics-review sample is needed before treating the recommendation as validated sentiment quality.',
              '', 'Full per-class precision, recall, F1, support, and prediction distributions are in '
              '[sentiment_model_comparison.json](sentiment_model_comparison.json).']
    (REPORTS / 'sentiment_model_comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f"Compared {len(reviews)} reviews. Baseline macro F1 {baseline['macro_f1']:.3f}; candidate {candidate['macro_f1']:.3f}.")


def main():
    parser = argparse.ArgumentParser(description='Compare the baseline with a product-review sentiment model.')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    run(args.batch_size)


if __name__ == '__main__':
    main()
