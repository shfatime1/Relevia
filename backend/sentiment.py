"""Predict and evaluate three-class review sentiment, or inspect one product."""
import argparse
import collections
import csv
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
CACHE = ROOT / '.cache' / 'sentiment_product_comparison'
REPORTS = ROOT / 'reports' / 'sentiment'
FINAL_REPORTS = REPORTS / 'final'
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = 'MilindSingh316/distilbert-clothing-review-sentiment'
REVISION = '1afdfec74e1df72dabf0b82e011d89ecbee9e614'
LABELS = ('negative', 'neutral', 'positive')
MAX_LENGTH = 128


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def review_ids(reviews):
    occurrences = collections.Counter()
    ids = []
    for review in reviews:
        content = json.dumps(review, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
        digest = hashlib.sha256(content.encode('utf-8')).hexdigest()
        occurrences[digest] += 1
        ids.append(f'{digest}-{occurrences[digest]}')
    return ids


def save_cache(fingerprint, predictions, confidence):
    temporary = CACHE / 'predictions.tmp.npz'
    np.savez_compressed(temporary, fingerprint=json.dumps(fingerprint, sort_keys=True),
                        predictions=predictions, confidence=confidence)
    temporary.replace(CACHE / 'predictions.npz')


def predict(reviews, batch_size):
    CACHE.mkdir(parents=True, exist_ok=True)
    fingerprint = {'reviews_sha256': file_hash(DATA / 'reviews.parquet'), 'model': MODEL,
                   'revision': REVISION, 'max_length': MAX_LENGTH, 'input': 'review_text_v1'}
    predictions = np.full(len(reviews), -1, dtype=np.int8)
    confidence = np.zeros(len(reviews), dtype=np.float32)
    try:
        with np.load(CACHE / 'predictions.npz', allow_pickle=False) as cached:
            if (json.loads(str(cached['fingerprint'])) == fingerprint and
                    cached['predictions'].shape == predictions.shape and cached['confidence'].shape == confidence.shape):
                predictions = cached['predictions'].copy()
                confidence = cached['confidence'].copy()
                if not (np.isin(predictions, [-1, 0, 1, 2]).all() and np.isfinite(confidence).all() and
                        ((confidence >= 0) & (confidence <= 1)).all()):
                    raise ValueError('Invalid cached predictions')
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        predictions.fill(-1); confidence.fill(0)
    pending = np.flatnonzero(predictions < 0).tolist()
    reused = len(reviews) - len(pending)
    print(f'Reusing {reused} cached predictions; {len(pending)} remaining.', flush=True)
    if not pending:
        return predictions, confidence, fingerprint
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu': torch.set_num_threads(min(4, os.cpu_count() or 1))
    options = dict(revision=REVISION, cache_dir=str(ROOT / '.cache' / 'huggingface'))
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, **options)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL, local_files_only=True, **options)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, **options)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL, **options)
    model.to(device).eval()
    mapping = [str(model.config.id2label[i]).lower() for i in range(model.config.num_labels)]
    if sorted(mapping) != sorted(LABELS): raise ValueError(f'Unexpected model labels: {mapping}')
    # Group similar lengths to reduce padding; output remains in source order.
    pending.sort(key=lambda i: len(reviews[i]['text']))
    print(f'Running {len(pending)} reviews on {device} in batches of {batch_size}.', flush=True)
    for start in range(0, len(pending), batch_size):
        indices = pending[start:start + batch_size]
        inputs = tokenizer([reviews[i]['text'] for i in indices], padding=True, truncation=True,
                           max_length=MAX_LENGTH, return_tensors='pt').to(device)
        with torch.inference_mode():
            probs = torch.softmax(model(**inputs).logits.float(), dim=-1)
            scores, classes = probs.max(dim=-1)
        predictions[indices] = [LABELS.index(mapping[i]) for i in classes.cpu().tolist()]
        confidence[indices] = scores.cpu().numpy()
        done = min(start + len(indices), len(pending))
        if done == len(pending) or done // 512 != start // 512:
            save_cache(fingerprint, predictions, confidence)
            print(f'Predicted {done}/{len(pending)} new reviews.', flush=True)
    return predictions, confidence, fingerprint


def run(batch_size, baseline=False):
    started = time.perf_counter()
    reviews = pq.read_table(DATA / 'reviews.parquet').to_pylist()
    products = pq.read_table(DATA / 'products.parquet').to_pylist()
    product_ids = [p['parent_asin'] for p in products]
    if len(set(product_ids)) != len(product_ids) or any(r['parent_asin'] not in set(product_ids) for r in reviews):
        raise ValueError('Invalid product IDs or unmatched reviews')
    if any(not isinstance(r['text'], str) or not r['text'].strip() or r['rating'] not in (1, 2, 3, 4, 5) for r in reviews):
        raise ValueError('Each review must have non-empty text and an integer star rating from 1 to 5')
    ids = review_ids(reviews)
    predicted, confidence, fingerprint = predict(reviews, batch_size)
    rows = [{'review_id': review_id, 'parent_asin': review['parent_asin'], 'rating': review['rating'],
             'sentiment': LABELS[int(label)], 'confidence': float(score)}
            for review_id, review, label, score in zip(ids, reviews, predicted, confidence)]
    review_output = 'review_sentiment_tweet_baseline.parquet' if baseline else 'review_sentiment.parquet'
    product_output = 'product_sentiment_tweet_baseline.parquet' if baseline else 'product_sentiment.parquet'
    pq.write_table(pa.Table.from_pylist(rows), DATA / review_output, compression='zstd')
    grouped = collections.defaultdict(list)
    for row in rows: grouped[row['parent_asin']].append(row)
    aggregates = []
    for product_id in product_ids:
        items = grouped[product_id]
        if not items: raise ValueError(f'Product has no reviews: {product_id}')
        counts = collections.Counter(item['sentiment'] for item in items)
        result = {'parent_asin': product_id, 'review_count': len(items),
                  'average_rating': sum(item['rating'] for item in items) / len(items)}
        for label in LABELS:
            result[f'{label}_count'] = counts[label]
            result[f'{label}_percent'] = counts[label] * 100.0 / len(items)
        aggregates.append(result)
    pq.write_table(pa.Table.from_pylist(aggregates), DATA / product_output, compression='zstd')
    if not baseline:
        evaluate(reviews, rows, fingerprint)
    print(f'Wrote {len(rows)} review predictions and {len(aggregates)} product summaries in {time.perf_counter()-started:.1f}s.', flush=True)


def evaluate(reviews, rows, fingerprint):
    truth = ['negative' if r['rating'] <= 2 else 'neutral' if r['rating'] == 3 else 'positive' for r in reviews]
    predicted = [r['sentiment'] for r in rows]
    matrix = confusion_matrix(truth, predicted, labels=LABELS)
    report = classification_report(truth, predicted, labels=LABELS, output_dict=True, zero_division=0)
    metrics = {'model': MODEL, 'revision': REVISION, 'input': 'review text only', 'max_length_tokens': MAX_LENGTH,
               'reviews_sha256': fingerprint['reviews_sha256'], 'review_count': len(rows),
               'weak_label_rule': '1-2 negative, 3 neutral, 4-5 positive',
               'labels_order': list(LABELS), 'agreement_rate': report['accuracy'],
               'confusion_matrix': matrix.tolist(), 'classification_report': report,
               'star_label_distribution': dict(collections.Counter(truth)),
               'prediction_distribution': dict(collections.Counter(predicted))}
    FINAL_REPORTS.mkdir(parents=True, exist_ok=True)
    (FINAL_REPORTS / 'sentiment_final_metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap='Blues')
    ax.set(xticks=range(3), yticks=range(3), xticklabels=LABELS, yticklabels=LABELS,
           xlabel='Predicted sentiment', ylabel='Star-derived weak label', title='Sentiment vs. star-derived labels')
    for row in range(3):
        for col in range(3):
            ax.text(col, row, str(matrix[row, col]), ha='center', va='center',
                    color='white' if matrix[row, col] > matrix.max() / 2 else 'black')
    fig.colorbar(image, ax=ax); fig.tight_layout()
    fig.savefig(FINAL_REPORTS / 'sentiment_final_confusion_matrix.png', dpi=160); plt.close(fig)
    with (FINAL_REPORTS / 'sentiment_final_examples.csv').open('w', encoding='utf-8', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(['review_id', 'parent_asin', 'rating', 'weak_label', 'sentiment', 'confidence', 'review_text'])
        # Highest-confidence examples in each weak-label/prediction cell; not a random sample.
        for actual in LABELS:
            for guess in LABELS:
                selected = [i for i, (a, p) in enumerate(zip(truth, predicted)) if a == actual and p == guess]
                selected.sort(key=lambda i: (-rows[i]['confidence'], ids_key(rows[i])))
                for i in selected[:3]:
                    writer.writerow([rows[i]['review_id'], rows[i]['parent_asin'], rows[i]['rating'], actual,
                                     guess, rows[i]['confidence'], reviews[i]['text']])
    lines = ['# Final sentiment evaluation (product-review model)\n',
             f'Analyzed {len(rows):,} reviews with [{MODEL}](https://huggingface.co/{MODEL}) at revision `{REVISION}`. '
             f'The model reads review text (not the rating), truncated to {MAX_LENGTH} tokens.\n',
             'Star ratings are weak labels: 1–2 stars are negative, 3 neutral, and 4–5 positive. '
             'Agreement measures correspondence with those labels, not accuracy against verified sentiment. '
             'The confusion matrix uses weak labels as rows and predictions as columns. Precision, recall, '
             'and F1 in the JSON compare each class to those weak labels.\n',
             f"Agreement: **{metrics['agreement_rate']:.1%}**; macro F1: **{report['macro avg']['f1-score']:.3f}**.\n",
             f"Neutral precision: **{report['neutral']['precision']:.1%}**; neutral recall: **{report['neutral']['recall']:.1%}**. "
             'These figures show substantial disagreement on the neutral class under the star-label proxy.\n',
             'The model was trained on clothing reviews rather than electronics reviews; domain shift, sarcasm, '
             'mixed opinions, truncation, and uncalibrated confidence can affect results. A star rating can disagree '
             'with the sentiment of the text, especially for three-star reviews. The selected source reviews are '
             'not a representative sample of all Amazon reviews. Examples are the highest-confidence rows per '
             'non-empty confusion-matrix cell, not a random sample. Manual labeling is needed for a ground-truth evaluation.\n']
    (FINAL_REPORTS / 'sentiment_final_evaluation.md').write_text('\n'.join(lines), encoding='utf-8')


def ids_key(row):
    return row['review_id']


def inspect(product_id):
    rows = pq.read_table(DATA / 'product_sentiment.parquet').to_pylist()
    match = next((p for p in rows if p['parent_asin'] == product_id), None)
    if match is None: raise SystemExit(f'No sentiment summary for product {product_id}')
    product = next((p for p in pq.read_table(DATA / 'products.parquet').to_pylist() if p['parent_asin'] == product_id), None)
    print(json.dumps({'product_id': product_id, 'title': product['title'] if product else None, **match}, indent=2))


def validate():
    reviews = pq.read_table(DATA / 'reviews.parquet').to_pylist()
    products = pq.read_table(DATA / 'products.parquet').to_pylist()
    rows = pq.read_table(DATA / 'review_sentiment.parquet').to_pylist()
    summaries = pq.read_table(DATA / 'product_sentiment.parquet').to_pylist()
    if len(reviews) != 8662 or len(products) != 1000 or len(rows) != len(reviews) or len(summaries) != len(products):
        raise ValueError('Unexpected input or output counts')
    ids = review_ids(reviews)
    if len(set(ids)) != len(ids): raise ValueError('Review IDs must be unique')
    grouped = collections.defaultdict(list)
    truth = []
    for source, row, review_id in zip(reviews, rows, ids):
        if (row['review_id'] != review_id or row['parent_asin'] != source['parent_asin'] or
                row['rating'] != source['rating'] or row['sentiment'] not in LABELS or
                not 0 <= row['confidence'] <= 1):
            raise ValueError(f'Invalid review prediction: {review_id}')
        grouped[row['parent_asin']].append(row)
        truth.append('negative' if source['rating'] <= 2 else 'neutral' if source['rating'] == 3 else 'positive')
    summaries_by_id = {row['parent_asin']: row for row in summaries}
    if len(summaries_by_id) != len(products) or set(summaries_by_id) != {p['parent_asin'] for p in products}:
        raise ValueError('Product summary IDs do not match products')
    for asin, summary in summaries_by_id.items():
        items = grouped[asin]
        if not items or summary['review_count'] != len(items): raise ValueError(f'Incorrect count: {asin}')
        if not np.isclose(summary['average_rating'], np.mean([x['rating'] for x in items])): raise ValueError(f'Incorrect rating: {asin}')
        for label in LABELS:
            count = sum(x['sentiment'] == label for x in items)
            if summary[f'{label}_count'] != count or not np.isclose(summary[f'{label}_percent'], count * 100 / len(items)):
                raise ValueError(f'Incorrect {label} aggregation: {asin}')
    metrics = json.loads((FINAL_REPORTS / 'sentiment_final_metrics.json').read_text(encoding='utf-8'))
    if (metrics['model'] != MODEL or metrics['revision'] != REVISION or
            metrics['reviews_sha256'] != file_hash(DATA / 'reviews.parquet') or
            metrics['confusion_matrix'] != confusion_matrix(truth, [r['sentiment'] for r in rows], labels=LABELS).tolist()):
        raise ValueError('Report does not match predictions and source')
    fingerprint = {'reviews_sha256': file_hash(DATA / 'reviews.parquet'), 'model': MODEL,
                   'revision': REVISION, 'max_length': MAX_LENGTH, 'input': 'review_text_v1'}
    with np.load(CACHE / 'predictions.npz', allow_pickle=False) as cached:
        if json.loads(str(cached['fingerprint'])) != fingerprint or np.any(cached['predictions'] < 0):
            raise ValueError('Cached predictions are incomplete or stale')
        if [LABELS[int(i)] for i in cached['predictions']] != [r['sentiment'] for r in rows] or not np.allclose(cached['confidence'], [r['confidence'] for r in rows]):
            raise ValueError('Cached predictions differ from output')
    print(f'Validated {len(rows)} predictions, {len(summaries)} product summaries, metrics, and complete reusable cache.')


def main():
    global MODEL, REVISION, CACHE
    parser = argparse.ArgumentParser(description='Analyze review sentiment or inspect a product summary.')
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('run')
    prepare.add_argument('--batch-size', type=int, default=32)
    prepare.add_argument('--baseline', action='store_true', help='Recreate the separate tweet-model baseline datasets')
    product = commands.add_parser('product')
    product.add_argument('parent_asin')
    commands.add_parser('validate')
    args = parser.parse_args()
    if args.command == 'run':
        if args.batch_size < 1: parser.error('Batch size must be positive')
        if args.baseline:
            MODEL = 'ZyroGod/distilbert-sentiment-classifier'
            REVISION = 'cbfee930df13824cf267b8d93921dd3a1783af83'
            CACHE = ROOT / '.cache' / 'sentiment'
        run(args.batch_size, baseline=args.baseline)
    elif args.command == 'product':
        inspect(args.parent_asin)
    else:
        validate()


if __name__ == '__main__':
    main()
