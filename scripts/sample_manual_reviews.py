"""Create a reproducible, disagreement-enriched manual sentiment sample."""
import collections
import csv
import random
from pathlib import Path

import pyarrow.parquet as pq

from backend.sentiment import review_ids

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
OUTPUT = ROOT / 'reports' / 'sentiment' / 'manual' / 'manual_review_sample.csv'
SEED = 20261112


def main():
    if OUTPUT.exists():
        raise SystemExit(f'{OUTPUT} already exists; refusing to overwrite possible human labels.')
    reviews = pq.read_table(DATA / 'reviews.parquet').to_pylist()
    baseline = pq.read_table(DATA / 'review_sentiment_tweet_baseline.parquet').to_pylist()
    candidate = pq.read_table(DATA / 'review_sentiment_product_comparison.parquet').to_pylist()
    if len(reviews) != len(baseline) or len(reviews) != len(candidate):
        raise ValueError('Source and prediction row counts differ')
    source_ids = review_ids(reviews)
    pools = collections.defaultdict(list)
    for review, review_id, first, second in zip(reviews, source_ids, baseline, candidate):
        if any(row['review_id'] != review_id or row['parent_asin'] != review['parent_asin'] or
               row['rating'] != review['rating'] for row in (first, second)):
            raise ValueError(f'Prediction/source mismatch: {review_id}')
        if review['rating'] not in (1, 2, 3, 4, 5) or any(row['sentiment'] not in ('negative', 'neutral', 'positive') for row in (first, second)):
            raise ValueError(f'Invalid rating or sentiment: {review_id}')
        disagreed = first['sentiment'] != second['sentiment']
        pools[(int(review['rating']), disagreed)].append({
            'review_id': review_id, 'product_id': review['parent_asin'],
            'rating': int(review['rating']), 'review_text': review['text'],
            'baseline_prediction': first['sentiment'],
            'product_review_prediction': second['sentiment'], 'human_label': '',
        })
    rng = random.Random(SEED)
    selected = []
    seen_content = set()
    for rating in range(1, 6):
        for disagreed, count in ((True, 5), (False, 4)):
            candidates = sorted(pools[(rating, disagreed)], key=lambda row: row['review_id'])
            rng.shuffle(candidates)
            chosen = []
            for row in candidates:
                content_hash = row['review_id'].rsplit('-', 1)[0]
                if content_hash not in seen_content:
                    chosen.append(row)
                    seen_content.add(content_hash)
                if len(chosen) == count:
                    break
            if len(chosen) != count:
                raise ValueError(f'Insufficient distinct reviews for {rating} stars, disagreement={disagreed}')
            selected.extend(chosen)
    rng.shuffle(selected)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('x', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    print(f'Wrote {len(selected)} reviews to {OUTPUT}; seed={SEED}.')


if __name__ == '__main__':
    main()
