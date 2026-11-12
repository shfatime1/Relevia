"""Content-based recommendations using the existing text, image and sentiment data."""
import argparse
import json
import sys

import numpy as np
import pyarrow.parquet as pq

from backend.image_search import ImageSearch
from backend.search import ProductSearch, ROOT


class Recommender:
    def __init__(self, text_search=None, image_search=None):
        # These loaders validate their caches and rebuild them only when necessary.
        self.text = text_search if text_search is not None else ProductSearch()
        self.image = image_search if image_search is not None else ImageSearch()
        self.products = self.text.products
        self.ids = self.text.ids
        if self.ids != [p['parent_asin'] for p in self.image.products]:
            raise ValueError('Text and image product mappings do not match')
        self.positions = {product_id: i for i, product_id in enumerate(self.ids)}
        summaries = pq.read_table(ROOT / 'data' / 'product_sentiment.parquet').to_pylist()
        self.sentiment = {row['parent_asin']: row for row in summaries}
        if len(summaries) != len(self.sentiment) or set(self.sentiment) != set(self.ids):
            raise ValueError('Sentiment summaries must contain exactly one row per product')
        for row in summaries:
            counts = [row[f'{label}_count'] for label in ('positive', 'neutral', 'negative')]
            if any(count is None or count < 0 for count in counts) or sum(counts) != row['review_count']:
                raise ValueError(f"Invalid sentiment counts for {row['parent_asin']}")

    @staticmethod
    def similarities(index, position):
        vector = np.ascontiguousarray(index.reconstruct(position).reshape(1, -1))
        scores, positions = index.search(vector, index.ntotal)
        result = np.empty(index.ntotal, dtype=np.float32)
        result[positions[0]] = scores[0]
        return result

    def recommend(self, product_id, limit=5):
        if product_id not in self.positions:
            raise ValueError(f'Unknown product ID: {product_id}')
        if limit < 1:
            raise ValueError('Limit must be positive')
        position = self.positions[product_id]
        selected = self.products[position]
        text_scores = self.similarities(self.text.index, position)
        image_scores = self.similarities(self.image.index, position)
        results = []
        for i, product in enumerate(self.products):
            if i == position or product['main_category'] != selected['main_category']:
                continue
            same_category = product['original_category'] == selected['original_category']
            source = selected.get('categories') or []
            candidate = product.get('categories') or []
            same_subcategory = (same_category and len(source) > 3 and len(candidate) > 3
                                and source[3] == candidate[3])
            category_score = 1.0 if same_subcategory else 0.8 if same_category else 0.4
            summary = self.sentiment[product['parent_asin']]
            adjustment = 0.02 * (summary['positive_count'] - summary['negative_count']) / (summary['review_count'] + 5)
            score = 0.55 * float(text_scores[i]) + 0.30 * float(image_scores[i]) + 0.15 * category_score + adjustment
            category_reason = (f"Same source subcategory: {source[3]}" if same_subcategory else
                               f"Same source category: {product['original_category']}" if same_category else
                               f"Same main category fallback: {product['main_category']}; source category: {product['original_category']}")
            results.append({
                'product_id': product['parent_asin'], 'title': product['title'],
                'category': product['main_category'], 'original_category': product['original_category'],
                'price': product['price'], 'image_path': product['image_path'],
                'similarity_score': round(score, 6),
                'text_similarity': round(float(text_scores[i]), 6),
                'visual_similarity': round(float(image_scores[i]), 6),
                'sentiment_adjustment': round(adjustment, 6),
                'reasons': [category_reason,
                            f"Text similarity {text_scores[i]:.3f}; visual similarity {image_scores[i]:.3f}",
                            f"Review sentiment adjustment {adjustment:+.4f} from {summary['review_count']} reviews"],
            })
        return sorted(results, key=lambda row: (
            row['original_category'] != selected['original_category'],
            -row['similarity_score'], row['product_id']))[:limit]


def main():
    parser = argparse.ArgumentParser(description='Recommend similar products by product ID.')
    parser.add_argument('product_id', help='Product parent_asin')
    parser.add_argument('--top', type=int, default=5, help='Maximum number of recommendations (default: 5)')
    args = parser.parse_args()
    if args.top < 1:
        parser.error('--top must be positive')
    try:
        recommender = Recommender()
        results = recommender.recommend(args.product_id, args.top)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'{exc}\n')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    print(json.dumps({'product_id': args.product_id, 'recommendations': results}, ensure_ascii=False, indent=2))
    if len(results) < args.top:
        print(f'Only {len(results)} eligible matches found in the catalog.', file=sys.stderr)


if __name__ == '__main__':
    main()
