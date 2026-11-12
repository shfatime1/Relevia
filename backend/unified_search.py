"""Unified hybrid text and CLIP image search with catalog filters."""
import argparse
import json
import math
import sys

class UnifiedSearch:
    def __init__(self):
        self.text_search = None
        self.image_search = None

    @staticmethod
    def filter_results(results, category=None, min_price=None, max_price=None, limit=5):
        if limit < 1:
            raise ValueError('Limit must be positive')
        if category is not None and not category.strip():
            raise ValueError('Category must not be empty')
        for price in (min_price, max_price):
            if price is not None and (not math.isfinite(price) or price < 0):
                raise ValueError('Price bounds must be finite and nonnegative')
        if min_price is not None and max_price is not None and min_price > max_price:
            raise ValueError('Minimum price cannot exceed maximum price')
        category = category.strip().casefold() if category is not None else None
        filtered = []
        for result in results:
            product = result['product']
            if category and category not in (str(product.get('main_category') or '').casefold(),
                                             str(product.get('original_category') or '').casefold()):
                continue
            price = product.get('price')
            if (min_price is not None or max_price is not None) and (price is None or not math.isfinite(price)):
                continue
            if min_price is not None and price < min_price or max_price is not None and price > max_price:
                continue
            filtered.append({
                'product_id': product['parent_asin'], 'title': product['title'],
                'price': price, 'category': product['main_category'],
                'original_category': product['original_category'], 'image_path': product['image_path'],
                'relevance_score': result['score'], 'ranking_reasons': result['reasons'],
            })
            if len(filtered) == limit:
                break
        return filtered

    def text(self, query, category=None, min_price=None, max_price=None, limit=5):
        if not query.strip():
            raise ValueError('Text query must not be empty')
        if self.text_search is None:
            from backend.search import ProductSearch
            self.text_search = ProductSearch()
        search = self.text_search
        products = {product['parent_asin']: product for product in search.products}
        lexical = search.keyword(query, len(products))
        semantic = search.semantic(query, len(products))
        keyword_ranks = {result['parent_asin']: rank for rank, result in enumerate(lexical, 1)}
        results = []
        for rank, result in enumerate(semantic, 1):
            product_id = result['parent_asin']
            keyword_rank = keyword_ranks.get(product_id)
            # Equal-weight reciprocal rank fusion; missing keyword matches contribute zero.
            score = 61 / 2 * (1 / (60 + rank) + (1 / (60 + keyword_rank) if keyword_rank else 0))
            reasons = [f'E5 semantic rank {rank}; cosine {result["score"]:.4f}',
                       f'BM25 keyword rank {keyword_rank}' if keyword_rank else 'No BM25 token match']
            results.append({'product': products[product_id], 'score': score, 'reasons': reasons})
        results.sort(key=lambda result: (-result['score'], result['product']['parent_asin']))
        return self.filter_results(results, category, min_price, max_price, limit)

    def image(self, path, category=None, min_price=None, max_price=None, limit=5):
        if self.image_search is None:
            from backend.image_search import ImageSearch
            self.image_search = ImageSearch()
        search = self.image_search
        products = {product['parent_asin']: product for product in search.products}
        matches = search.search(path, len(products))
        results = [{'product': products[match['parent_asin']], 'score': match['score'],
                    'reasons': [f'CLIP image cosine similarity {match["score"]:.4f}']}
                   for match in matches]
        return self.filter_results(results, category, min_price, max_price, limit)


def main():
    parser = argparse.ArgumentParser(description='Search products by text or image.')
    query = parser.add_mutually_exclusive_group(required=True)
    query.add_argument('--text', help='English or Azerbaijani text query')
    query.add_argument('--image', help='Path to a query image')
    parser.add_argument('--category', help='Exact main or original category, case-insensitive')
    parser.add_argument('--min-price', type=float, help='Inclusive minimum price')
    parser.add_argument('--max-price', type=float, help='Inclusive maximum price')
    parser.add_argument('--top', type=int, default=5, help='Maximum number of results')
    args = parser.parse_args()
    # Validate filters before loading models and indexes.
    try:
        UnifiedSearch.filter_results([], args.category, args.min_price, args.max_price, args.top)
        search = UnifiedSearch()
        options = dict(category=args.category, min_price=args.min_price, max_price=args.max_price, limit=args.top)
        results = search.text(args.text, **options) if args.text is not None else search.image(args.image, **options)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'{exc}\n')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    print(json.dumps({'mode': 'text' if args.text is not None else 'image', 'results': results},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
