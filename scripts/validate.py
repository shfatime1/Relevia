"""Report and check the generated Phase 1 artifacts."""
from collections import Counter
from pathlib import Path
import argparse
import math
import json
import re
import sqlite3
import sys

import pyarrow.parquet as pq
from PIL import Image
from prepare import MAIN_CATEGORY, SPEC

root = Path(__file__).resolve().parent.parent
data = root / 'data'
parser = argparse.ArgumentParser()
parser.add_argument('--expected-size', type=int, help='Fail if the product count differs')
args = parser.parse_args()
products = pq.read_table(data / 'products.parquet').to_pylist()
reviews = pq.read_table(data / 'reviews.parquet').to_pylist()
ids = [p['parent_asin'] for p in products]
counts = Counter(r['parent_asin'] for r in reviews)
missing = Counter()
errors = []
if args.expected_size is not None and len(products) != args.expected_size: errors.append('Unexpected product count')
for p in products:
    for key in ('parent_asin', 'category', 'original_category', 'main_category', 'categories', 'brand', 'title', 'description', 'technical_features', 'price', 'average_rating', 'rating_number', 'image_path'):
        value = p.get(key)
        if value is None or value == '' or value == []: missing[key] += 1
    if not re.fullmatch(r'[A-Z0-9]{10}', p['parent_asin']) or len(p['title'].strip()) < 12 or len(p['title'].split()) < 3 or not isinstance(p['price'], (float, int)) or not math.isfinite(p['price']) or p['price'] <= 0 or not p['technical_features'] or not all(SPEC.search(f) for f in p['technical_features']):
        errors.append(f"Invalid price or features: {p['parent_asin']}")
    if p['original_category'] != p['category'] or p['main_category'] != MAIN_CATEGORY.get(p['original_category']): errors.append(f"Invalid category: {p['parent_asin']}")
    if not 5 <= counts[p['parent_asin']] <= 10: errors.append(f"Invalid review count: {p['parent_asin']}")
    try:
        with Image.open(root / p['image_path']) as im:
            im.load()
            if im.format != 'WEBP' or max(im.size) > 512: errors.append(f"Invalid image: {p['parent_asin']}")
    except Exception as exc: errors.append(f"Image error {p['parent_asin']}: {exc}")
if len(ids) != len(set(ids)): errors.append('Duplicate product IDs')
if set(counts) - set(ids): errors.append('Reviews without matching product')
if any(not isinstance(r['rating'], (float, int)) or not 1 <= r['rating'] <= 5 for r in reviews): errors.append('Invalid review rating')
if (data / 'progress.sqlite').exists():
    db = sqlite3.connect(data / 'progress.sqlite')
    fields = ('parent_asin', 'asin', 'user_id', 'rating', 'title', 'text', 'timestamp', 'helpful_vote', 'verified_purchase')
    source = Counter(tuple(json.loads(raw).get(k) for k in fields) for (raw,) in db.execute('SELECT review FROM reviews'))
    output = Counter(tuple(r.get(k) for k in fields) for r in reviews)
    if output - source: errors.append('Reviews not found in cached original source records')
    db.close()
images = list((data / 'images').glob('*.webp'))
if len(images) != len(products): errors.append(f'Image count {len(images)} differs from product count {len(products)}')
print('Products:', len(products), 'Reviews:', len(reviews), 'Images:', len(images))
print('Main-category distribution:', dict(Counter(p['main_category'] for p in products).most_common()))
print('Original-category distribution:', dict(Counter(p['original_category'] for p in products).most_common()))
print('Review counts per product:', dict(sorted(Counter(counts.values()).items())))
print('Review rating distribution:', dict(sorted(Counter(r['rating'] for r in reviews).items())))
print('Missing values:', dict(missing))
print('Disk usage (bytes):', sum(p.stat().st_size for p in data.rglob('*') if p.is_file()))
print('Errors:', errors or 'none')
sys.exit(bool(errors))
