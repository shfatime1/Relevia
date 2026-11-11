"""Stream the original Electronics JSONL and select a balanced, reviewed sample."""
import argparse
import collections
import concurrent.futures
import io
import json
import math
import re
import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
BASE = 'https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/'
META = BASE + 'meta_categories/meta_Electronics.jsonl'
REVIEWS = BASE + 'review_categories/Electronics.jsonl'
# Exclusions are applied to the source taxonomy and title, not used as categories.
EXCLUDE = re.compile(r'\b(warrant(?:y|ies)|protection plan|service plan|gift card|subscription|software|digital download|renewed|refurbished|replacement|repair|case|cases|cover|covers|skin|skins|decal|decals|sticker|stickers|screen protector|mount|mounts|stand|stands|cable|cables|adapter|adapters|charger|chargers|accessor(?:y|ies)|parts|batter(?:y|ies)|bundle|bundles|kit|kits)\b', re.I)
AMBIGUOUS = re.compile(r'\b(holiday|store|earpads?|cord management|extension cords?|screen protectors?|sleeves?|clips?|wristbands?|styluses|rings|simulated|all)\b', re.I)
PROMO = re.compile(r'\b(warrant(?:y|ies)|protection plan|service plan|gift card|subscription|digital download|renewed|refurbished|replacement|repair|for parts|compatible with|case for|cover for|bundle|bundles)\b', re.I)
NON_ELECTRONIC = re.compile(r'\b(tripods?|monopods?|binoculars?|scopes?|lenses|magnifiers?|film photography|lighting & studio|microscopes?|slides?|backdrops?|photo booth|photo.?booth|fabric|decals?|stickers?|skins?|screen protectors?|sleeves?|covers?|cases?|cords?|cables?|plug adapters?)\b', re.I)
SPEC = re.compile(r'\b(wireless|bluetooth|wi-?fi|usb|hdmi|digital|analog|optical|audio|video|resolution|memory|storage|sensor|frequency|power|voltage|watts?|stereo|waterproof|processor|display|screen|camera|speaker|headphone|router|drive|battery|channel|pixels?|charging|connectivity)\b|\d+(?:\.\d+)?\s*(?:gb|tb|mb|mhz|ghz|khz|hz|mp|mm|cm|inch(?:es)?|volts?|amps?|w(?:atts?)?|mah)\b', re.I)

# Group the actual source taxonomy branches; unknown branches are left for review.
CATEGORY_GROUPS = {
    'Computers & Components': ('Computers & Tablets', 'Computer Components', 'Gaming Computers', 'Computers', 'Monitors', 'Printers & Scanners'),
    'Data Storage': ('Data Storage',),
    'Networking': ('Networking Products',),
    'Headphones & Earbuds': ('Headphones & Earbuds', 'Amplifiers'),
    'Speakers & Home Audio': ('Speakers', 'Portable Speakers & Docks', 'Home Theater', 'Stereo System Components', 'Compact Radios & Stereos', 'Boomboxes'),
    'Portable Audio & Radios': ('Radios', 'CB & Two-Way Radios', 'Digital Voice Recorders', 'MP3 & MP4 Players', 'Portable CD Players'),
    'TV & Video Players': ('Televisions', 'TV-DVD Combos', 'TV on the Go', 'Portable & Handheld TVs', 'DVD Players & Recorders', 'DVD-VCR Combos', 'Blu-ray Players & Recorders', 'Portable DVD Players', 'Streaming Media Players', 'AV Receivers & Amplifiers'),
    'Cameras & Photography': ('Digital Cameras', 'Video', 'Underwater Photography', 'Flashes'),
    'Security & Surveillance': ('Video Surveillance', 'Surveillance Video Equipment', 'Home Security Systems', 'Biometrics', 'Motion Detectors', 'Horns & Sirens'),
    'Vehicle Electronics & GPS': ('Car Electronics', 'Marine Electronics', 'GPS Trackers', 'Sports & Handheld GPS', 'Sports & Vehicle GPS', 'Item Finders'),
    'Wearable Technology': ('Smartwatches', 'Glasses'),
    'Power & Lighting': ('Power Converters', 'Power Strips & Surge Protectors', 'Reading Lights'),
}
MAIN_CATEGORY = {category: group for group, categories in CATEGORY_GROUPS.items() for category in categories}


def stream(url):
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                yield json.loads(line)


def eligible(row):
    asin = row.get('parent_asin')
    title = row.get('title')
    cats = row.get('categories') or []
    price = row.get('price')
    features = [x.strip() for x in (row.get('features') or []) if isinstance(x, str) and len(x.strip()) >= 15 and SPEC.search(x)]
    if not isinstance(asin, str) or not re.fullmatch(r'[A-Z0-9]{10}', asin): return None
    if not isinstance(title, str) or len(title.strip()) < 12 or len(title.split()) < 3 or PROMO.search(title) or NON_ELECTRONIC.search(title): return None
    if not isinstance(price, (float, int)) or isinstance(price, bool) or not math.isfinite(price) or price <= 0: return None
    if len(cats) < 3 or cats[0] != 'Electronics' or any(not isinstance(x, str) for x in cats): return None
    # Broad, meaningful source taxonomy branch; reject accessory-oriented paths.
    if EXCLUDE.search(' '.join(cats[2:])) or AMBIGUOUS.search(' '.join(cats[2:])) or NON_ELECTRONIC.search(' '.join(cats[2:])) or not features or not row.get('images'): return None
    category = cats[2].strip()
    if len(category) < 3 or category.lower() in {'other', 'electronics', 'products'}: return None
    urls = []
    for image in row['images']:
        if isinstance(image, dict):
            for key in ('hi_res', 'large'):
                url = image.get(key)
                if isinstance(url, str) and url.startswith('https://m.media-amazon.com/') and url not in urls: urls.append(url)
    if not urls: return None
    return {'parent_asin': asin, 'category': category, 'categories': cats,
            'brand': (row.get('details') or {}).get('Brand') or row.get('store'),
            'title': title.strip(), 'description': row.get('description') or [],
            'technical_features': features, 'price': float(price),
            'average_rating': row.get('average_rating'), 'rating_number': row.get('rating_number'), 'urls': urls}


def image_for(product):
    path = DATA / 'images' / (product['parent_asin'] + '.webp')
    if path.exists():
        try:
            with Image.open(path) as im: im.verify()
            return True
        except Exception: path.unlink()
    for url in product['urls'][:4]:
        try:
            with requests.get(url, stream=True, timeout=(10, 20)) as response:
                response.raise_for_status()
                content = bytearray()
                for block in response.iter_content(65536):
                    content.extend(block)
                    if len(content) > 15_000_000: raise ValueError('image too large')
            with Image.open(io.BytesIO(content)) as im:
                im = ImageOps.exif_transpose(im).convert('RGB')
                im.thumbnail((512, 512), Image.Resampling.LANCZOS)
                im.save(path, 'WEBP', quality=82, method=4)
            return True
        except (requests.RequestException, OSError, ValueError, Image.DecompressionBombError):
            continue
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=1000)
    parser.add_argument('--metadata-limit', type=int, default=200000, help='Number of metadata records to inspect (0 = all)')
    parser.add_argument('--review-limit', type=int, default=0, help='Number of review records to inspect (0 = all)')
    parser.add_argument('--candidates-per-category', type=int, default=500)
    args = parser.parse_args()
    if args.size < 1 or args.candidates_per_category < 1: parser.error('size and candidate cap must be positive')
    DATA.mkdir(exist_ok=True); (DATA / 'images').mkdir(exist_ok=True)
    db = sqlite3.connect(DATA / 'progress.sqlite')
    db.executescript('CREATE TABLE IF NOT EXISTS candidates (asin TEXT PRIMARY KEY, category TEXT, product TEXT); CREATE TABLE IF NOT EXISTS reviews (asin TEXT, review TEXT); CREATE INDEX IF NOT EXISTS reviews_asin ON reviews(asin); CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT);')
    def state(key):
        row = db.execute('SELECT value FROM state WHERE key=?', (key,)).fetchone()
        return row[0] if row else None
    settings = json.dumps([args.metadata_limit, args.review_limit, args.candidates_per_category])
    if state('settings') and state('settings') != settings: raise SystemExit('Settings changed: remove data/progress.sqlite and rerun to start a fresh selection.')
    db.execute('INSERT OR REPLACE INTO state VALUES (?,?)', ('settings', settings)); db.commit()
    if not state('metadata_done'):
        counts = collections.Counter()
        db.execute('DELETE FROM candidates'); db.commit()
        for n, row in enumerate(stream(META), 1):
            p = eligible(row)
            if p and counts[p['category']] < args.candidates_per_category:
                inserted = db.execute('INSERT OR IGNORE INTO candidates VALUES (?,?,?)', (p['parent_asin'], p['category'], json.dumps(p))).rowcount
                counts[p['category']] += inserted
            if n % 10000 == 0: db.commit(); print(f'metadata {n:,}: {sum(counts.values()):,} candidates', flush=True)
            if args.metadata_limit and n >= args.metadata_limit: break
        db.execute('INSERT OR REPLACE INTO state VALUES (?,?)', ('metadata_done', str(n))); db.commit()
        print('Discovered candidate categories:', counts.most_common(), flush=True)
    candidates = {asin for (asin,) in db.execute('SELECT asin FROM candidates')}
    if not candidates: raise SystemExit('No eligible metadata candidates found.')
    if not state('reviews_done'):
        db.execute('DELETE FROM reviews'); db.commit()
        counts = collections.Counter()
        for n, row in enumerate(stream(REVIEWS), 1):
            asin = row.get('parent_asin'); rating = row.get('rating'); text = row.get('text')
            if asin in candidates and counts[asin] < 30 and isinstance(rating, (int, float)) and not isinstance(rating, bool) and 1 <= rating <= 5 and isinstance(text, str) and text.strip():
                db.execute('INSERT INTO reviews VALUES (?,?)', (asin, json.dumps(row))); counts[asin] += 1
            if n % 100000 == 0:
                db.commit(); qualified = sum(v >= 5 for v in counts.values())
                print(f'reviews {n:,}: {qualified:,} candidates with 5+ reviews', flush=True)
                if qualified >= args.size * 2:
                    print('Candidate reserve reached; ending review stream early.', flush=True)
                    break
            if args.review_limit and n >= args.review_limit: break
        db.execute('INSERT OR REPLACE INTO state VALUES (?,?)', ('reviews_done', str(n))); db.commit()
    product_path = DATA / 'products.parquet'
    review_path = DATA / 'reviews.parquet'
    if product_path.exists() != review_path.exists():
        raise SystemExit('Both existing parquet files are required to preserve the previous selection.')
    products = pq.read_table(product_path).to_pylist() if product_path.exists() else []
    reviews = pq.read_table(review_path).to_pylist() if review_path.exists() else []
    if len(products) > args.size:
        raise SystemExit('Requested size is smaller than the existing selection; existing products will not be removed.')
    for p in products:
        original = p.get('original_category') or p['category']
        if original not in MAIN_CATEGORY: raise SystemExit(f'Unmapped existing category: {original}')
        p['original_category'] = original
        p['main_category'] = MAIN_CATEGORY[original]
    existing_ids = {p['parent_asin'] for p in products}
    rows = db.execute('SELECT c.product FROM candidates c JOIN reviews r ON c.asin=r.asin GROUP BY c.asin HAVING count(*) >= 5 ORDER BY c.asin').fetchall()
    groups = collections.defaultdict(list)
    for (raw,) in rows:
        p = json.loads(raw)
        if p['parent_asin'] in existing_ids or p['category'] not in MAIN_CATEGORY: continue
        if NON_ELECTRONIC.search(p['title']) or NON_ELECTRONIC.search(' '.join(p['categories'][2:])): continue
        p['technical_features'] = [feature for feature in p['technical_features'] if SPEC.search(feature)]
        if not p['technical_features']: continue
        groups[p['category']].append(p)
    category_cap = max(1, math.ceil(args.size * 0.10))
    print('Reviewed candidates by category:', {k: len(v) for k, v in groups.items()}, flush=True)
    # Increase the cap only when all remaining categories have reached it.
    selected = []; positions = collections.Counter()
    selected_by_category = collections.Counter(p['original_category'] for p in products)
    while len(products) + len(selected) < args.size:
        active = [c for c in sorted(groups) if positions[c] < len(groups[c])]
        if not active: break
        eligible_categories = [c for c in active if selected_by_category[c] < category_cap]
        if not eligible_categories:
            category_cap = min(selected_by_category[c] for c in active) + 1
            continue
        batch = []
        for category in eligible_categories[:args.size - len(products) - len(selected)]:
            batch.append(groups[category][positions[category]])
            positions[category] += 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(image_for, batch))
        for p, ok in zip(batch, results):
            if ok:
                selected.append(p); selected_by_category[p['category']] += 1
        print(f'Image-eligible products: {len(products) + len(selected):,}/{args.size:,}', flush=True)
    for p in selected:
        rs = [json.loads(x[0]) for x in db.execute('SELECT review FROM reviews WHERE asin=?', (p['parent_asin'],))]
        # First select one review from each observed rating, then fill in source order.
        chosen = []; seen = set()
        for r in rs:
            if r['rating'] not in seen: chosen.append(r); seen.add(r['rating'])
        chosen_ids = {id(r) for r in chosen}
        chosen.extend(r for r in rs if id(r) not in chosen_ids)
        for r in chosen[:10]:
            reviews.append({k: r.get(k) for k in ('parent_asin', 'asin', 'user_id', 'rating', 'title', 'text', 'timestamp', 'helpful_vote', 'verified_purchase')})
        p.pop('urls'); p['image_path'] = 'data/images/' + p['parent_asin'] + '.webp'
        p['original_category'] = p['category']; p['main_category'] = MAIN_CATEGORY[p['category']]
        products.append(p)
    if not products: raise SystemExit('No products passed review and image eligibility; no parquet files written.')
    pq.write_table(pa.Table.from_pylist(products), product_path, compression='zstd')
    pq.write_table(pa.Table.from_pylist(reviews), review_path, compression='zstd')
    print(f'Wrote {len(products)} products and {len(reviews)} reviews. Requested {args.size}.', flush=True)
    db.close()


if __name__ == '__main__': main()
