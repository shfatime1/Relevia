"""Add reviewed Electronics peripherals without altering the original selection."""
import collections
import argparse
import concurrent.futures
import json
import math
import os
import re
import shutil
import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from PIL import Image

from prepare import DATA, META, REVIEWS, MAIN_CATEGORY, PROMO, SPEC, image_for

TARGETS = {'mouse': 25, 'keyboard': 25, 'webcam': 15, 'microphone': 10,
           'hub': 15, 'cable_power': 20, 'monitor': 10}
REVIEW_FIELDS = ('parent_asin', 'asin', 'user_id', 'rating', 'title', 'text',
                 'timestamp', 'helpful_vote', 'verified_purchase')
CHUNK = 16 * 1024 * 1024
# Exclude replacement parts, protective products and unrelated peripherals even when
# the source taxonomy is broad or a title mentions another product for compatibility.
REJECT = re.compile(r'\b(replacement|refurbished|renewed|repair|case|cover|skin|protector|protective|sticker|decal|mouse\s*pad|mouse\s*(?:jiggler|mover)|desk\s*mat|keycaps?|key\s*switch(?:es)?|switches\s+for|warranty|service|subscription|bundle|combo|kit|converter\s+for|stand\s+for|[1-9]\d{1,2}[-\s]*pack)\b', re.I)


def product_type(categories, title):
    if len(categories) < 3 or categories[0] != 'Electronics':
        return None
    leaf = categories[-1].casefold()
    branch = categories[2]
    title = title.casefold()
    if branch == 'Monitors' and re.search(r'\bmonitor\b', title) and not re.search(r'\b(replacement|refurbished|renewed|repair|laptop screen|monitor (?:mount|arm|hood|cover|magnifier|sunshade|stand)|baby monitor|2.pack)\b', title):
        return 'monitor'
    if REJECT.search(title): return None
    if branch == 'Computer Accessories & Peripherals':
        if leaf == 'mice' and re.search(r'\b(mouse|mice)\b', title) and not re.search(r'\bkeyboard\b', title): return 'mouse'
        if leaf == 'keyboards' and re.search(r'\bkeyboard\b', title) and not re.search(r'\b(mouse|mice|case|switch|(?:receiver|dongle|adapter)\s+for)\b', title): return 'keyboard'
        if leaf == 'webcams' and re.search(r'\b(web\s*cam|webcam)\b', title) and not re.search(r'\b(speakerphone|blue yeti)\b', title): return 'webcam'
        if leaf in ('computer microphones', 'microphones') and re.search(r'\b(microphone|mic)\b', title) and not re.search(r'\b(car|vehicle|speakerphone|mic.lock|electret|headset|headphones)\b', title): return 'microphone'
        if leaf in ('usb hubs', 'hubs', 'docking stations') and re.search(r'\b(hub|docking station|dock)\b', title) and not re.search(r'\b(hard drive|ssd|sata|dual.bay|drive dock)\b', title): return 'hub'
        if any(c in ('Cables & Interconnects', 'Computer Cable Adapters') for c in categories) and re.search(r'\b(cable|adapter)\b', title): return 'cable_power'
    if branch == 'Networking Products' and leaf == 'hubs' and re.search(r'\busb\b.*\bhub\b|\bhub\b.*\busb\b', title): return 'hub'
    if branch in ('Laptop Accessories', 'Power Accessories') and leaf in ('chargers & adapters', 'ac adapters') and re.search(r'\b(charger|adapter|power supply)\b', title) and not re.search(r'\b(shaver|scooter|vacuum|toothbrush)\b', title): return 'cable_power'
    return None


def model_keys(title):
    """Catch obvious same-brand/model duplicates without merging different products."""
    words = title.casefold().split()
    if not words: return set()
    return {(words[0], re.sub(r'[^a-z0-9]', '', token)) for token in re.findall(r'\b[a-z]{1,5}[- ]?\d{3,5}[a-z]?\b', title.casefold())}


def candidate(row):
    asin, title, cats, price = row.get('parent_asin'), row.get('title'), row.get('categories') or [], row.get('price')
    if not isinstance(asin, str) or not re.fullmatch(r'[A-Z0-9]{10}', asin): return None
    if not isinstance(title, str) or len(title.strip()) < 12 or len(title.split()) < 3 or PROMO.search(title): return None
    if not isinstance(cats, list) or not all(isinstance(c, str) for c in cats): return None
    kind = product_type(cats, title)
    if not kind or cats[2] not in MAIN_CATEGORY: return None
    if not isinstance(price, (float, int)) or isinstance(price, bool) or not math.isfinite(price) or price <= 0: return None
    features = [f.strip() for f in (row.get('features') or []) if isinstance(f, str) and len(f.strip()) >= 15 and SPEC.search(f)]
    if not features: return None
    urls = []
    for image in row.get('images') or []:
        if isinstance(image, dict):
            for key in ('hi_res', 'large'):
                url = image.get(key)
                if isinstance(url, str) and url.startswith('https://m.media-amazon.com/') and url not in urls: urls.append(url)
    if not urls: return None
    return kind, {'parent_asin': asin, 'category': cats[2], 'categories': cats,
                  'brand': (row.get('details') or {}).get('Brand') or row.get('store'),
                  'title': title.strip(), 'description': row.get('description') or [],
                  'technical_features': features, 'price': float(price),
                  'average_rating': row.get('average_rating'), 'rating_number': row.get('rating_number'), 'urls': urls}


def batches(url, offset):
    session = requests.Session()
    while True:
        end = offset + CHUNK - 1
        response = session.get(url, headers={'Range': f'bytes={offset}-{end}'}, timeout=(30, 180))
        response.raise_for_status()
        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
        if response.status_code != 206 or not match or int(match.group(1)) != offset:
            raise RuntimeError(f'Source did not honor byte range at {offset}: {response.status_code} {response.headers.get("Content-Range")}')
        data = response.content
        last = data.rfind(b'\n')
        if last < 0: raise RuntimeError('Source record exceeds batch size')
        yield offset + last + 1, data[:last + 1].splitlines(), int(match.group(3))
        offset += last + 1
        if offset >= int(match.group(3)): return


def main():
    parser = argparse.ArgumentParser(description='Expand with bounded, resumable Electronics source scans.')
    parser.add_argument('--metadata-mib', type=int, default=512, help='Maximum metadata bytes scanned in MiB')
    parser.add_argument('--reviews-mib', type=int, default=4096, help='Maximum review bytes scanned in MiB')
    parser.add_argument('--scan-only', action='store_true', help='Cache source candidates and reviews without changing outputs')
    parser.add_argument('--prepare-only', action='store_true', help='Download candidate images and list selections without changing Parquet files')
    parser.add_argument('--commit-prepared', action='store_true', help='Write the previously prepared selection')
    parser.add_argument('--from-original', action='store_true', help='Restage against the preserved pre-expansion Parquet files')
    args = parser.parse_args()
    if args.metadata_mib < 1 or args.reviews_mib < 1: parser.error('Scan limits must be positive')
    product_path, review_path = DATA / 'products.parquet', DATA / 'reviews.parquet'
    original_products, original_reviews = pq.read_table(product_path), pq.read_table(review_path)
    if args.from_original:
        saved_products = pq.read_table(DATA / 'phase6_original_products.parquet')
        saved_reviews = pq.read_table(DATA / 'phase6_original_reviews.parquet')
        if not original_products.slice(0, saved_products.num_rows).equals(saved_products) or not original_reviews.slice(0, saved_reviews.num_rows).equals(saved_reviews):
            raise ValueError('Current output no longer preserves the original rows')
        original_products, original_reviews = saved_products, saved_reviews
    products, reviews = original_products.to_pylist(), original_reviews.to_pylist()
    existing = {p['parent_asin'] for p in products}
    db = sqlite3.connect(DATA / 'expansion.sqlite')
    db.executescript('CREATE TABLE IF NOT EXISTS candidates (asin TEXT PRIMARY KEY, kind TEXT NOT NULL, product TEXT NOT NULL);'
                     'CREATE TABLE IF NOT EXISTS reviews (asin TEXT NOT NULL, review TEXT NOT NULL, UNIQUE(asin,review));'
                     'CREATE INDEX IF NOT EXISTS review_asin ON reviews(asin);'
                     'CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT);')
    def get(key):
        row = db.execute('SELECT value FROM state WHERE key=?', (key,)).fetchone()
        return row[0] if row else None
    def put(key, value): db.execute('INSERT OR REPLACE INTO state VALUES (?,?)', (key, str(value)))
    old = sqlite3.connect(f'file:{(DATA / "progress.sqlite").as_posix()}?mode=ro', uri=True)
    for asin, raw in old.execute('SELECT asin,product FROM candidates'):
        if asin in existing: continue
        p = json.loads(raw)
        kind = product_type(p['categories'], p['title'])
        if kind:
            p['technical_features'] = [f for f in p['technical_features'] if SPEC.search(f)]
            if not p['technical_features']: continue
            db.execute('INSERT OR REPLACE INTO candidates VALUES (?,?,?)', (asin, kind, json.dumps(p)))
            for (review,) in old.execute('SELECT review FROM reviews WHERE asin=?', (asin,)):
                db.execute('INSERT OR IGNORE INTO reviews VALUES (?,?)', (asin, review))
    old.close(); put('seeded', 1); db.commit()
    for asin, kind, raw in db.execute('SELECT asin,kind,product FROM candidates').fetchall():
        product = json.loads(raw)
        product['technical_features'] = [f for f in product['technical_features'] if SPEC.search(f)]
        if not product['technical_features'] or product_type(product['categories'], product['title']) != kind:
            db.execute('DELETE FROM reviews WHERE asin=?', (asin,))
            db.execute('DELETE FROM candidates WHERE asin=?', (asin,))
        else:
            db.execute('UPDATE candidates SET product=? WHERE asin=?', (json.dumps(product), asin))
    db.commit()
    counts = collections.Counter(dict(db.execute('SELECT kind,count(*) FROM candidates GROUP BY kind')))
    metadata_start = int(get('metadata_offset') or 0)
    if not get('metadata_done') and metadata_start < args.metadata_mib * 1024 * 1024:
        for offset, lines, total in batches(META, metadata_start):
            for line in lines:
                found = candidate(json.loads(line))
                if not found: continue
                kind, product = found
                if product['parent_asin'] in existing or counts[kind] >= TARGETS[kind] * 6: continue
                inserted = db.execute('INSERT OR IGNORE INTO candidates VALUES (?,?,?)', (product['parent_asin'], kind, json.dumps(product))).rowcount
                counts[kind] += inserted
            put('metadata_offset', offset)
            if all(counts[k] >= v * 6 for k, v in TARGETS.items()) or offset >= total: put('metadata_done', 1)
            db.commit()
            print(f'Metadata {offset:,}/{total:,} bytes; candidates {dict(counts)}', flush=True)
            if get('metadata_done') or offset >= args.metadata_mib * 1024 * 1024: break
    if int(get('metadata_offset') or 0) > metadata_start and int(get('reviews_offset') or 0):
        print('Candidate pool grew; review stream must restart to include earlier matching reviews.', flush=True)
        put('reviews_offset', 0); db.execute("DELETE FROM state WHERE key='reviews_done'"); db.commit()
    candidates = {asin: kind for asin, kind in db.execute('SELECT asin,kind FROM candidates')}
    review_counts = collections.Counter(dict(db.execute('SELECT asin,count(*) FROM reviews GROUP BY asin')))
    def qualified(): return collections.Counter(candidates[asin] for asin, count in review_counts.items() if count >= 5 and asin in candidates)
    if not get('reviews_done') and int(get('reviews_offset') or 0) < args.reviews_mib * 1024 * 1024:
        for offset, lines, total in batches(REVIEWS, int(get('reviews_offset') or 0)):
            for line in lines:
                row = json.loads(line); asin = row.get('parent_asin'); rating = row.get('rating'); text = row.get('text')
                if asin in candidates and review_counts[asin] < 30 and isinstance(rating, (int, float)) and not isinstance(rating, bool) and 1 <= rating <= 5 and isinstance(text, str) and text.strip():
                    inserted = db.execute('INSERT OR IGNORE INTO reviews VALUES (?,?)', (asin, json.dumps(row))).rowcount
                    review_counts[asin] += inserted
            put('reviews_offset', offset)
            q = qualified()
            if all(q[k] >= v * 2 for k, v in TARGETS.items()) or offset >= total: put('reviews_done', 1)
            db.commit()
            print(f'Reviews {offset:,}/{total:,} bytes; qualified {dict(q)}', flush=True)
            if get('reviews_done') or offset >= args.reviews_mib * 1024 * 1024: break
    if args.scan_only:
        print('Scan only; datasets and images unchanged.', flush=True)
        db.close()
        return
    selected = []
    already_added = collections.Counter(kind for asin, kind in candidates.items() if asin in existing)
    seen_titles = {' '.join(p['title'].casefold().split()) for p in products}
    seen_models = set().union(*(model_keys(p['title']) for p in products))
    if args.commit_prepared:
        for entry in json.loads((DATA / 'expansion_selection.json').read_text(encoding='utf-8')):
            p, kind = entry['product'], entry['kind']
            asin = p['parent_asin']
            cached = db.execute('SELECT product FROM candidates WHERE asin=?', (asin,)).fetchone()
            if asin in existing or not cached or json.loads(cached[0]) != p or candidates.get(asin) != kind or review_counts[asin] < 5 or product_type(p['categories'], p['title']) != kind:
                raise ValueError(f'Prepared candidate is no longer eligible: {asin}')
            selected.append((p, kind))
    for kind, target in ([] if args.commit_prepared else TARGETS.items()):
        options = []
        for asin, raw in db.execute('SELECT asin,product FROM candidates WHERE kind=? ORDER BY asin', (kind,)):
            if asin in existing or review_counts[asin] < 5: continue
            p = json.loads(raw); title = ' '.join(p['title'].casefold().split())
            models = model_keys(p['title'])
            if title not in seen_titles and not models.intersection(seen_models):
                options.append((p, asin)); seen_titles.add(title); seen_models.update(models)
        start = 0
        while start < len(options):
            remaining = target - already_added[kind] - sum(1 for _, k in selected if k == kind)
            if remaining <= 0: break
            batch = options[start:start + min(remaining, 12)]
            start += len(batch)
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                valid = list(pool.map(lambda pair: image_for(pair[0]), batch))
            selected.extend((p, kind) for (p, _), ok in zip(batch, valid) if ok)
            for (p, _), ok in zip(batch, valid):
                if not ok: (DATA / 'images' / f"{p['parent_asin']}.webp").unlink(missing_ok=True)
        print(f'Selected {kind}: {already_added[kind] + sum(1 for _, k in selected if k == kind)}/{target}', flush=True)
    if args.prepare_only:
        output = DATA / 'expansion_selection.json'
        output.write_text(json.dumps([{'kind': kind, 'product': p, 'review_count': review_counts[p['parent_asin']]} for p, kind in selected], ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Prepared {len(selected)} products; review selection in {output}. Parquet files unchanged.', flush=True)
        db.close()
        return
    for p, kind in selected:
        rows = [json.loads(raw) for (raw,) in db.execute('SELECT review FROM reviews WHERE asin=? ORDER BY rowid', (p['parent_asin'],))]
        if len(rows) < 5 or any(row.get('parent_asin') != p['parent_asin'] for row in rows):
            raise ValueError(f"Invalid matching reviews: {p['parent_asin']}")
        chosen, seen = [], set()
        for row in rows:
            if row['rating'] not in seen: chosen.append(row); seen.add(row['rating'])
        chosen_ids = {id(row) for row in chosen}
        chosen.extend(row for row in rows if id(row) not in chosen_ids)
        reviews.extend({key: row.get(key) for key in REVIEW_FIELDS} for row in chosen[:10])
        p.pop('urls'); p['image_path'] = f"data/images/{p['parent_asin']}.webp"
        p['original_category'] = p['category']; p['main_category'] = MAIN_CATEGORY[p['category']]
        products.append(p)
    if not selected: print('No new eligible products; output unchanged.'); return
    # Stage using the original Arrow schemas, and verify every original row before replacement.
    staged_products = DATA / 'expanded_products.tmp.parquet'; staged_reviews = DATA / 'expanded_reviews.tmp.parquet'
    pq.write_table(pa.Table.from_pylist(products, schema=original_products.schema), staged_products, compression='zstd')
    pq.write_table(pa.Table.from_pylist(reviews, schema=original_reviews.schema), staged_reviews, compression='zstd')
    assert pq.read_table(staged_products).slice(0, original_products.num_rows).equals(original_products)
    assert pq.read_table(staged_reviews).slice(0, original_reviews.num_rows).equals(original_reviews)
    for p, _ in selected:
        with Image.open(DATA.parent / p['image_path']) as image:
            image.load()
            if image.format != 'WEBP' or max(image.size) > 512: raise ValueError(f"Invalid image: {p['parent_asin']}")
    for path in (product_path, review_path):
        backup = DATA / f'phase6_original_{path.name}'
        if not backup.exists(): shutil.copy2(path, backup)
        shutil.copy2(path, DATA / f'phase6_before_{path.name}')
    try:
        os.replace(staged_products, product_path); os.replace(staged_reviews, review_path)
    except Exception:
        shutil.copy2(DATA / f'phase6_before_{product_path.name}', product_path)
        shutil.copy2(DATA / f'phase6_before_{review_path.name}', review_path)
        raise
    for path in (product_path, review_path):
        (DATA / f'phase6_before_{path.name}').unlink(missing_ok=True)
    print(f'Preserved {original_products.num_rows} products and {original_reviews.num_rows} reviews; added {len(selected)} products and {len(reviews) - original_reviews.num_rows} reviews.', flush=True)
    db.close()


if __name__ == '__main__': main()
