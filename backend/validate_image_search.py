"""Check visual index integrity and inspect representative search results."""
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from backend.image_search import CACHE, INDEX_DIR, ImageSearch


def main():
    search = ImageSearch()
    assert search.cache_hit and search.embedded_count == 0
    assert len(search.products) == len(search.entries) == search.index.ntotal == 1000
    assert len({entry['id'] for entry in search.entries}) == 1000
    vectors = np.load(INDEX_DIR / 'embeddings.npy', allow_pickle=False)
    assert vectors.shape == (1000, 512) and search.index.d == 512
    assert np.isfinite(vectors).all() and np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    assert np.allclose(search.index.reconstruct_n(0, len(vectors)), vectors, atol=1e-6)
    for entry, product in zip(search.entries, search.products):
        assert entry['id'] == product['parent_asin'] and entry['path'] == product['image_path']
    by_id = {p['parent_asin']: p for p in search.products}
    examples = []
    for category in ('Headphones & Earbuds', 'Digital Cameras', 'Data Storage'):
        examples.append(next(p for p in search.products if p['original_category'] == category))
    # Exclude the catalog path itself but do not exclude an external copy.
    first_path = search.paths[search.products.index(examples[0])]
    search.search(first_path, 1)  # load the model before timing
    with tempfile.TemporaryDirectory(dir=CACHE) as folder:
        external = Path(folder) / first_path.name
        shutil.copyfile(first_path, external)
        assert examples[0]['parent_asin'] in [r['parent_asin'] for r in search.search(external, 5)]
    timings = []
    print(f'Index: {search.index.ntotal} vectors x {search.index.d}; cache reused: {search.cache_hit}; device: {search.device}; setup: {search.setup_seconds:.2f}s')
    for product in examples:
        started = time.perf_counter()
        results = search.search(Path(__file__).resolve().parent.parent / product['image_path'], 5)
        timings.append((time.perf_counter() - started) * 1000)
        assert len(results) == 5 and product['parent_asin'] not in [r['parent_asin'] for r in results]
        for result in results:
            original = by_id[result['parent_asin']]
            assert all(result[key] == original[key] for key in ('title', 'price', 'main_category', 'image_path'))
        print('\nQUERY:', product['original_category'], product['parent_asin'], product['title'])
        print(json.dumps(results, ensure_ascii=False))
    print(f'Average warm query latency: {sum(timings) / len(timings):.2f} ms')
    print('Checks passed: normalized vectors, FAISS rows, ordered product IDs, metadata, self exclusion and external image search.')


if __name__ == '__main__':
    main()
