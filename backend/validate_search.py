"""Check the cached index and show representative retrieval results and timings."""
import json
import time
from pathlib import Path

import numpy as np

from backend.search import INDEX_DIR, ProductSearch


def main():
    search = ProductSearch()
    assert len(search.ids) == len(set(search.ids)) == search.index.ntotal == 1000
    assert search.load_index()
    for key in ('dataset_sha256', 'revision'):
        original = search.fingerprint[key]
        search.fingerprint[key] = 'changed'
        assert not search.load_index(), f'Changed {key} did not invalidate cache'
        search.fingerprint[key] = original
    assert search.load_index()
    vectors = np.load(INDEX_DIR / 'embeddings.npy', allow_pickle=False)
    assert vectors.shape == (1000, 384)
    assert np.isfinite(vectors).all() and np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    for i in (0, 499, 999):
        assert np.allclose(search.index.reconstruct(i), vectors[i], atol=1e-6)
    by_id = {product['parent_asin']: product for product in search.products}
    print(f'Index: {search.index.ntotal} vectors x {search.index.d}; cache reused: {search.cache_hit}; setup: {search.setup_seconds:.2f}s')
    queries = ('Bose QuietComfort', 'wireless headphones with noise cancellation',
               'simsiz bluetooth qulaqlıq', 'ev üçün təhlükəsizlik kamerası')
    search.semantic('warm up', 1)
    durations = {'BM25': [], 'semantic': []}
    for query in queries:
        print('\nQUERY:', query)
        for name, method in (('BM25', search.keyword), ('semantic', search.semantic)):
            started = time.perf_counter()
            results = method(query, 5)
            durations[name].append((time.perf_counter() - started) * 1000)
            if query == 'Bose QuietComfort':
                assert results and results[0]['parent_asin'] == 'B0BL5YHZPC'
            if name == 'semantic' and query == 'wireless headphones with noise cancellation':
                assert results[0]['main_category'] == 'Headphones & Earbuds'
            if name == 'semantic' and query == 'ev üçün təhlükəsizlik kamerası':
                assert results[0]['main_category'] == 'Security & Surveillance'
            for result in results:
                original = by_id[result['parent_asin']]
                assert all(result[key] == original[key] for key in ('title', 'price', 'main_category', 'image_path'))
                assert (Path(__file__).resolve().parent.parent / result['image_path']).is_file()
            print(name, json.dumps(results, ensure_ascii=False))
    print('\nAverage warm query latency (ms):', {name: round(sum(v) / len(v), 2) for name, v in durations.items()})
    print('Checks passed: vector dimensions and norms, FAISS contents, IDs, result metadata and image paths.')


if __name__ == '__main__':
    main()
