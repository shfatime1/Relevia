"""Local BM25 and multilingual E5 product search."""
import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / '.cache'
os.environ.setdefault('HF_HOME', str(CACHE / 'huggingface'))

import faiss
import numpy as np
import pyarrow.parquet as pq
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DATA = ROOT / 'data' / 'products.parquet'
INDEX_DIR = CACHE / 'text_search'
MODEL = 'intfloat/multilingual-e5-small'
REVISION = '614241f622f53c4eeff9890bdc4f31cfecc418b3'
MAX_LENGTH = 256
TEXT_VERSION = 1


def product_text(product):
    def values(value):
        if isinstance(value, list):
            return ' '.join(str(item).strip() for item in value if item)
        return str(value).strip() if value is not None else ''

    return '\n'.join(filter(None, (values(product.get(key)) for key in (
        'title', 'brand', 'main_category', 'original_category', 'categories',
        'technical_features', 'description'))))


def tokenize(text):
    return re.findall(r'[^\W_]+', text.casefold(), flags=re.UNICODE)


def digest_file(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class ProductSearch:
    def __init__(self, batch_size=16):
        started = time.perf_counter()
        self.products = pq.read_table(DATA).to_pylist()
        self.ids = [p['parent_asin'] for p in self.products]
        if not self.ids or any(not isinstance(asin, str) or not asin for asin in self.ids) or len(set(self.ids)) != len(self.ids):
            raise ValueError('Products must have unique non-empty parent_asin IDs')
        self.texts = [product_text(p) for p in self.products]
        self.tokens = [tokenize(text) for text in self.texts]
        self.bm25 = BM25Okapi(self.tokens)
        self.model = None
        self.batch_size = batch_size
        self.fingerprint = {
            'dataset_sha256': digest_file(DATA), 'model': MODEL, 'revision': REVISION,
            'max_length': MAX_LENGTH, 'text_version': TEXT_VERSION,
            'document_prefix': 'passage: ', 'query_prefix': 'query: ',
            'normalize': True, 'ids': self.ids,
        }
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        self.cache_hit = self.load_index()
        if not self.cache_hit:
            self.build_index()
        self.setup_seconds = time.perf_counter() - started

    def get_model(self):
        if self.model is None:
            # Keep inference on CPU and avoid excessive thread oversubscription.
            torch.set_num_threads(min(4, os.cpu_count() or 1))
            options = dict(device='cpu', revision=REVISION, cache_folder=str(CACHE / 'huggingface'))
            try:
                self.model = SentenceTransformer(MODEL, local_files_only=True, **options)
            except OSError:
                self.model = SentenceTransformer(MODEL, **options)
            self.model.max_seq_length = MAX_LENGTH
        return self.model

    def load_index(self):
        try:
            manifest = json.loads((INDEX_DIR / 'manifest.json').read_text(encoding='utf-8'))
            if manifest['fingerprint'] != self.fingerprint:
                return False
            if any(digest_file(INDEX_DIR / name) != manifest['sha256'][name] for name in ('embeddings.npy', 'products.faiss')):
                return False
            vectors = np.load(INDEX_DIR / 'embeddings.npy', allow_pickle=False)
            index = faiss.read_index(str(INDEX_DIR / 'products.faiss'))
            if vectors.shape != (len(self.ids), 384) or index.ntotal != len(self.ids) or index.d != vectors.shape[1]:
                return False
            self.index = index
            return True
        except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError):
            return False

    def build_index(self):
        print(f'Building E5 embeddings for {len(self.ids)} products on CPU...', flush=True)
        vectors = self.get_model().encode(
            ['passage: ' + text for text in self.texts], batch_size=self.batch_size,
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True,
        ).astype('float32')
        if vectors.shape != (len(self.ids), 384) or not np.isfinite(vectors).all():
            raise ValueError(f'Invalid embeddings: {vectors.shape}')
        vectors = np.ascontiguousarray(vectors)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        # Manifest is written last; incomplete caches are rebuilt on the next run.
        np.save(INDEX_DIR / 'embeddings.npy', vectors, allow_pickle=False)
        faiss.write_index(self.index, str(INDEX_DIR / 'products.faiss'))
        manifest = {'fingerprint': self.fingerprint, 'sha256': {
            name: digest_file(INDEX_DIR / name) for name in ('embeddings.npy', 'products.faiss')}}
        (INDEX_DIR / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False), encoding='utf-8')

    def result(self, indices, scores):
        fields = ('parent_asin', 'title', 'price', 'main_category', 'image_path')
        return [{**{key: self.products[int(i)].get(key) for key in fields}, 'score': float(score)}
                for i, score in zip(indices, scores)]

    def keyword(self, query, limit=5):
        terms = tokenize(query)
        if not terms:
            return []
        scores = self.bm25.get_scores(terms)
        # Do not return unrelated zero-match products for unmatched queries.
        words = set(terms)
        matches = [i for i, tokens in enumerate(self.tokens) if words.intersection(tokens)]
        indices = sorted(matches, key=lambda i: (-scores[i], i))[:limit]
        return self.result(indices, [scores[i] for i in indices])

    def semantic(self, query, limit=5):
        if not query.strip():
            return []
        vector = self.get_model().encode(['query: ' + query], normalize_embeddings=True,
                                         convert_to_numpy=True).astype('float32')
        scores, indices = self.index.search(np.ascontiguousarray(vector), min(limit, len(self.ids)))
        return self.result(indices[0], scores[0])


def main():
    parser = argparse.ArgumentParser(description='Compare local BM25 and multilingual E5 product search.')
    parser.add_argument('query', help='Search query (English or Azerbaijani)')
    parser.add_argument('--top', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.top < 1 or args.batch_size < 1 or not args.query.strip():
        parser.error('Query must not be empty; top and batch size must be positive.')
    search = ProductSearch(batch_size=args.batch_size)
    print(f'Index: {search.index.ntotal} products, {search.index.d} dimensions; '
          f'{"reused" if search.cache_hit else "built"} in {search.setup_seconds:.2f}s')
    for label, method in (('BM25', search.keyword), ('Semantic', search.semantic)):
        started = time.perf_counter()
        results = method(args.query, args.top)
        print(f'\n{label} ({(time.perf_counter() - started) * 1000:.1f} ms)')
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
