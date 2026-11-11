"""CLIP image-to-image search over local product photographs."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / '.cache'
os.environ.setdefault('HF_HOME', str(CACHE / 'huggingface'))

import faiss
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageOps
from transformers import CLIPImageProcessor, CLIPModel

DATA = ROOT / 'data' / 'products.parquet'
INDEX_DIR = CACHE / 'image_search'
MODEL = 'openai/clip-vit-base-patch32'
REVISION = '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
CONFIG = {'model': MODEL, 'revision': REVISION, 'preprocessing': 'clip-rgb-exif-v1', 'dimension': 512}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_image(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert('RGB')


class ImageSearch:
    def __init__(self, batch_size=16):
        started = time.perf_counter()
        if batch_size < 1:
            raise ValueError('Batch size must be positive')
        self.products = pq.read_table(DATA).to_pylist()
        ids = [p['parent_asin'] for p in self.products]
        if not ids or any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
            raise ValueError('Products must have unique non-empty parent_asin IDs')
        self.paths = [(ROOT / p['image_path']).resolve(strict=True) for p in self.products]
        if len(set(self.paths)) != len(self.paths):
            raise ValueError('Each product must have a unique image path')
        self.entries = [{'id': product['parent_asin'], 'path': product['image_path'], 'sha256': digest(path)}
                        for product, path in zip(self.products, self.paths)]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            torch.set_num_threads(min(4, os.cpu_count() or 1))
        self.batch_size = batch_size
        self.model = None
        self.processor = None
        self.cache_hit = False
        self.embedded_count = 0
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        self.load_or_build()
        self.setup_seconds = time.perf_counter() - started

    def load_model(self):
        if self.model is None:
            options = dict(revision=REVISION, cache_dir=str(CACHE / 'huggingface'))
            try:
                processor = CLIPImageProcessor.from_pretrained(MODEL, local_files_only=True, **options)
                model = CLIPModel.from_pretrained(MODEL, local_files_only=True, **options)
            except OSError:
                processor = CLIPImageProcessor.from_pretrained(MODEL, **options)
                model = CLIPModel.from_pretrained(MODEL, **options)
            self.processor = processor
            self.model = model.to(self.device).eval()

    def encode(self, paths):
        self.load_model()
        images = [read_image(path) for path in paths]
        inputs = self.processor(images=images, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            output = self.model.get_image_features(**inputs)
            features = output if isinstance(output, torch.Tensor) else output.pooler_output
            features = torch.nn.functional.normalize(features.float(), p=2, dim=1)
        vectors = features.cpu().numpy().astype('float32', copy=False)
        if vectors.shape != (len(paths), CONFIG['dimension']) or not np.isfinite(vectors).all():
            raise ValueError(f'Invalid image embeddings: {vectors.shape}')
        return vectors

    def load_or_build(self):
        old = None
        vectors = None
        try:
            old = json.loads((INDEX_DIR / 'manifest.json').read_text(encoding='utf-8'))
            vector_path = INDEX_DIR / 'embeddings.npy'
            if old['config'] == CONFIG and digest(vector_path) == old['vectors_sha256']:
                vectors = np.load(vector_path, allow_pickle=False)
                if vectors.shape != (len(old['entries']), CONFIG['dimension']) or not np.isfinite(vectors).all():
                    vectors = None
            if vectors is not None and old['entries'] == self.entries:
                index_path = INDEX_DIR / 'products.faiss'
                if digest(index_path) == old['index_sha256']:
                    index = faiss.read_index(str(index_path))
                    if index.d == CONFIG['dimension'] and index.ntotal == len(self.entries):
                        self.index = index
                        self.cache_hit = True
                        return
        except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError):
            pass
        reused = {}
        if vectors is not None:
            reused = {(entry['id'], entry['sha256']): vector for entry, vector in zip(old['entries'], vectors)}
        result = np.empty((len(self.entries), CONFIG['dimension']), dtype=np.float32)
        pending = []
        for i, entry in enumerate(self.entries):
            key = (entry['id'], entry['sha256'])
            if key in reused:
                result[i] = reused[key]
            else:
                pending.append(i)
        self.embedded_count = len(pending)
        print(f'Encoding {len(pending)} images on {self.device}; reusing {len(self.entries) - len(pending)} cached vectors.', flush=True)
        for offset in range(0, len(pending), self.batch_size):
            batch = pending[offset:offset + self.batch_size]
            try:
                result[batch] = self.encode([self.paths[i] for i in batch])
            except (OSError, ValueError) as exc:
                raise RuntimeError(f'Failed to decode or encode images: {[str(self.paths[i]) for i in batch]}') from exc
            print(f'Encoded {min(offset + len(batch), len(pending))}/{len(pending)}', flush=True)
        self.index = faiss.IndexFlatIP(CONFIG['dimension'])
        self.index.add(np.ascontiguousarray(result))
        vector_path = INDEX_DIR / 'embeddings.npy'
        index_path = INDEX_DIR / 'products.faiss'
        np.save(vector_path, result, allow_pickle=False)
        faiss.write_index(self.index, str(index_path))
        manifest = {'config': CONFIG, 'entries': self.entries, 'vectors_sha256': digest(vector_path), 'index_sha256': digest(index_path)}
        (INDEX_DIR / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False), encoding='utf-8')

    def search(self, image_path, limit=5):
        if limit < 1:
            raise ValueError('Limit must be positive')
        path = Path(image_path).expanduser().resolve(strict=True)
        # Only the catalog path itself is excluded; identical external files are valid queries.
        excluded = next((i for i, candidate in enumerate(self.paths) if candidate == path), None)
        vector = self.encode([path])
        count = min(len(self.products), limit + (excluded is not None))
        scores, indices = self.index.search(np.ascontiguousarray(vector), count)
        fields = ('parent_asin', 'title', 'price', 'main_category', 'image_path')
        return [{**{key: self.products[int(i)].get(key) for key in fields}, 'score': float(score)}
                for i, score in zip(indices[0], scores[0]) if i != excluded][:limit]


def main():
    parser = argparse.ArgumentParser(description='Find visually similar products from a local image.')
    parser.add_argument('image', help='Path to a local image')
    parser.add_argument('--top', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.top < 1 or args.batch_size < 1:
        parser.error('Top and batch size must be positive')
    search = ImageSearch(batch_size=args.batch_size)
    print(f'Index: {search.index.ntotal} products, {search.index.d} dimensions; '
          f'{"reused" if search.cache_hit else "built"} in {search.setup_seconds:.2f}s; device: {search.device}')
    started = time.perf_counter()
    result = search.search(args.image, args.top)
    print(f'Search: {(time.perf_counter() - started) * 1000:.1f} ms')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
