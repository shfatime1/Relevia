"""Gradio Space adapter for the existing Relevia application."""
import asyncio
import json
import os
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from threading import Event, Thread
from unittest.mock import patch

import spaces
import gradio as gr
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend import api, image_search, search

ROOT = Path(__file__).resolve().parent
DIST = ROOT / 'dist'

# An optional local path reuses existing model weights without copying them.
if model_cache := os.environ.get('RELEVIA_MODEL_CACHE_ROOT'):
    search.CACHE = image_search.CACHE = Path(model_cache).resolve()


def missing_text_cache(self):
    raise RuntimeError('Text index cache is missing or invalid; rebuilding is disabled in the Space')


def load_image_cache(self):
    """Load only the exact image index; never encode or rewrite a cache."""
    try:
        directory = image_search.INDEX_DIR
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        vectors_path, index_path = directory / 'embeddings.npy', directory / 'products.faiss'
        if (manifest['config'] != image_search.CONFIG or manifest['entries'] != self.entries
                or image_search.digest(vectors_path) != manifest['vectors_sha256']
                or image_search.digest(index_path) != manifest['index_sha256']):
            raise ValueError('cache fingerprint mismatch')
        vectors = image_search.np.load(vectors_path, allow_pickle=False)
        index = image_search.faiss.read_index(str(index_path))
        if (vectors.shape != (len(self.entries), image_search.CONFIG['dimension'])
                or not image_search.np.isfinite(vectors).all()
                or index.d != image_search.CONFIG['dimension'] or index.ntotal != len(self.entries)):
            raise ValueError('invalid cache dimensions or vectors')
        self.index = index
        self.cache_hit = True
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        raise RuntimeError('Image index cache is missing or invalid; rebuilding is disabled in the Space') from exc


search.ProductSearch.build_index = missing_text_cache
image_search.ImageSearch.load_or_build = load_image_cache


@asynccontextmanager
async def lifespan(app):
    # ImageSearch otherwise auto-selects CUDA during startup. E5 already uses CPU.
    async with AsyncExitStack() as stack:
        with patch.object(image_search.torch.cuda, 'is_available', return_value=False):
            await stack.enter_async_context(api.lifespan(app))
        yield


app = gr.Server()
app.include_router(api.app.router)
app.mount('/images', StaticFiles(directory=ROOT / 'data' / 'images'), name='images')


@app.api(name='gpu_check')
@spaces.GPU(duration=10)
def gpu_check() -> float:
    """Run a small calculation on the allocated GPU when explicitly called."""
    import torch
    return (torch.ones(1, device='cuda') + 1).item()


@app.get('/api/health')
def health():
    return {'status': 'ok'}


@app.get('/')
def index():
    return FileResponse(DIST / 'index.html')


app.mount('/assets', StaticFiles(directory=DIST / 'assets'), name='assets')

if __name__ == '__main__':
    # Gradio's five-second server-start timer cannot encompass model loading.
    # Keep Relevia's lifespan open on one thread before starting the server.
    ready, stop = Event(), Event()
    errors = []

    def initialize():
        async def run():
            async with lifespan(app):
                ready.set()
                stop.wait()
        try:
            asyncio.run(run())
        except BaseException as exc:
            errors.append(exc)
            ready.set()

    worker = Thread(target=initialize, daemon=True)
    worker.start()
    try:
        ready.wait()
        if errors:
            raise errors[0]
        app.launch(server_name='0.0.0.0', ssr_mode=False)
    finally:
        stop.set()
        worker.join()
