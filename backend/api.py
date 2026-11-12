"""Local REST API for the Relevia catalog and existing search models."""
from collections import defaultdict
from contextlib import asynccontextmanager
import math
import mimetypes
from pathlib import Path
import tempfile
from threading import Lock
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
import pyarrow.parquet as pq

from backend.image_search import ImageSearch
from backend.recommend import Recommender
from backend.search import ProductSearch, ROOT
from backend.sentiment import review_ids
from backend.unified_search import UnifiedSearch

DATA = ROOT / 'data'
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load and validate the existing indexes once per process, then warm both models.
    text = ProductSearch()
    image = ImageSearch()
    recommender = Recommender(text, image)
    unified = UnifiedSearch()
    unified.text_search, unified.image_search = text, image
    text.get_model()
    image.load_model()
    products = {row['parent_asin']: row for row in text.products}
    reviews = pq.read_table(DATA / 'reviews.parquet').to_pylist()
    predictions = pq.read_table(DATA / 'review_sentiment.parquet').to_pylist()
    predictions_by_id = {row['review_id']: row for row in predictions}
    ids = review_ids(reviews)
    if len(predictions) != len(predictions_by_id) or set(predictions_by_id) != set(ids):
        raise ValueError('Review predictions do not match the source reviews')
    reviews_by_product = defaultdict(list)
    for review_id, review in zip(ids, reviews):
        prediction = predictions_by_id[review_id]
        if (review['parent_asin'] not in products or prediction['parent_asin'] != review['parent_asin']
                or prediction['rating'] != review['rating']):
            raise ValueError(f'Review prediction mismatch: {review_id}')
        reviews_by_product[review['parent_asin']].append({
            'review_id': review_id, 'title': review['title'], 'text': review['text'],
            'rating': review['rating'], 'timestamp': review['timestamp'],
            'helpful_vote': review['helpful_vote'], 'verified_purchase': review['verified_purchase'],
            'sentiment': prediction['sentiment'], 'confidence': prediction['confidence'],
        })
    for product_id, summary in recommender.sentiment.items():
        product_reviews = reviews_by_product[product_id]
        if (len(product_reviews) != summary['review_count'] or any(
                sum(review['sentiment'] == label for review in product_reviews) != summary[f'{label}_count']
                for label in ('positive', 'negative', 'neutral'))):
            raise ValueError(f'Review sentiment summary mismatch: {product_id}')
    app.state.products = products
    app.state.reviews = reviews_by_product
    app.state.summaries = recommender.sentiment
    app.state.recommender = recommender
    app.state.search = unified
    app.state.text_lock, app.state.image_lock = Lock(), Lock()
    yield


app = FastAPI(title='Relevia API', version='0.5.0', lifespan=lifespan)
mimetypes.add_type('image/webp', '.webp')
app.mount('/images', StaticFiles(directory=DATA / 'images'), name='images')


def image_url(path):
    return f'/images/{Path(path).name}' if path else None


def product_card(product):
    return {
        'product_id': product['parent_asin'], 'title': product['title'], 'brand': product['brand'],
        'category': product['main_category'], 'original_category': product['original_category'],
        'price': product['price'], 'rating': product['average_rating'],
        'rating_count': product['rating_number'], 'image_url': image_url(product['image_path']),
    }


def product_or_404(request, product_id):
    product = request.app.state.products.get(product_id)
    if product is None:
        raise HTTPException(status_code=404, detail='Product not found')
    return product


def check_filters(category, min_price, max_price):
    if category is not None and not category.strip():
        raise HTTPException(status_code=422, detail='Category must not be empty')
    if any(value is not None and (not math.isfinite(value) or value < 0) for value in (min_price, max_price)):
        raise HTTPException(status_code=422, detail='Price bounds must be finite and nonnegative')
    if min_price is not None and max_price is not None and min_price > max_price:
        raise HTTPException(status_code=422, detail='Minimum price cannot exceed maximum price')


def with_image_urls(results):
    return [{**row, 'image_url': image_url(row.get('image_path'))} for row in results]


@app.get('/api/categories')
def categories(request: Request):
    products = request.app.state.products.values()
    return {
        'main_categories': sorted({p['main_category'] for p in products if p['main_category']}),
        'original_categories': sorted({p['original_category'] for p in products if p['original_category']}),
    }


@app.get('/api/products')
def catalog(request: Request, page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
            category: str | None = None, min_price: float | None = None, max_price: float | None = None):
    check_filters(category, min_price, max_price)
    category = category.strip().casefold() if category else None
    matches = [p for p in request.app.state.products.values()
               if (not category or category in (p['main_category'].casefold(), p['original_category'].casefold()))
               and (min_price is None or p['price'] >= min_price)
               and (max_price is None or p['price'] <= max_price)]
    start = (page - 1) * page_size
    return {'items': [product_card(p) for p in matches[start:start + page_size]],
            'total': len(matches), 'page': page, 'page_size': page_size}


@app.get('/api/products/{product_id}')
def product_details(product_id: str, request: Request):
    product = product_or_404(request, product_id)
    return {**product_card(product), 'description': product['description'] or [],
            'specifications': product['technical_features'] or [], 'categories': product['categories'] or []}


@app.get('/api/search/text')
def search_text(request: Request, q: str = Query(min_length=1, max_length=500),
                category: str | None = None, min_price: float | None = None,
                max_price: float | None = None, limit: int = Query(5, ge=1, le=100)):
    check_filters(category, min_price, max_price)
    if not q.strip():
        raise HTTPException(status_code=422, detail='Query must not be empty')
    with request.app.state.text_lock:
        results = request.app.state.search.text(q, category, min_price, max_price, limit)
    return {'query': q, 'items': with_image_urls(results)}


@app.post('/api/search/image')
def search_image(request: Request, file: UploadFile = File(...), category: str | None = None,
                 min_price: float | None = None, max_price: float | None = None,
                 limit: int = Query(5, ge=1, le=100)):
    check_filters(category, min_price, max_price)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'upload'
        size = 0
        with path.open('wb') as target:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail='Image exceeds 10 MiB')
                target.write(chunk)
        try:
            with Image.open(path) as image:
                image.verify()
            with request.app.state.image_lock:
                results = request.app.state.search.image(path, category, min_price, max_price, limit)
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail='Invalid or unreadable image') from exc
    return {'items': with_image_urls(results)}


@app.get('/api/products/{product_id}/recommendations')
def recommendations(product_id: str, request: Request):
    product_or_404(request, product_id)
    return {'product_id': product_id,
            'items': with_image_urls(request.app.state.recommender.recommend(product_id, 5))}


@app.get('/api/products/{product_id}/sentiment')
def sentiment(product_id: str, request: Request, sentiment: Literal['positive', 'negative', 'neutral'] | None = None):
    product_or_404(request, product_id)
    reviews = request.app.state.reviews[product_id]
    filtered = [review for review in reviews if sentiment is None or review['sentiment'] == sentiment]
    return {'product_id': product_id, 'summary': request.app.state.summaries[product_id],
            'sentiment_filter': sentiment, 'total_reviews': len(filtered), 'reviews': filtered}
