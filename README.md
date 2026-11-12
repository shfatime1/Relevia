# Relevia — Phase 1: Data Preparation

This pipeline streams the original [Amazon Reviews 2023 Electronics metadata and reviews](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) and joins records by `parent_asin`. It produces product and review Parquet files and one local WebP image per selected product. It does not build models or serve an application.

## Run locally

Python 3.10 or later is recommended. From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python scripts\prepare.py --size 1000
.venv\Scripts\python scripts\validate.py
```

On macOS/Linux or Google Colab, use `python -m venv .venv` followed by `.venv/bin/python -m pip install -r requirements.txt` and `.venv/bin/python scripts/prepare.py --size 1000`. In Colab, installing `requirements.txt` into the notebook environment and calling the script with `!python` also works.

The original uncompressed files are approximately 5.2 GB of metadata and 22.6 GB of reviews. They are streamed, not stored as complete downloads. Defaults inspect the first 200,000 metadata rows and stream reviews until the source ends or at least twice the requested number of candidates have five reviews (checked every 100,000 records). Use `--metadata-limit 0` for all metadata or `--review-limit N` to bound a trial. `--candidates-per-category` controls the maximum saved candidate pool for each source category. The source order is not random, so limited runs are not representative samples. The script may produce fewer products than requested when the source sample, filtering, review coverage, or image availability is insufficient.

Candidates require valid IDs and titles, positive numeric prices, technical-looking feature text, and source taxonomy paths. The third level of the source Electronics taxonomy is retained as `original_category` (and `category` for compatibility); `main_category` groups related observed categories using the mapping in `scripts/prepare.py`. Unknown categories are skipped until explicitly mapped. Accessory and other ambiguous paths are excluded. Conservative text filters reject service, warranty, promotional and replacement listings; automated filters are imperfect and do not establish that every remaining listing is a physical electronics product. Selection starts with a cap of 10% of the requested size per original category, increasing it only when needed to use available eligible candidates. Images must download and decode before selection. Ratings and review text come directly from the source; up to 10 reviews are retained per product, with observed rating values represented first among the first 30 matching nonempty reviews. This preserves available diversity within that bounded pool, not necessarily the full source history. Empty descriptions or missing brands and aggregate ratings remain missing rather than being invented.

`data/progress.sqlite` caches completed metadata and review stages. Existing products, reviews and images are preserved when extending the selection; requesting fewer products than already exist stops with an error. Interrupted source stages restart their stream. Cache settings must match the original run: for the current local cache use `--metadata-limit 100000 --review-limit 3000000`. Changing stream limits or candidate caps requires a fresh progress database and a new scan. Output includes `data/products.parquet`, `data/reviews.parquet`, and `data/images/`. The validator reports both category distributions, ratings, missing values, image integrity, and disk usage; when the cache exists it also checks reviews against the original cached records. Use `scripts/validate.py --expected-size 1000` to require the requested count. Dataset terms, provenance, and citation details are on the dataset card; check them before redistribution or downstream use.

## Phase 2A: text search

After installing the updated requirements, compare the two methods with:

```powershell
.venv\Scripts\python -X utf8 backend\search.py "wireless bluetooth headphones"
.venv\Scripts\python -X utf8 backend\search.py "simsiz bluetooth qulaqlıq" --top 5
.venv\Scripts\python -X utf8 -m backend.validate_search
```

Use `.venv/bin/python` on macOS/Linux. The first search downloads the pinned [multilingual E5 small model](https://huggingface.co/intfloat/multilingual-e5-small) and computes normalized 384-dimensional embeddings on CPU in batches. Product text combines title, brand, main and original category, source categories, technical features, and available descriptions. E5 inputs are truncated at 256 tokens; the categories and features precede descriptions to keep product-identifying fields near the start. Queries use `query: ` and products use `passage: `, as required by the model.

BM25 uses case-folded Unicode word tokens and returns only products with matching terms. Semantic search uses a FAISS inner-product index over normalized vectors, so its scores are cosine similarities. Scores from the two methods have different scales and should not be compared directly. Azerbaijani queries work without a translation service, although the predominantly English catalog, multilingual model coverage, and absence of product-specific training can produce weak cross-language matches; BM25 needs shared words or model names.

The model is cached under `.cache/huggingface` and vectors, FAISS index, and ordered product-ID manifest under `.cache/text_search`; `.cache/` is Git-ignored. A dataset hash, model revision, text configuration, and cache-file checksums control reuse. Changing these rebuilds the embeddings and index automatically. Existing Phase 1 artifacts are read only. The CLI setup time includes any model download and index build, and its first semantic query also loads the model; warm-query timing is reported by the validation command.

## Phase 2B: visual search

After installing the updated requirements, search by an existing product image or an external image:

```powershell
.venv\Scripts\python -X utf8 backend\image_search.py data\images\B06XYHDX83.webp --top 5
.venv\Scripts\python -X utf8 backend\image_search.py C:\path\to\photo.jpg
.venv\Scripts\python -X utf8 -m backend.validate_image_search
```

The first run downloads the pinned [CLIP ViT-B/32 model](https://huggingface.co/openai/clip-vit-base-patch32) and computes normalized 512-dimensional image vectors in batches. CUDA is used when available in the installed PyTorch build; otherwise inference runs on CPU. The separate `.cache/image_search` directory contains the vectors, ordered product-ID and image-hash manifest, and FAISS inner-product index. Unchanged image vectors are reused; changed images are re-encoded and a changed index is rebuilt. Model/configuration changes invalidate the vector cache. The model and indexes are Git-ignored.

Searching a catalog image excludes that product's resolved image path from the results. An external file, including a copy of a catalog image, is treated as an external query. Similarity is visual cosine similarity, not a guarantee that items have matching specifications or functions; results need task-specific evaluation. Phase 1 data and the Phase 2A text-search index are not modified.

## Phase 3: review sentiment

```powershell
.venv\Scripts\python -X utf8 backend\sentiment.py run --batch-size 32
.venv\Scripts\python -X utf8 backend\sentiment.py product B06XYHDX83
.venv\Scripts\python -X utf8 backend\sentiment.py validate
```

The default is the pinned three-class `MilindSingh316/distilbert-clothing-review-sentiment` model, applied to review text with a 128-token limit. It selects CUDA when the installed PyTorch supports it, otherwise CPU. Predictions are checkpointed in `.cache/sentiment_product_comparison` and reused when the review file and model configuration are unchanged. Generated `data/review_sentiment.parquet` and `data/product_sentiment.parquet` are Git-ignored. The original tweet-trained model's results remain in `data/review_sentiment_tweet_baseline.parquet` and `data/product_sentiment_tweet_baseline.parquet`; its cache remains in `.cache/sentiment`. If the ignored baseline datasets are absent, recreate them with `python backend/sentiment.py run --baseline` using the virtual environment; this leaves the default datasets and saved evaluation reports unchanged. Evaluation artifacts under `reports/sentiment/` are intended to be tracked and grouped into `baseline`, `final`, `comparison`, and `manual`.

Review IDs are SHA-256 hashes of the complete source row plus an occurrence number for identical rows; the source contains no review-ID field. Product summaries include counts, percentages, and mean observed review rating. On the 8,662 reviews, the selected model reached 76.36% agreement with star-derived weak labels, macro F1 of 0.626, and Neutral F1 of 0.306 (Neutral precision 22.7%, recall 47.0%). Against 45 AI-generated reference labels it reached 64.44% agreement and macro F1 0.549; those labels are not independent human ground truth, and the sample is small and enriched for model disagreements. Star-derived labels are also imperfect: mixed sentiment can occur at any rating. The model was trained on clothing reviews rather than electronics, longer texts are truncated, and confidence is not necessarily calibrated. Independently human-labeled, representative reviews are needed to assess sentiment quality. See `reports/sentiment/comparison/sentiment_model_comparison.md`, `reports/sentiment/manual/manual_review_evaluation.md`, and `reports/sentiment/final/sentiment_final_evaluation.md`; the original tweet-trained evaluation reports are in `reports/sentiment/baseline/`.

Reproduce the model comparison with `python -m backend.compare_sentiment` and the AI-label evaluation with `python scripts/evaluate_labeled_reviews.py`, using the virtual environment. `python -m scripts.sample_manual_reviews` recreates the fixed-seed sample only when its output does not already exist, to avoid overwriting labels. These evaluations require the ignored source, baseline, and comparison prediction datasets; regenerating missing predictions can require model downloads and CPU/GPU inference.

## Phase 4A: product recommendations

```powershell
.venv\Scripts\python -X utf8 -m backend.recommend B0001G6U4S
```

The CLI recommends up to five products and accepts `--top` to change the maximum. It reuses the Phase 2 E5 and CLIP indexes and Phase 3 product sentiment summaries; the existing search loaders rebuild stale indexes as described above. Candidates must share the selected product's main category. Products in the same original source category are ranked first; if there are too few, other products in the same main category fill the remaining slots. Within each tier, matching finer source subcategories receive a small bonus, and the score combines text cosine similarity (55%), image cosine similarity (30%), category relevance (15%), and a small adjustment based on predicted sentiment. Scores are ranking scores, not probabilities or pure cosine similarities; a fallback result can have a higher score than an original-category result because the category tier takes priority. The result includes the individual similarities and reasons.

Small main categories can produce fewer than five matches. Broad categories and image similarity can still group products with different functions or specifications, particularly when using the main-category fallback, so recommendations are not a guarantee of compatibility. Sentiment is based on only 5–10 selected reviews per product and a model with the limitations described above. See `reports/recommendation/` for real examples and validation details.

## Phase 4B: unified search

```powershell
.venv\Scripts\python -X utf8 -m backend.unified_search --text "simsiz bluetooth qulaqlıq" --category "Headphones & Earbuds" --max-price 60
.venv\Scripts\python -X utf8 -m backend.unified_search --image data\images\B0001G6U4S.webp --min-price 50 --max-price 200
```

Choose exactly one of `--text` or `--image`. Optional `--category` matches an entire main or original category, case-insensitively; `--min-price` and `--max-price` are inclusive, and `--top` defaults to five. Filters are applied to the full ranked catalog before limiting results. Text search combines BM25 and E5 with equal-weight reciprocal-rank fusion; keyword-unmatched products can still rank through E5. The displayed text score is a relative fusion score, while the image score is CLIP cosine similarity, so scores across modes or queries are not comparable. The CLI reuses the existing indexes and caches. Image searches exclude the query product when its catalog image path is used, as in Phase 2B.

Search quality depends on the English catalog and pretrained models; Azerbaijani queries, specialized terms and image matches can return unrelated products. Categories and prices are metadata filters, not compatibility checks. See `reports/search/` for real examples and limitations.

## Phase 5A: local API

Install the updated requirements, then start the local server from the repository root:

```powershell
.venv\Scripts\python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for interactive OpenAPI documentation, or `/redoc` for the reference. Startup loads the existing data, indexes, E5 model and CLIP model once per server process; missing or stale caches follow the existing Phase 2 rebuild behavior. Images are available at the relative URLs returned in `image_url` under `/images/`.

- `GET /api/categories` lists main and original categories.
- `GET /api/products?page=1&page_size=20&category=Digital%20Cameras&min_price=50&max_price=200` lists products. Category and price filters are optional; price bounds are inclusive. Categories match a complete main or original name without regard to case.
- `GET /api/products/{product_id}` returns a product's description, specifications, price, ratings and image URL.
- `GET /api/search/text?q=wireless%20headphones` performs hybrid search. Optional `category`, `min_price`, `max_price` and `limit` query parameters also apply to `POST /api/search/image`, which accepts a multipart `file` field containing a decodable image of at most 10 MiB. Uploaded copies are external images, so a matching catalog product may appear in results.
- `GET /api/products/{product_id}/recommendations` returns up to five related products.
- `GET /api/products/{product_id}/sentiment?sentiment=positive` returns the product summary and original review text joined to the final model's predictions. The optional sentiment filter accepts `positive`, `neutral` or `negative`; the summary remains unfiltered. Review timestamps are source Unix milliseconds.

Missing products return 404, invalid parameters or images return 422, and an image larger than 10 MiB returns 413. Empty matches return an empty list. This local API has no authentication; search, recommendation and sentiment limitations described above still apply. To run the essential endpoint checks against the existing local data and caches:

```powershell
.venv\Scripts\python -X utf8 -m unittest discover -s tests -v
```
