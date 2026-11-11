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
