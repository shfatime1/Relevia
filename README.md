# Relevia — AI-Powered Product Search & Recommendation

Relevia is an electronics product search and recommendation platform built using NLP, Computer Vision, and Machine Learning. Users can search for products using text or images, explore similar items, and analyze customer reviews.

**[Live Demo](https://shfatime-relevia.hf.space)**

## Features

- **Hybrid Text Search:** Combines BM25 and multilingual E5 embeddings using Reciprocal Rank Fusion (RRF).
- **Natural Language Search:** Supports English and basic Azerbaijani queries, including price and rating constraints, without an external LLM API.
- **Visual Search:** Uses CLIP embeddings and FAISS to retrieve visually similar products.
- **Recommendation System:** Content-based recommendations combining text similarity, image similarity, category relevance, and sentiment.
- **Sentiment Analysis:** Uses a pretrained DistilBERT model to classify customer reviews as positive, neutral, or negative.
- **Product Filtering:** Search by category, price, and other available filters.

## Dataset

Source: [Amazon Reviews 2023 — Electronics](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)

- **1,120 products** across 12 main categories
- **9,748 customer reviews**
- Product metadata, specifications, prices, ratings, and images

The dataset was cleaned, filtered, and processed into Parquet files. Product images were stored locally in WebP format.

## Tech Stack

| Component | Technologies |
|---|---|
| Data Processing | Python, Pandas, Parquet |
| Text Search | BM25, multilingual E5, FAISS, RRF |
| Image Search | CLIP ViT-B/32, FAISS |
| Sentiment Analysis | DistilBERT |
| Recommendation | Content-based Hybrid Ranking |
| Backend | FastAPI |
| Frontend | React, Vite |
| Deployment | Hugging Face Spaces, Gradio Server |

## How It Works

1. Product metadata and reviews are preprocessed from the Amazon dataset.
2. E5 and CLIP generate text and image embeddings, stored in FAISS indexes.
3. User queries are processed and matched against the product catalog.
4. Search results are ranked and filtered using the available product metadata.
5. Users can view recommendations and sentiment summaries for individual products.

The deployed application runs E5 and CLIP inference on CPU and reuses precomputed indexes and sentiment predictions.

## Run Locally

Install the Python dependencies, frontend packages, and Pixi. Prepare the required datasets and model caches, then run:

```bash
pip install -r requirements.txt
cd frontend
npm install
cd ..
pixi run main
```

Frontend: `http://localhost:5173`

API documentation: `http://localhost:8000/docs`

**Note:** The generated datasets, images, model weights, and FAISS caches are not included in the Git repository. See the scripts and backend modules for data preparation and index generation.

## Limitations

- Azerbaijani search is supported through a limited terminology dictionary and multilingual embeddings; some queries may return less relevant results.
- Visual similarity does not guarantee matching product specifications.
- Sentiment predictions are based on a pretrained model and may contain errors.
- Product information and prices come from a historical dataset, not live shopping APIs.

## Links

**Live Demo:** https://shfatime-relevia.hf.space

**GitHub:** https://github.com/shfatime1/Relevia
