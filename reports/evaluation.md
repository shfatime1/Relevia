# Historical search and recommendation results

The 1,000-product Phase 4 checks used BM25 and multilingual E5 with equal-weight reciprocal-rank fusion; image search used CLIP cosine similarity. These examples lack independent relevance judgments. [Search results](search/examples.json) and [recommendation results](recommendation/examples.json) record the queries and ranked products. These results predate later query-understanding changes.

| Search example | Top-five result |
|---|---|
| `Canon PowerShot` | Five PowerShot cameras |
| `wireless headphones for listening to music` | Five wireless headphones |
| `simsiz bluetooth qulaqlıq` | One headphone; four unrelated products |
| `şəkil çəkmək üçün rəqəmsal kamera` | Five cameras |
| Camera image | Five cameras |
| Router image | Two routers, earbuds and two external drives |

Category and inclusive-price filters returned five qualifying results in both tested modes. Catalog images excluded themselves; external copies could return the matching product.

The 1,000-product recommendation check returned five unique same-main-category results for every seed, with original-category matches first; 48 seeds used fallback. The earlier strict-category rule returned five for 952 seeds and none for 12. A fingerprint reader fell back to alarms and cameras, which are not substitutes. Coverage is not a relevance judgment. Ranking used E5, CLIP, category relevance and a small sentiment adjustment.

## Expanded catalog

Phase 6B checked 1,120 products and 9,748 reviews. All 1,120 images decoded; original rows, image hashes, 8,662 sentiment predictions and 1,000 text and image vectors matched the pre-expansion snapshots.

| Artifact | Expanded count | Reused | Newly inferred |
|---|---:|---:|---:|
| Review sentiment | 9,748 | 8,662 | 1,086 |
| E5 text vectors | 1,120 × 384 | 1,000 | 120 |
| CLIP image vectors | 1,120 × 512 | 1,000 | 120 |

The text, image and sentiment runs took 161.6, 97.5 and 311.8 seconds respectively on CPU; processes overlapped. Product summaries were recomputed for 1,120 products. Indexes matched product order, vectors were normalized and caches reloaded. The [expanded sentiment metrics](sentiment/expanded/sentiment_final_metrics.json) use star-derived weak labels.

Hybrid `mouse`, `keyboard` and `webcam` queries each returned five matching types; `simsiz siçan` returned one mouse followed by unrelated items. A webcam image returned three webcams, one security camera and earbuds. All 1,120 seeds received five unique recommendations within their main category; a webcam received four webcams and one microphone. These checks establish coverage and behavior, not relevance or compatibility.
