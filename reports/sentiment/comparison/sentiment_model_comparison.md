# Sentiment model comparison

Both models classified the same 8,662 review texts using at most 128 tokens. Ratings were used only to form weak comparison labels: 1–2 Negative, 3 Neutral, 4–5 Positive. Source SHA-256: `d9285a26f4e8682587b4665e23b9f5b67c60d357ca8880e0d86a0cc291909d3d`.

Baseline: [ZyroGod/distilbert-sentiment-classifier](https://huggingface.co/ZyroGod/distilbert-sentiment-classifier) at `cbfee930df13824cf267b8d93921dd3a1783af83`. Product-review model: [MilindSingh316/distilbert-clothing-review-sentiment](https://huggingface.co/MilindSingh316/distilbert-clothing-review-sentiment) at `1afdfec74e1df72dabf0b82e011d89ecbee9e614`; its model card describes training on clothing reviews.

| Metric | Tweet-trained baseline | Product-review model |
|---|---:|---:|
| Agreement | 70.75% | 76.36% |
| Macro F1 | 0.552 | 0.626 |
| Negative F1 | 0.642 | 0.700 |
| Neutral F1 | 0.183 | 0.306 |
| Positive F1 | 0.832 | 0.872 |

Neutral precision / recall: baseline 13.5% / 28.4%; product-review model 22.7% / 47.0%.

Confusion matrices use star-derived labels as rows and predictions as columns, ordered Negative, Neutral, Positive.

**Tweet-trained baseline**

| Weak label | Negative | Neutral | Positive |
|---|---:|---:|---:|
| Negative | 1,163 | 282 | 168 |
| Neutral | 336 | 208 | 188 |
| Positive | 512 | 1,048 | 4,757 |

**Product-review model**

| Weak label | Negative | Neutral | Positive |
|---|---:|---:|---:|
| Negative | 1,244 | 297 | 72 |
| Neutral | 276 | 344 | 112 |
| Positive | 419 | 872 | 5,026 |

**Recommendation:** The product-review model has the higher Neutral F1 on this weak-label comparison and is now the default. The choice remains provisional pending independently human-labeled evaluation.

These metrics measure agreement with ratings rather than human-verified text sentiment. The 3-star Neutral proxy can contain mixed or strongly polarized text, and class imbalance affects agreement. The candidate was trained on clothing, not electronics; its product-review training may use related rating-derived labels. Overlap with the evaluation data has not been independently verified. Both models truncate longer reviews. The selected reviews are not representative of the full Amazon catalog. A blinded, human-labeled electronics-review sample is needed before treating the recommendation as validated sentiment quality.

Full per-class precision, recall, F1, support, and prediction distributions are in [sentiment_model_comparison.json](sentiment_model_comparison.json).
