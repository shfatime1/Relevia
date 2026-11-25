# Historical sentiment comparisons

The tweet baseline (`ZyroGod/distilbert-sentiment-classifier`, revision `cbfee930df13824cf267b8d93921dd3a1783af83`) and current product-review model (`MilindSingh316/distilbert-clothing-review-sentiment`, revision `1afdfec74e1df72dabf0b82e011d89ecbee9e614`) classified the same 8,662 reviews, truncated to 128 tokens. Ratings supplied weak labels: 1–2 negative, 3 neutral, 4–5 positive. Source SHA-256: `d9285a26f4e8682587b4665e23b9f5b67c60d357ca8880e0d86a0cc291909d3d`.

| Weak-label metric | Tweet baseline | Product-review model |
|---|---:|---:|
| Agreement | 70.75% | 76.36% |
| Macro F1 | 0.552 | 0.626 |
| Negative F1 | 0.642 | 0.700 |
| Neutral F1 | 0.183 | 0.306 |
| Positive F1 | 0.832 | 0.872 |
| Neutral precision / recall | 13.5% / 28.4% | 22.7% / 47.0% |

The labels were 72.9% positive, 18.6% negative and 8.5% neutral; an always-positive classifier would agree with 72.9%. Confusion matrices and complete class metrics are in [comparison JSON](sentiment_model_comparison.json) and [baseline JSON](../baseline/sentiment_metrics.json). Ratings are not verified text sentiment; mixed reviews, model domain shift and truncation limit interpretation. The selected reviews are not representative of the full Amazon catalog.

## AI-labeled sample

The 45-review sample contained nine reviews per star rating and oversampled disagreements (25 of 45). Reference labels were **AI-generated, not independent human ground truth**: 25 negative, 5 neutral and 15 positive. The sample is not representative. Ratings and predictions were present in the labeling file; it is unknown whether they were hidden, so contamination is possible.

| AI-reference metric | Tweet baseline | Product-review model |
|---|---:|---:|
| Agreement | 48.89% | 64.44% |
| Macro F1 | 0.470 | 0.549 |
| Negative precision / recall / F1 | 0.833 / 0.400 / 0.541 | 0.905 / 0.760 / 0.826 |
| Neutral precision / recall / F1 | 0.158 / 0.600 / 0.250 | 0.182 / 0.400 / 0.250 |
| Positive precision / recall / F1 | 0.643 / 0.600 / 0.621 | 0.615 / 0.533 / 0.571 |

See [sample metrics](../manual/manual_review_metrics.json) for confusion matrices, model revisions and input hashes.
