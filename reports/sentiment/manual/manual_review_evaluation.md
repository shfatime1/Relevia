# AI-labeled review sample evaluation

**The reference labels are AI-generated, not independent human ground truth.** The sample is small and not representative of all reviews: it contains nine reviews per star rating and deliberately oversamples model disagreements. These metrics describe agreement with these labels, not unbiased population performance.

Reference labels: {'positive': 15, 'negative': 25, 'neutral': 5}; model disagreements: 25 of 45.

| Metric | Tweet-trained baseline | Product-review model |
|---|---:|---:|
| Accuracy | 48.89% | 64.44% |
| Macro F1 | 0.470 | 0.549 |
| Negative precision | 0.833 | 0.905 |
| Negative recall | 0.400 | 0.760 |
| Negative f1-score | 0.541 | 0.826 |
| Neutral precision | 0.158 | 0.182 |
| Neutral recall | 0.600 | 0.400 |
| Neutral f1-score | 0.250 | 0.250 |
| Positive precision | 0.643 | 0.615 |
| Positive recall | 0.600 | 0.533 |
| Positive f1-score | 0.621 | 0.571 |

Confusion matrices use AI-generated reference labels as rows and model predictions as columns.

**Tweet-trained baseline**

| Reference | Negative | Neutral | Positive |
|---|---:|---:|---:|
| Negative | 10 | 12 | 3 |
| Neutral | 0 | 3 | 2 |
| Positive | 2 | 4 | 9 |

**Product-review model**

| Reference | Negative | Neutral | Positive |
|---|---:|---:|---:|
| Negative | 19 | 4 | 2 |
| Neutral | 0 | 2 | 3 |
| Positive | 2 | 5 | 8 |

**Recommendation:** The product-review model has the higher macro F1 on this sample and is now the default. This is provisional: the AI labeling process may share biases with the models, the sample has only 45 items, and the Neutral class has few examples. The labeling CSV contains ratings and model predictions; whether those fields were hidden during AI labeling is unknown, so label contamination cannot be ruled out. Obtain independently human-labeled, representative reviews to validate its use.

Full metrics, model revisions, and input file hashes are in [manual_review_metrics.json](manual_review_metrics.json).
