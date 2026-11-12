# Baseline sentiment diagnostics (tweet-trained model)

The saved evaluation compares predictions for 8,662 reviews with **weak labels** derived from ratings: 1–2 stars = Negative, 3 = Neutral, and 4–5 = Positive. The matrix below uses weak labels as rows and predictions as columns.

| Star-derived label | Predicted Negative | Predicted Neutral | Predicted Positive | Total |
|---|---:|---:|---:|---:|
| Negative | 1,163 | 282 | 168 | 1,613 |
| Neutral | 336 | 208 | 188 | 732 |
| Positive | 512 | 1,048 | 4,757 | 6,317 |

The largest individual disagreement is **Positive → Neutral: 1,048 reviews**, or 16.6% of Positive weak labels. In the reverse direction, 188 Neutral weak labels were predicted Positive; together, this pair accounts for 1,236 disagreements. Neutral ↔ Negative accounts for 618 (336 Neutral → Negative and 282 Negative → Neutral); Negative ↔ Positive accounts for 680.

Only **208 of 732** three-star reviews were predicted Neutral: Neutral recall is **28.4%**. Of those three-star reviews, 45.9% were predicted Negative and 25.7% Positive. Conversely, only **208 of 1,538** predicted-Neutral reviews have a Neutral weak label, giving Neutral precision of **13.5%** and F1 of **0.183**. Of all predicted-Neutral reviews, 68.1% have Positive and 18.3% Negative weak labels. These are disagreements with star labels, not necessarily verified sentiment errors.

## Imbalance and interpretation

The weak labels are 72.9% Positive, 18.6% Negative, and 8.5% Neutral. A trivial always-Positive predictor would agree with **72.9%** of these weak labels, compared with the model's **70.7%**. That baseline would have zero recall for both minority classes, so overall agreement alone is misleading. The model's macro F1 is **0.552**, compared with weighted F1 **0.742**; the latter is dominated by the large Positive class. The selected reviews were not sampled to represent the full Amazon review population.

The rating is an imperfect proxy for text sentiment. A three-star review may contain strong praise and complaints, and a rating can reflect factors absent from the text. The model was trained on English tweets, reads only review text, and truncates it at 128 tokens; domain shift and omitted text can contribute to disagreement. Confidence is a model score, not a verified probability of correctness. A manually labeled, representative sample is needed to determine which side of a disagreement reflects the expressed sentiment.

## Illustrative reviews

These examples come from [sentiment_examples.csv](sentiment_examples.csv), which selects high-confidence examples in each matrix cell. They illustrate failure modes but are not a random or statistically representative sample. IDs below are prefixes of the saved `review_id` values.

- **Neutral → Negative**, `80fe70f7a3245a` (3 stars; confidence 0.968): “The unit works fine” but the reviewer says the business practices and sync software “absolutely suck.” A mixed review can express strongly negative sentiment despite a three-star label.
- **Neutral → Positive**, `2b47617b815f3a` (3 stars; 0.966): the reviewer says the product needs a flash or bright room, then writes “we really like it” and “Lots of fun.” The positive prediction captures part of a mixed review.
- **Positive → Neutral**, `516ab668303894` (5 stars; 0.936): “on time and all work as advertised.” The terse endorsement is positive by rating, while the prediction is Neutral.
- **Negative → Neutral**, `d0b2f260ed4643` (2 stars; 0.942): the reviewer reports an opened box, missing parts, and a return. The predicted Neutral label misses an apparent complaint.
- **Negative → Positive**, `c5352c1b2075eb` (2 stars; 0.967): “Happy with my purchase.” The text itself is positive although the rating supplies a Negative weak label; the reason for the mismatch is unknown.

See [sentiment_metrics.json](sentiment_metrics.json) and [sentiment_evaluation.md](sentiment_evaluation.md) for the full metrics and evaluation context.
