# Final sentiment evaluation (product-review model)

Analyzed 8,662 reviews with [MilindSingh316/distilbert-clothing-review-sentiment](https://huggingface.co/MilindSingh316/distilbert-clothing-review-sentiment) at revision `1afdfec74e1df72dabf0b82e011d89ecbee9e614`. The model reads review text (not the rating), truncated to 128 tokens.

Star ratings are weak labels: 1–2 stars are negative, 3 neutral, and 4–5 positive. Agreement measures correspondence with those labels, not accuracy against verified sentiment. The confusion matrix uses weak labels as rows and predictions as columns. Precision, recall, and F1 in the JSON compare each class to those weak labels.

Agreement: **76.4%**; macro F1: **0.626**.

Neutral precision: **22.7%**; neutral recall: **47.0%**. These figures show substantial disagreement on the neutral class under the star-label proxy.

The model was trained on clothing reviews rather than electronics reviews; domain shift, sarcasm, mixed opinions, truncation, and uncalibrated confidence can affect results. A star rating can disagree with the sentiment of the text, especially for three-star reviews. The selected source reviews are not a representative sample of all Amazon reviews. Examples are the highest-confidence rows per non-empty confusion-matrix cell, not a random sample. Manual labeling is needed for a ground-truth evaluation.
