# Baseline sentiment evaluation (tweet-trained model)

Analyzed 8,662 reviews with [ZyroGod/distilbert-sentiment-classifier](https://huggingface.co/ZyroGod/distilbert-sentiment-classifier) at revision `cbfee930df13824cf267b8d93921dd3a1783af83`. The model reads review text (not the rating), truncated to 128 tokens.

Star ratings are weak labels: 1–2 stars are negative, 3 neutral, and 4–5 positive. Agreement measures correspondence with those labels, not accuracy against verified sentiment. The confusion matrix uses weak labels as rows and predictions as columns. Precision, recall, and F1 in the JSON compare each class to those weak labels.

Agreement: **70.7%**; macro F1: **0.552**.

Neutral precision: **13.5%**; neutral recall: **28.4%**. These figures show substantial disagreement on the neutral class under the star-label proxy.

The model was trained on English tweets rather than electronics reviews; domain shift, sarcasm, mixed opinions, truncation, and uncalibrated confidence can affect results. A star rating can disagree with the sentiment of the text, especially for three-star reviews. The selected source reviews are not a representative sample of all Amazon reviews. Examples are the highest-confidence rows per non-empty confusion-matrix cell, not a random sample. Manual labeling is needed for a ground-truth evaluation.
