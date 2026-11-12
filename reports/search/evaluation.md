# Phase 4B unified search evaluation

`examples.json` records actual top-five results for two English text queries, two Azerbaijani text queries, two catalog images, and one filtered query in each mode. These examples are illustrative and do not provide relevance judgments, precision, or a representative quality measurement. Run the CLI commands in the README with the same catalog and cached models to reproduce the results.

## Observations

- `Canon PowerShot` returned five Canon PowerShot cameras. The natural-language query `wireless headphones for listening to music` returned five wireless headphone products.
- `simsiz bluetooth qulaqlıq` returned one headphone product in the top five alongside a Bluetooth speaker hat, clock, speaker, and sunglasses. The shared word `bluetooth` can favor keyword matches that do not match the full Azerbaijani intent. `şəkil çəkmək üçün rəqəmsal kamera` returned five cameras in this sample. Performance is variable; the catalog is mostly English, and no translation or language-specific training is used.
- A Canon camera image returned five cameras. A router image returned two routers, earbuds, and two external drives. CLIP can match visual appearance or packaging instead of product function.
- The text filter for `Headphones & Earbuds` and prices 20–60 returned five products in that price range. The camera-image filter for `Cameras & Photography` and prices 50–200 returned five cameras in that range. The limits are inclusive. Empty-result cases were checked with a nonexistent category and an unattainable price. Invalid reversed price bounds were rejected.

## Method and checks

Text results combine BM25 and multilingual E5 using equal-weight reciprocal-rank fusion: `61/2 * (1/(60 + E5 rank) + 1/(60 + BM25 rank))`. A product without a BM25 token match receives no BM25 contribution but can still be found by E5. Ranks are computed over the full catalog; category and price filters are then applied before taking the requested number of results. Text scores are relative ranking values, not probabilities. Image results use CLIP cosine similarity; the scores from the two modes and different queries should not be compared as a common relevance scale.

Categories match the full main or original category name case-insensitively. A catalog image path excludes its own product, while an external copy is treated as an external query and may return that product. Validation checked five unique, correctly mapped results per example, score ordering, category and price constraints, query-image exclusion, and external-image behavior. A filtered text query was compared with filtering the full 1,000-result ranking to verify that filtering does not truncate the candidate set prematurely. Existing E5 and CLIP caches were reused without recomputing catalog embeddings. Source and recommendation code were left unchanged.
