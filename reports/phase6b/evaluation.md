# Phase 6B: expanded catalog validation

The expanded catalog contains 1,120 products and 9,748 reviews. The 1,000 original product rows and 8,662 original review rows matched their Phase 6A snapshots; the original 1,000 image hashes and all 8,662 saved sentiment prediction rows also matched. All 1,120 WebP images decoded successfully. The source product and review data were not changed in this phase.

## Updated artifacts

| Artifact | Count | Update |
| --- | ---: | --- |
| Review sentiment | 9,748 | Reused 8,662 predictions and inferred 1,086 with the selected product-review model. |
| Product sentiment | 1,120 | Recomputed summaries from the matching review predictions. |
| E5 text FAISS | 1,120 × 384 | Reused 1,000 validated vectors and encoded 120 passages. |
| CLIP image FAISS | 1,120 × 512 | Reused 1,000 ID- and image-hash-matched vectors and encoded 120 images. |

Both embedding arrays are finite and normalized, their original 1,000 vectors are bit-for-bit unchanged, and product IDs and index order match the dataset. Cache reloads succeeded without re-encoding. This local CPU run took 311.8 seconds for sentiment including reporting, 161.6 seconds for the text build and 97.5 seconds for the image build; the processes overlapped, so these are not isolated benchmarks.

The expanded review evaluation is in [the sentiment report](../sentiment/expanded/sentiment_final_evaluation.md); the previous reports are preserved. Agreement with star-derived labels was 76.15%, macro F1 0.625 and neutral F1 0.307. Ratings are weak labels, not independent ground truth; the clothing-review model can also miss electronics-specific or mixed sentiment.

## Search and integration

FastAPI was exercised against the updated cached models and indexes. Top hybrid results were:

| Query | First result | Observation |
| --- | --- | --- |
| mouse | B000068MP5, Microsoft Wireless Optical Mouse Blue | All five top results were mice. |
| keyboard | B07VW79ZSP, iClever DK03 Bluetooth Keyboard | All five top results were keyboards. |
| webcam | B09G1D9CR4, Webcam 1080p HD Computer Camera | All five top results were webcams. |
| simsiz siçan | B000BFO320, Logitech LX5 Cordless Optical Mouse | The first result was a mouse, but the other four included unrelated network equipment and a camera. |

A mouse search filtered to the Computer Accessories & Peripherals source category and prices up to $25 returned five matches satisfying both filters. Local image searches for new mouse B000068MP5 and keyboard B000AOYWVE returned five visually similar mice and keyboards, respectively, excluding the queried catalog image. The webcam B000BDH2XY returned three webcams, a security camera and earbuds: visual similarity alone does not guarantee product-type relevance. The upload endpoint treats uploaded files as external images and returned the matching catalog item first in all three cases, as designed.

The API checks also exercised new product details, served WebP images, original reviews, sentiment summaries and five recommendations, with reusable index instances. Sample measured hybrid API request times were 144.5 ms for the first mouse query, then 45.8–55.2 ms for the other three queries on this machine; they are illustrative rather than a benchmark.

## Recommendations

All **1,120 of 1,120** products received five unique recommendations; none had fewer than five or zero. Every result excluded its selected product, stayed within its main category, and respected the ordering of original-category matches before main-category fallback.

Representative new products: the Microsoft mouse B000068MP5 returned five mice; the Lenovo keyboard B000AOYWVE returned five keyboards; the SoundTech microphone B002XGDXJY returned five microphones; the Cable Matters hub B00PHPWLPA returned five hubs or adapters; and the Samsung monitor B00A37NU1Q returned five monitors. The Logitech webcam B000BDH2XY returned four webcams and a microphone. Source categories can contain multiple product types, so category relevance plus multimodal similarity does not guarantee exact-type matches. No ranking or category logic was changed in this phase.
