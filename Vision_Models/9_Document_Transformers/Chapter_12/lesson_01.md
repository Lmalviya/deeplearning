# 13.1 OCR in Practice — EasyOCR

## Problem

Extraction Service (Chapter 2.2) needs a concrete, working OCR implementation for pages with no
usable text layer. This lesson is the practical reference — working code, actual output shape,
and the tool-selection trade-offs — for the choice made in Chapter 2.2: **EasyOCR** for
ease of integration during initial build-out.

## Solution / Concept: Working Setup and Usage

```python
import easyocr

reader = easyocr.Reader(['en'], gpu=True)  # loads once; reuse across calls

results = reader.readtext(image_path)
# results: list of (bbox, text, confidence) tuples
for bbox, text, conf in results:
    print(f"{conf:.2f}  {text}  {bbox}")
```

**Output shape**, matching what Chapter 2.2 assumes downstream: each detected text region
returns a **bounding box** (four corner points), the **recognized text string**, and a
**confidence score** (0–1). This is the raw material for both direct text-content use and any
layout-aware model consuming word/region positions (Chapter 13.4).

**Visualizing detections** (useful for debugging extraction quality, not part of the
production pipeline):

```python
from PIL import Image, ImageDraw

img = Image.open(image_path).convert("RGB")
draw = ImageDraw.Draw(img)
for bbox, text, conf in results:
    draw.polygon([tuple(p) for p in bbox], outline="red", width=2)
img.show()
```

## Known Pitfall: Detection Granularity

EasyOCR, by default, tends to group detections at a **line/sentence level**, not strictly
word-level. This matters directly for Chapter 13.4 (LayoutLMv3), which expects word-level
bounding boxes — feeding it line-level boxes assigns every word in a line the same box,
losing fine-grained intra-line position. Mitigations: reconfigure for finer-grained detection
where the OCR engine supports it, post-process by proportionally splitting a line-level box
across its constituent words, or accept the coarser signal if the consuming model doesn't
critically depend on word-level precision.

## Trade-offs: EasyOCR vs. Alternatives

| Engine | Gain | Cost | Fit |
|---|---|---|---|
| EasyOCR (chosen for MVP) | Fast to integrate, GPU-accelerated, minimal system-level setup friction, works well out of the box for prototyping | Line-level-leaning detection granularity (see above); generally lower throughput/accuracy ceiling than more specialized engines at scale | Initial build-out and iteration speed, per Chapter 2.2's MVP choice |
| PaddleOCR | Often better throughput/accuracy trade-off at volume; includes layout analysis in the same ecosystem (Chapter 13.2) | More complex install/dependency surface (see Chapter 13.2's known pitfalls); steeper learning curve | Natural upgrade path once GPU cost or accuracy becomes the binding constraint (Chapter 2.2, Chapter 11.1) |
| Tesseract | Mature, well-documented, native word-level output via `image_to_data`, no GPU required | Generally lower accuracy on noisy/low-quality scans than modern deep-learning-based engines | Useful specifically when word-level granularity is required and a lighter-weight, CPU-only engine is acceptable |
| Cloud OCR APIs (AWS Textract, Google Document AI) | Removes GPU fleet management for this stage entirely; typically strong accuracy | Per-call cost that scales directly with volume — a real Chapter 11.1 cost-driver consideration at 100M-document scale | Worth evaluating specifically against the Chapter 11.1 cost breakdown once self-hosted GPU OCR's fleet cost is well understood and comparable |

## When to Use / When Not To

- **Use EasyOCR** during initial build-out and while iterating on the extraction pipeline's
  correctness — its ease of setup directly serves development speed.
- **Reconsider the engine choice** once the system reaches the scale where the Chapter 11.1
  cost breakdown makes OCR/HTR compute a meaningful line item, or once accuracy on real
  production documents shows a measurable gap that a more specialized engine would close.

## Summary

EasyOCR provides `(bounding box, text, confidence)` triples per detected region, is fast to set
up, and is the right MVP choice for the reasons stated in Chapter 2.2 — with one known,
important caveat: its detection granularity leans toward line-level rather than strict
word-level, which matters directly once a layout-aware classification model (Chapter 13.4) is
introduced downstream. The upgrade path (PaddleOCR, cloud APIs) is a tool swap, not an
architecture change, consistent with the routing-vs-tooling separation established in Chapter
2.2.