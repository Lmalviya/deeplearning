# 2.2 OCR Engines, Layout Parsers, and Handwriting Recognition (HTR)

## Problem

Once a page has been identified as "no usable text layer" (Lesson 2.1), it needs to be
converted into some machine-usable representation of its content. But "read the page" is not
one problem — it splits into at least three genuinely different sub-problems, each needing a
different kind of tool: reading printed words, understanding the page's structural regions,
and reading handwriting. Conflating these leads to using the wrong tool and getting
confidently wrong output (not a clean failure).

## Solution / Concept: Three Distinct Tools

### OCR Engines (Optical Character Recognition)

Takes a page image, outputs **text** — but more precisely, for each detected word or line:
the recognized text string, a **bounding box** (physical position on the page), and a
**confidence score**. Example output:

```
"Invoice"    box=(120,45,80,20)   conf=0.98
"Total:"     box=(120,600,60,15)  conf=0.91
```

This means OCR gives **both content and spatial layout** in one pass — relevant later when
choosing between text-only and layout-aware model architectures (Chapter 3).

Tools: Tesseract, EasyOCR, PaddleOCR (open-source); AWS Textract, Google Document AI (cloud).

**Detection granularity varies by engine/config**: some detect at word level, others at
line/sentence level. Line-level detection assigns every word in that line the *same* bounding
box, losing fine-grained intra-line position — relevant for layout-aware models (Chapter 3.3),
which expect word-level boxes. Fixes: reconfigure the OCR engine for word-level output (e.g.,
Tesseract's `image_to_data`), proportionally split a line-level box across its words, or accept
the coarser signal if classes are visually distinct enough not to need it.

### Layout Parsers / Layout Analysis

Segments a page into **structural regions** — "this is a table," "this is a title," "this is
a paragraph," "this is a signature block" — rather than individual words. Tools: LayoutParser,
Detectron2-based models, PaddleOCR's PP-StructureV3, DocLayNet-trained models.

This is useful as a **classification feature even without reading any text**: an invoice has a
table region near the bottom; a resume has no table region at all, just stacked text blocks; a
contract is almost entirely one large paragraph region with a signature block at the end.
Rule-based classification on layout alone works well for visually distinct classes, but is
brittle: a real-world document that doesn't follow the assumed template (e.g., a one-page
invoice with no line-item table) breaks a hand-written rule. This tension is resolved properly
in Chapter 3 (learned models vs. rules).

### Handwriting Recognition (HTR)

A genuinely harder, different problem from printed OCR. Printed OCR works because fonts are
consistent, regular, machine-generated shapes. Handwriting has no fixed glyph shapes — every
writer's strokes differ, cursive connects letters unpredictably, and stroke order/style vary
enormously. HTR is dominated by **sequence models trained end-to-end** (not "detect character
→ classify character" the way classic OCR works): CRNN (CNN + RNN + CTC loss) architectures,
or transformer-based models like TrOCR (Microsoft). Trained on paired (handwriting image,
transcription) datasets like the IAM Handwriting Database.

**Practical danger:** printed-OCR engines fed handwritten pages don't fail cleanly — they can
confidently output wrong words with plausible-looking confidence scores. Detecting "is this
page handwritten" *before* choosing the extraction engine is itself a small classification
problem, not something to skip.

## Trade-offs

| Approach to printed-vs-handwritten detection | Gain | Cost |
|---|---|---|
| OCR confidence as a heuristic (run printed OCR, flag low-confidence output as "possibly handwritten") | Free — no extra model needed | Conflates handwriting with any low-quality/blurry printed page; not safe as the sole signal |
| Dedicated binary classifier (printed vs. handwritten vs. optionally mixed) | Texturally distinct problem (stroke width uniformity, baseline regularity) — genuinely easier than full OCR; public training data exists (RVL-CDIP for printed, IAM for handwritten) | Requires training and maintaining one more small model |

## When to Use / When Not To

- **Use OCR** whenever the page is printed/typed and no text layer exists.
- **Use a layout parser** when classification or extraction benefits from structural cues
  independent of exact word content, or as a feature alongside text/image signals in a fusion
  model (Chapter 3.3).
- **Use HTR, never printed OCR,** on pages identified as handwritten — printed OCR is not a
  safe fallback here.
- **Page-level printed/handwritten classification is an acceptable phase-1 simplification**
  when mixed printed+handwritten *regions on the same page* (e.g., a printed form with
  handwritten fill-ins) are rare or tolerable to get wrong; otherwise this needs region-level
  detection (layout parser + per-region classification), deferred as a phase-2 refinement.

## Summary

"Read the page" splits into OCR (printed text + word boxes + confidence), layout parsing
(structural regions, independent of word content), and HTR (a genuinely different,
sequence-modeling problem for handwriting). Printed OCR silently fails on handwriting rather
than erroring out, so a printed-vs-handwritten decision must be made explicitly, ideally with a
dedicated lightweight classifier, before choosing which engine reads a given page.