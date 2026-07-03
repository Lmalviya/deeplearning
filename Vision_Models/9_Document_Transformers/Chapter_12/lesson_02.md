# 13.2 Layout Detection in Practice — PaddleOCR PP-StructureV3

## Problem

Layout parsing (Chapter 2.2) needs a concrete tool to segment a page into structural regions
(title, text block, table, figure) rather than individual words. This lesson documents the
working setup — including two real, non-obvious dependency pitfalls encountered in practice —
for **PaddleOCR's PP-StructureV3** pipeline.

## Known Pitfall 1: `PPStructure` Was Removed in PaddleOCR 3.x

Older tutorials and documentation reference `from paddleocr import PPStructure` — this class
was **removed entirely** in PaddleOCR 3.x in favor of a redesigned pipeline class,
`PPStructureV3`, with a different API shape. Attempting the old import raises:

```
ImportError: cannot import name 'PPStructure' from 'paddleocr'
```

**Fix — use the current API:**

```python
from paddleocr import PPStructureV3

pipeline = PPStructureV3(
    use_doc_orientation_classify=False,  # disable for speed if input is already upright
    use_doc_unwarping=False,             # disable for speed if input isn't perspective-distorted
    device="gpu"                          # or "cpu"
)

output = pipeline.predict(input=image_path)
for res in output:
    res.print()  # structured layout regions with type + bbox
    res.save_to_json(save_path="layout_output")
```

## Known Pitfall 2: Generic Dependency Error on Pipeline Creation

Even with the correct import, pipeline creation can fail with an unhelpful generic error:

```
RuntimeError: A dependency error occurred during pipeline creation. Please refer to the
installation documentation to ensure all required dependencies are installed.
```

This message hides the real cause: PaddleOCR/PaddleX split OCR-related dependencies into an
**optional `[ocr]` extra**, and this generic `RuntimeError` is a wrapper around a more specific
missing-dependency error underneath.

**Fix:**

```bash
pip install -q "paddlex[ocr]"
```

If the same generic error persists after this fix, a diagnostic call surfaces the real
underlying error rather than the generic wrapper:

```python
from paddlex.utils.deps import require_extra
require_extra("ocr", obj_name="PPStructureV3")
```

## Output Shape

Each detected region returns a **type** (e.g., `title`, `text`, `table`, `image`) and a
**bounding box** — structurally different from OCR's word-level output (Chapter 13.1): this is
region-level, not word-level, segmentation. This is the concrete implementation of the layout
signal referenced throughout Chapter 2.2 and Chapter 9's classification design.

## Trade-offs: Speed vs. Feature Completeness

| Setting | Gain | Cost |
|---|---|---|
| `use_doc_orientation_classify=False`, `use_doc_unwarping=False` | Faster inference — skips sub-modules not needed for already-upright, non-distorted scans | Loses automatic orientation/unwarping correction — only safe to disable if input quality is already reasonably controlled (e.g., not raw phone photos, per Chapter 2.3's preprocessing discussion) |
| Full pipeline (all sub-modules enabled, including table/formula recognition) | Handles a much broader range of document conditions with less upstream preprocessing needed | Slower per-page inference — a real throughput/cost trade-off at scale (Chapter 11.1) |

## When to Use / When Not To

- **Use PP-StructureV3** whenever layout-region signal (not just word-level OCR text) is
  needed — either as a classification feature (Chapter 2.2's early rule-based intuition) or as
  a component feeding a joint fusion model in a future architecture iteration.
- **Disable the orientation/unwarping sub-modules** specifically when input is known to be
  reasonably clean (e.g., scanner output, not raw phone photos) — re-enable them if photo-input
  volume is significant, consistent with the perspective-correction discussion in Chapter 2.3.

## Summary

PP-StructureV3 is PaddleOCR's current (3.x) layout-analysis pipeline — the older `PPStructure`
class no longer exists, and pipeline creation requires the optional `paddlex[ocr]` extra to be
installed explicitly, even though the error message that surfaces when it's missing doesn't say
so directly. Once running, it returns region-level types and boxes, structurally distinct from
and complementary to OCR's word-level output from Lesson 13.1.