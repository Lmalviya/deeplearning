# 13.5 Data Sourcing Reference

## Problem

Building reference sets and validation data for the five phase-1 classes (Invoice, Contract,
Resume, ID Document, Receipt) requires real datasets — but the most obvious source
(`aharley/rvl_cdip` on Hugging Face) hits a hard compatibility wall, and several other
candidate sources have their own non-obvious gotchas. This lesson is a concrete reference for
what actually worked, per class.

## Known Pitfall: Hugging Face Dataset-Script Deprecation

The canonical RVL-CDIP dataset repo (`aharley/rvl_cdip`) uses an old-style **loading script**
(arbitrary Python code defining how to build the dataset), and current versions of the
`datasets` library **no longer support loading-script-based datasets**:

```
RuntimeError: Dataset scripts are no longer supported, but found rvl_cdip.py
```

**Fix — use a parquet-based mirror instead**, which sidesteps the loading-script mechanism
entirely:

```python
from datasets import load_dataset
ds = load_dataset("chainyo/rvl-cdip", split="train", streaming=True)
# ds.features['label'].names gives the class-name-to-int mapping directly
```

**General lesson, applicable beyond RVL-CDIP:** when a Hugging Face dataset repo throws this
specific error, search for a parquet-format mirror of the same underlying data before assuming
the dataset is unusable — this pattern (a maintained parquet mirror existing alongside a
deprecated scripted original) is common, not a one-off.

## Per-Class Data Sources Used

| Class | Source | Access method | Notes |
|---|---|---|---|
| Invoice, Resume | `chainyo/rvl-cdip` (Hugging Face, parquet) | `load_dataset(..., streaming=True)`, filter by label index, save ~150 images/class | Streaming avoids downloading the full 400K-image dataset when only a small reference set is needed |
| Receipt | `jsdnrs/ICDAR2019-SROIE` (Hugging Face) | `load_dataset("jsdnrs/ICDAR2019-SROIE", split="all")` | Explicitly documented by the maintainer as the recommended loading method — no script issues |
| ID Document | `unidpro/synthetic-printed-usa-passports-dataset` (Kaggle) | `kagglehub.dataset_download("unidpro/synthetic-printed-usa-passports-dataset")` | Synthetic data — no privacy/copyright concern; using the exact dataset slug via `kagglehub` bypasses Kaggle's sometimes-unreliable "Add Data" UI search entirely |
| Contract | CUAD (Contract Understanding Atticus Dataset) text, rendered to synthetic page images | `load_dataset("theatticusproject/cuad-qa", split="train")`, then render each unique passage as a grayscale page image via PIL | No ready-made contract *image* dataset exists publicly — this is a deliberate synthetic-rendering workaround, with a known train/production distribution gap (no scan noise, skew, or font variation) accepted for phase-1 speed |
| Handwriting (HTR validation) | `Teklia/IAM-line` (Hugging Face, parquet) | `load_dataset("Teklia/IAM-line", split="test")` | Script-free, avoids the same loading-script issue other IAM mirrors on Hugging Face can hit |

## Contract-Class Rendering Approach (Since No Ready-Made Dataset Exists)

```python
from PIL import Image, ImageDraw, ImageFont
import textwrap

def render_text_as_page(text, out_path, width=1000, height=1300):
    img = Image.new("L", (width, height), color=255)  # grayscale — matches other classes' scan character
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 18)
    margin, y = 60, 60
    for line in textwrap.wrap(text, width=85):
        if y > height - margin:
            break
        draw.text((margin, y), line, fill=0, font=font)
        y += 24
    img.save(out_path)
```

**Deliberate choice: grayscale output.** Matches the visual character of the other four
classes' scanned-document sources, avoiding an accidental "contract = the only color-image
class" shortcut that a classifier could latch onto and that wouldn't hold in production.

**Deduplication note:** CUAD reuses the same context passage across multiple QA pairs — filter
for uniqueness and a minimum passage length (e.g., 300+ characters) before rendering, or the
reference set ends up with near-duplicate or trivially-short "documents."

## Trade-offs

| Choice | Gain | Cost |
|---|---|---|
| Streaming + filtering large datasets (RVL-CDIP) rather than full download | Only pulls the small reference-set volume actually needed, fast | Streaming iteration order depends on the dataset's shard layout — verify class balance in the pulled sample rather than assuming it |
| `kagglehub` with an exact dataset slug, rather than Kaggle's "Add Data" search UI | Reliable, scriptable, bypasses UI search inconsistencies | Requires knowing the exact slug in advance (found via a web search of the dataset name, not guessed) |
| Synthetic rendering for the Contract class | Unblocks phase-1 entirely despite no ready-made image dataset existing | Known, accepted distribution gap versus real scanned contracts (no noise, skew, font variety) — the first place to look if real-world contract classification underperforms |

## Summary

RVL-CDIP's canonical Hugging Face repo is unusable with current `datasets` library versions due
to the loading-script deprecation — a parquet mirror (`chainyo/rvl-cdip`) is the working
substitute, and this same "look for a parquet mirror" pattern applies to other
loading-script-based datasets encountered later. Receipt and handwriting-validation data came
from clean, ready-to-use Hugging Face datasets; ID-document data came from a synthetic Kaggle
dataset pulled reliably via `kagglehub`'s exact-slug download rather than the "Add Data" search
UI; and Contract data — for which no ready-made image dataset exists — was synthesized by
rendering CUAD's real contract text as grayscale page images, a deliberate, documented
distribution-gap trade-off accepted for phase-1 speed.