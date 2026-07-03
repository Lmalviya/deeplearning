# 13.4 LayoutLMv3 as a Frozen Embedding Extractor

## Problem

Chapter 2.3 and Chapter 9 rely on a frozen embedding backbone feeding a prototype/KNN
classifier — this lesson documents the concrete, working difference between using LayoutLMv3
the *wrong* way (as a fine-tuned fixed-class classifier, which contradicts the extensibility
architecture) and the *right* way (as a frozen feature extractor) for this system.

## Known Pitfall: Missing Tesseract Dependency

`LayoutLMv3Processor` runs OCR internally (via Tesseract, by default) to obtain word boxes.
Without the underlying Tesseract binary installed, this raises:

```
ImportError: LayoutLMv3ImageProcessor requires the PyTesseract library but it was not found
in your environment.
```

**Fix — install both the system binary and the Python wrapper:**

```bash
apt-get install -y tesseract-ocr -q
pip install -q pytesseract
```

A runtime restart may be needed afterward so the process picks up the newly-installed binary on
`PATH`.

## The Critical Distinction: `ForSequenceClassification` vs. `Model`

**This is the single most important practical point in this lesson.** Two different classes
from the same `transformers` library produce architecturally different things:

```python
# WRONG for this system's architecture — attaches a NEW, RANDOMLY-INITIALIZED
# classification head with a fixed num_labels. Contradicts Chapter 2.3/9's
# extensibility requirement entirely — this head has to be trained, and adding
# a class means growing and retraining it (exactly the Chapter 9.1 problem).
from transformers import LayoutLMv3ForSequenceClassification
model = LayoutLMv3ForSequenceClassification.from_pretrained(
    "microsoft/layoutlmv3-base", num_labels=5
)
# model(...).logits is meaningless until this head is trained — num_labels=5 only
# configures architecture shape, it carries zero learned semantic information.
```

```python
# RIGHT for this system's architecture — loads only the backbone, no classification
# head at all. Used purely as a frozen feature extractor.
from transformers import LayoutLMv3Processor, LayoutLMv3Model
import torch

processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=True)
model = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base").to("cuda")
model.eval()  # frozen — no training happens

def get_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
    encoding = processor(img, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**encoding)
    # Mean-pool the token+patch sequence into one document-level vector
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
```

## Building the Prototype/KNN Classifier

```python
import numpy as np
from pathlib import Path

class_folders = {"invoice": ..., "resume": ..., "receipt": ..., "id_document": ..., "contract": ...}
REF_SAMPLES_PER_CLASS = 10
prototypes = {}

for class_name, folder in class_folders.items():
    image_paths = list(Path(folder).glob("*.jpg"))[:REF_SAMPLES_PER_CLASS]
    embeddings = [get_embedding(str(p)) for p in image_paths]
    prototypes[class_name] = np.mean(embeddings, axis=0)  # centroid/prototype method
```

**Prototype (centroid) vs. true KNN — a real distinction, not interchangeable terms:**
averaging all reference examples into one vector per class (above) is the *prototype* method —
cheap, but can land in a "blurry middle" if a class secretly contains multiple visually distinct
sub-styles. **True KNN** keeps every individual reference embedding (no averaging) and votes
among the k nearest at inference time — more robust to within-class sub-style variation,
at the cost of comparing against more vectors per inference (still cheap at the reference-set
sizes discussed in Chapter 9.3).

```python
def classify_topk_knn(image_path, reference_embeddings, reference_labels, k=5):
    emb = get_embedding(image_path)
    sims = np.array([
        np.dot(emb, ref) / (np.linalg.norm(emb) * np.linalg.norm(ref))
        for ref in reference_embeddings
    ])
    top_k_idx = np.argsort(-sims)[:k]
    top_k_labels = [reference_labels[i] for i in top_k_idx]
    top_k_sims = sims[top_k_idx]

    from collections import defaultdict
    class_scores = defaultdict(float)
    for label, sim in zip(top_k_labels, top_k_sims):
        class_scores[label] += sim
    total = sum(class_scores.values())
    return sorted(((c, s / total) for c, s in class_scores.items()), key=lambda x: -x[1])
```

**Note on confidence sharpening:** raw cosine similarities tend to bunch tightly (e.g.,
0.70–0.95), producing washed-out, near-uniform confidence scores if converted to probabilities
naively. Dividing by a small temperature value before a softmax-style normalization (e.g.,
`scaled = sims / 0.05`) sharpens the gap between the top match and the rest — a real,
tunable hyperparameter validated against held-out data, not a fixed constant.

## Trade-offs

| Choice | Gain | Cost |
|---|---|---|
| `LayoutLMv3Model` (frozen extractor) — chosen | Satisfies the zero-downtime, no-retraining extensibility requirement (Chapter 2.3, Chapter 9) directly | Generally lower raw accuracy than a fully fine-tuned fixed-class model, per the domain-adaptation trade-offs (earlier ML-design phase) |
| Prototype/centroid method | Cheapest to compute and store (one vector per class) | Vulnerable to within-class sub-style blurring |
| True KNN (no averaging) | More robust to sub-style variation within a class | More reference vectors to compare per inference — still trivial at the reference-set sizes discussed in Chapter 9.3, but not free |

## Summary

`LayoutLMv3Model` (the bare backbone), not `LayoutLMv3ForSequenceClassification` (which attaches
an untrained, fixed-size classification head), is the correct class to load for this system's
frozen-embedding architecture — using the wrong one silently reintroduces the exact
retraining-to-add-a-class problem the whole architecture was designed to avoid. True KNN over a
kept reference set is more robust than the cheaper prototype/centroid averaging method when a
class has meaningful internal visual variation, and confidence scores need explicit temperature
sharpening before they're usable as a real decision signal.