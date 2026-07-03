# 13.3 Handwriting Recognition in Practice — TrOCR

## Problem

Extraction Service's printed-vs-handwritten fork (Chapter 2.2) routes handwritten pages to a
dedicated HTR engine, never to printed-text OCR. This lesson documents a working HTR setup
(**TrOCR**) and demonstrates concretely — not just theoretically — why printed OCR is unsafe on
handwriting.

## Solution / Concept: Working Setup and a Direct Comparison

**Sourcing a real handwritten sample** without hitting the same Hugging Face
loading-script deprecation issue covered in Lesson 13.5 — `Teklia/IAM-line` is a script-free,
parquet-based mirror of the IAM Handwriting Database:

```python
from datasets import load_dataset

iam_ds = load_dataset("Teklia/IAM-line", split="test")
sample = iam_ds[0]
hw_img = sample['image']       # PIL Image
ground_truth = sample['text']  # known-correct transcription
```

**Running plain printed-text OCR on it (expect this to struggle):**

```python
import easyocr
reader = easyocr.Reader(['en'], gpu=True)

hw_img.save("/tmp/handwritten_sample.png")
easyocr_result = reader.readtext("/tmp/handwritten_sample.png")
print("EasyOCR output:", [r[1] for r in easyocr_result])
```

**Running TrOCR (a real HTR model) on the same image:**

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to("cuda")

pixel_values = processor(images=hw_img, return_tensors="pt").pixel_values.to("cuda")
generated_ids = model.generate(pixel_values)
trocr_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("TrOCR output:", trocr_text)
```

**Comparing all three** — ground truth, EasyOCR's output, and TrOCR's output — makes the
Chapter 2.2 warning concrete rather than theoretical: EasyOCR (a printed-text engine) typically
produces garbled or empty output on handwriting, while TrOCR produces output close to the
ground truth. This is the direct evidence behind the architectural rule from Chapter 2.2: never
fall back to printed OCR on a page identified as handwritten — it doesn't fail safely, it fails
*confidently and wrong*.

## Trade-offs

| Approach | Gain | Cost |
|---|---|---|
| TrOCR (transformer-based, end-to-end sequence model) | Strong accuracy on genuine handwriting, straightforward Hugging Face integration | Heavier model than classic OCR — real GPU inference cost, relevant to the Chapter 11.1 cost breakdown |
| Classic CRNN + CTC-loss HTR architectures | Can be lighter-weight than a full transformer | Generally requires more custom setup/training infrastructure than a ready-to-use pretrained Hugging Face model |

## When to Use / When Not To

- **Use TrOCR (or an equivalent dedicated HTR model)** for any page routed by the
  printed-vs-handwritten classifier (Chapter 2.2) to the handwritten branch — never printed OCR.
- **Never treat printed-OCR's output on a handwritten page as a degraded-but-usable signal** —
  the comparison in this lesson shows the failure mode is confidently wrong text, not a
  gracefully lower-confidence version of the correct text, so there's no safe way to salvage it.

## Summary

TrOCR provides a working, easily-integrated HTR path via Hugging Face's `transformers` library,
sourced against handwriting samples from a script-free IAM mirror (`Teklia/IAM-line`) to avoid
the dataset-loading pitfalls covered in Lesson 13.5. Running the same image through both
EasyOCR and TrOCR side by side is the concrete demonstration behind Chapter 2.2's rule: printed
OCR must never be used as a fallback for handwritten content, since it fails confidently and
incorrectly rather than gracefully.