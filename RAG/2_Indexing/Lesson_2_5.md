# Lesson 2.5 — Multimodal RAG: Vision-Language Models, Multimodal Embeddings, and Cross-Modal Retrieval

---

## What Multimodal RAG Means and Why It Matters

Standard RAG operates entirely in text space. Documents go in as text, chunks come out as text, embeddings represent text, retrieval returns text. This works until your documents contain information that fundamentally cannot be expressed in text — or where the visual form of the information is the information.

Consider these cases:
- A medical imaging report where the radiologist's findings reference specific regions of an X-ray.
- A financial presentation where a revenue trend is shown as a line chart with no accompanying table.
- An engineering manual where a wiring diagram is the only description of how components connect.
- A product catalog where images are the primary content and text descriptions are secondary.
- A scanned invoice where the spatial layout of fields (not their text content alone) defines their meaning.

In all these cases, a text-only RAG system either loses the information entirely or produces a degraded, text-only approximation. Multimodal RAG extends the system to handle images alongside text — at the retrieval level, not just the generation level.

There are three distinct architectures for multimodal RAG. Understanding the trade-offs between them is the core of this lesson.

---

## Architecture 1 — Caption-Based Multimodal RAG

This is the simplest approach and the one described in Lesson 2.4. Convert images to text descriptions using a vision model, then treat those descriptions as regular text chunks.

```
Image → Vision Model → Text Caption → Embed as Text → Store in Vector DB
```

At retrieval time, the caption is retrieved just like any text chunk. The LLM receives the caption as part of its context.

### When This Works Well

- Charts and diagrams where the key information can be accurately verbalized: "Revenue grew from $3.8B in FY2022 to $4.2B in FY2023, a 10.5% increase."
- Figures with clear, unambiguous visual content.
- Cases where the text caption is sufficient for the LLM to answer the user's question without seeing the original image.

### When This Fails

**Information loss in captioning.** A vision model captioning a complex circuit diagram cannot capture every component and connection in text. The caption describes the general purpose; the diagram contains the precise detail. If the user asks a detailed question about component placement, the caption is insufficient.

**Caption quality variance.** Vision models produce captions of inconsistent quality. Simple bar charts get accurate captions. Complex multi-layer diagrams, scientific figures with dense notation, or low-quality images produce vague or inaccurate captions. You have no way of knowing which captions are accurate without manual verification.

**The retrieval-generation gap.** The user's query retrieves the caption (text). The LLM sees the caption (text). The original image is never part of the pipeline. For queries where the user would benefit from seeing the actual image, or where the image contains detail the caption missed, this architecture fails.

### Implementation Pattern

```python
from openai import OpenAI
import base64

def caption_image(image_bytes: bytes, context: str = "") -> str:
    """
    Generate a detailed caption for an image using a vision model.
    context: surrounding text from the document (helps the model understand 
             what the figure is about)
    """
    client = OpenAI()
    
    image_b64 = base64.b64encode(image_bytes).decode()
    
    system_prompt = """You are a precise technical document analyst. 
    When describing charts and figures, always include:
    - Exact values and data points visible in the figure
    - Axis labels, units, and scales
    - Trends, patterns, and anomalies
    - The key insight the figure communicates
    Never say 'approximately' when exact values are readable."""
    
    messages = [{"role": "user", "content": [
        {"type": "image_url", 
         "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        {"type": "text", 
         "text": f"Document context: {context}\n\nDescribe this figure in detail."}
    ]}]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        system=system_prompt,
        max_tokens=500
    )
    
    return response.choices[0].message.content
```

**The context parameter is important.** Passing surrounding document text to the vision model significantly improves caption quality — the model understands what the figure is about and what details matter.

---

## Architecture 2 — Multimodal Embeddings

Instead of converting images to text, embed images and text in the same vector space. Both text chunks and images are embedded using a multimodal embedding model. At retrieval time, a text query can directly retrieve images (and vice versa).

```
Text chunk → Multimodal Embedder → Vector (shared space)
Image      → Multimodal Embedder → Vector (shared space)

Query (text) → Multimodal Embedder → Query Vector
             → Search shared space → Returns text chunks AND images
```

The key property: the embedding model maps text and images to the same latent space such that semantically related text and images land near each other. A query about "quarterly revenue growth" should retrieve both text paragraphs about revenue and charts showing revenue trends.

### Models That Enable This

**CLIP (Contrastive Language-Image Pretraining, OpenAI 2021):**
The foundational model for multimodal embeddings. Trained on 400M (text, image) pairs from the web using contrastive learning — text descriptions are pulled close to their corresponding images, pulled apart from unrelated images.

CLIP produces 512-dimensional vectors for both text and images in the same space. A text query "a dog running in a park" will have a vector close to images of dogs running in parks.

```python
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def embed_text_clip(text: str) -> list[float]:
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    # Normalize to unit sphere (important for cosine similarity)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features[0].tolist()

def embed_image_clip(image: Image.Image) -> list[float]:
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features[0].tolist()
```

**CLIP Limitations:**
- Trained on web images. Struggles with specialized domain images: medical scans, technical diagrams, scientific figures, engineering drawings. A query about "left ventricular hypertrophy" will not reliably retrieve echocardiogram images using CLIP.
- 512 dimensions is low compared to modern text embeddings. Representational capacity is limited.
- Text input is limited to 77 tokens — cannot handle long text descriptions.

**Newer multimodal embedding models:**

`nomic-embed-multimodal` — Open source, 768 dimensions, better document understanding than CLIP.

`voyage-multimodal-3` (Voyage AI) — Strong performance on document-heavy multimodal retrieval. 1024 dimensions.

`Amazon Titan Multimodal Embeddings` — 1024 dimensions, handles both image and text, available through Bedrock.

`Cohere Embed v3` with image support — Multilingual, strong document retrieval performance.

### Indexing Mixed Content

When your index contains both text chunks and image embeddings, you need to handle them uniformly in the vector database.

```python
# Qdrant supports storing both text and image vectors in the same collection
# by using the same vector dimension

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

client = QdrantClient(url="http://localhost:6333")

# Create collection with shared vector space
client.create_collection(
    collection_name="multimodal_docs",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)

# Index a text chunk
client.upsert(
    collection_name="multimodal_docs",
    points=[PointStruct(
        id="chunk-001",
        vector=embed_text(chunk_text),  # 768-dim text embedding
        payload={
            "content_type": "text",
            "text": chunk_text,
            "page_number": 5,
            "doc_id": "annual-report-2024"
        }
    )]
)

# Index an image
client.upsert(
    collection_name="multimodal_docs",
    points=[PointStruct(
        id="figure-001",
        vector=embed_image(image),  # 768-dim image embedding
        payload={
            "content_type": "image",
            "image_path": "s3://docs/figures/annual-report-2024-fig-3.png",
            "caption": "Q4 revenue breakdown by product line",
            "page_number": 12,
            "doc_id": "annual-report-2024"
        }
    )]
)
```

At retrieval time, a text query vector searches the same collection and returns both text chunks and image results ranked by cosine similarity. The LLM receives text chunks as text context, and image results are passed as actual images (if the LLM is multimodal) or as captions (if text-only).

### Sending Images to the LLM

When a multimodal query retrieves an image, pass it directly to a vision-capable LLM:

```python
from openai import OpenAI
import base64
import requests

def generate_with_mixed_context(
    query: str,
    text_chunks: list[str],
    image_paths: list[str]
) -> str:
    client = OpenAI()
    
    # Build message content with interleaved text and images
    content = [{"type": "text", "text": f"Question: {query}\n\nContext:"}]
    
    # Add text chunks
    for chunk in text_chunks:
        content.append({"type": "text", "text": chunk})
    
    # Add retrieved images
    for img_path in image_paths:
        if img_path.startswith("s3://"):
            image_bytes = download_from_s3(img_path)
        else:
            image_bytes = open(img_path, "rb").read()
        
        image_b64 = base64.b64encode(image_bytes).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_b64}"}
        })
    
    content.append({
        "type": "text", 
        "text": "Answer the question based on the provided context and images."
    })
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}]
    )
    
    return response.choices[0].message.content
```

---

## Architecture 3 — Late Multimodal Fusion (ColPali)

This is the most sophisticated approach and represents the current state of the art for document-heavy multimodal retrieval.

### The Problem with Architectures 1 and 2

Both Architecture 1 (captioning) and Architecture 2 (multimodal embeddings) convert images to a single vector before retrieval. This means:
- For captioning: all the visual detail is condensed into text, losing precision.
- For multimodal embeddings: the entire image is condensed into one vector, losing fine-grained visual detail.

Neither approach can answer the question "which page of this 100-page PDF contains the specific diagram showing the neural network architecture with five layers?" with high precision, because the single-vector representation of a complex diagram does not encode enough detail for precise matching.

### ColPali: Patch-Level Image Embeddings

ColPali (Faysse et al., 2024) applies the ColBERT insight (per-token embeddings) to document images. Instead of embedding an image as a single vector, it embeds each **visual patch** of the image as its own vector.

```
Image (1024×1024 pixels)
    ↓
Divide into 32×32 patches
    ↓
Vision transformer encodes each patch in context of all other patches
    ↓
Each patch → one 128-dim vector
    ↓
One image = 1024 patch vectors (for a 32×32 grid)
```

At retrieval time, the text query is also encoded into per-token vectors (using a language model). Relevance is computed using the same MaxSim operation as ColBERT: for each query token, find the best-matching image patch.

```
Score(query, image) = Σᵢ max_j cosine(query_token_i, image_patch_j)
```

This means: a query about "five-layer neural network" will score high against an image that has patches encoding neural network visual patterns in a five-layer structure, even if those exact words never appear in any text near the image.

### Why ColPali Is Powerful for Document Retrieval

ColPali was specifically designed for **document page retrieval** — finding the right page in a document corpus based on a natural language query. Documents are indexed as page images (no OCR needed), and the patch embeddings capture both the visual layout and the text rendered on the page (since text renders as recognizable visual patterns in patch embeddings from a vision transformer).

This means:
- OCR errors do not degrade retrieval — the model reads the rendered text visually.
- Visual layout is a retrieval signal — a query about a "table comparing features" will match pages with tabular layouts.
- Charts, diagrams, and mixed layouts are handled natively.

### ColPali in Practice

```python
# Using the vidore/colpali library
from colpali_engine.models import ColPali, ColPaliProcessor
from PIL import Image
import torch

model_name = "vidore/colpali-v1.2"
model = ColPali.from_pretrained(model_name, torch_dtype=torch.bfloat16)
processor = ColPaliProcessor.from_pretrained(model_name)

# Index a document page (no OCR needed — pass the page as an image)
def embed_page_colpali(page_image: Image.Image) -> torch.Tensor:
    """Returns a matrix of patch embeddings, not a single vector."""
    batch = processor.process_images([page_image])
    with torch.no_grad():
        embeddings = model(**batch)  # Shape: (1, num_patches, embedding_dim)
    return embeddings[0]  # Shape: (num_patches, embedding_dim)

# Encode a query
def embed_query_colpali(query: str) -> torch.Tensor:
    batch = processor.process_queries([query])
    with torch.no_grad():
        embeddings = model(**batch)  # Shape: (1, query_len, embedding_dim)
    return embeddings[0]  # Shape: (query_len, embedding_dim)

# Compute MaxSim score
def maxsim_score(query_embeddings: torch.Tensor, 
                 page_embeddings: torch.Tensor) -> float:
    # query_embeddings: (query_len, dim)
    # page_embeddings: (num_patches, dim)
    
    # Compute all pairwise cosine similarities
    scores = torch.einsum('qd,pd->qp', query_embeddings, page_embeddings)
    
    # MaxSim: for each query token, take max similarity over all patches
    max_scores = scores.max(dim=1).values
    
    # Sum over query tokens
    return max_scores.sum().item()
```

### Storage Requirements

ColPali stores one embedding vector per patch per page. For a 1024×1024 image divided into 32×32 patches = 1024 patches × 128 dimensions × 4 bytes = 512KB per page. For a 10,000-page corpus, that is ~5GB. Manageable, but significantly more than single-vector approaches.

### Which Architecture to Choose

| | Caption-Based | Multimodal Embeddings | ColPali |
|---|---|---|---|
| **Retrieval method** | Text search over captions | Cross-modal vector search | Patch-level MaxSim |
| **OCR required** | Yes | No | No |
| **Handles complex diagrams** | Poor (caption loses detail) | Moderate | Strong |
| **Storage per image** | ~1KB (caption text) | ~4KB (single vector) | ~500KB (patch vectors) |
| **Index time cost** | High (vision API per image) | Low-moderate | Moderate |
| **Query time cost** | Standard | Standard | Higher (MaxSim over patches) |
| **Best for** | Charts with numeric data, captionable figures | Product images, photos, general mixed content | Dense document pages, technical diagrams, mixed layout pages |

---

## Practical Multimodal RAG Pipeline Design

For most enterprise document RAG systems, the right approach is a **hybrid pipeline** that applies different strategies based on content type:

```
Document page
    ↓
Content type detection (text-heavy, image-heavy, mixed)
    ↓
Text-heavy pages → Standard text chunking + text embedding
Image-heavy pages → ColPali page embedding OR caption + text embedding  
Mixed pages → Both: text chunks + figure captions
    ↓
All embeddings stored in same collection with content_type metadata
    ↓
At retrieval: search across all content types, route results appropriately
    ↓
Text results → LLM context as text
Image results → LLM context as images (if vision-capable LLM) or captions
```

### Decision Logic for Content Type

```python
def classify_page_content(page_image: Image, extracted_text: str) -> str:
    """
    Determine the dominant content type of a page.
    Returns: 'text_dominant', 'image_dominant', 'mixed'
    """
    page_area = page_image.width * page_image.height
    
    # Estimate text coverage from extracted text length
    words = len(extracted_text.split())
    text_score = min(words / 200, 1.0)  # normalize: 200 words = full text page
    
    # Estimate image coverage from image detection
    # (simplified: count non-white pixel ratio)
    import numpy as np
    img_array = np.array(page_image.convert('L'))
    non_white_ratio = (img_array < 240).mean()
    image_score = max(0, non_white_ratio - text_score * 0.3)
    
    if text_score > 0.6 and image_score < 0.3:
        return 'text_dominant'
    elif image_score > 0.5 and text_score < 0.3:
        return 'image_dominant'
    else:
        return 'mixed'
```

---

## Handling Specific Multimodal Document Types

### Medical Imaging Documents

Reports reference images (X-rays, MRIs, CT scans). The text and image together form the complete clinical record.

Strategy:
- Index report text normally.
- Index images using a domain-specific multimodal model (BioViL-T, MedCLIP, or fine-tuned CLIP on medical imagery — general CLIP performs poorly on medical scans).
- Link image embeddings to their report text via `doc_id` and `page_number` metadata.
- At retrieval time, when a report text chunk is retrieved, also retrieve its associated images.

### Engineering and Technical Diagrams

Wiring diagrams, CAD drawings, process flow diagrams. Single-vector image embeddings are too coarse. ColPali patch embeddings work better because they can distinguish specific components at the patch level.

Strategy:
- Use ColPali for page-level retrieval.
- For very technical diagrams where even ColPali is insufficient, add structured metadata extracted by a domain expert or specialized model: component types, connection topology, annotated regions.

### Product Catalogs and E-commerce

Images are the primary content. Text descriptions are secondary. Users query by visual characteristics ("blue ceramic vase with geometric pattern").

Strategy:
- Multimodal embeddings (CLIP or better) are the right fit here — the visual content is what makes products distinct.
- Text queries and image queries (reverse image search) both need to work.
- Fine-tune CLIP on your product category for significant quality improvement.

---

## Evaluation for Multimodal RAG

Standard RAG evaluation metrics (discussed in Part 6) measure text quality. For multimodal RAG, you also need:

**Cross-modal retrieval accuracy:** Given a text query, does the system retrieve the correct image? Measure Recall@K where K = 1, 5, 10.

**Caption quality:** For Architecture 1, measure the accuracy of generated captions. BLEU and ROUGE are not appropriate for captions — use human evaluation or ask a vision model to score how accurately the caption describes the image.

**Visual grounding:** When the LLM references information from a retrieved image, is it accurately reading the image? Spot-check by asking "where in the image does it say X?" and verifying the LLM's claim.

---

## Summary

- Multimodal RAG extends retrieval to images alongside text. Three architectures exist, each with different trade-offs.
- **Caption-based** (Architecture 1): Simple, works well for charts with readable data, loses visual detail for complex diagrams.
- **Multimodal embeddings** (Architecture 2): Direct text-to-image and image-to-text retrieval using shared embedding space. CLIP is the baseline; domain-specific fine-tuning often necessary.
- **ColPali** (Architecture 3): Per-patch image embeddings with MaxSim retrieval. Best precision for dense document pages and technical diagrams. Higher storage cost.
- For most enterprise document systems, a hybrid pipeline is best: text-dominant pages → text chunking, image-heavy pages → multimodal embeddings or ColPali, figures within text pages → captions + text embedding.
- Domain matters heavily for multimodal models. General CLIP performs poorly on medical, engineering, or scientific domain images. Fine-tuning or domain-specific models are necessary.
- Always pass retrieved images directly to a vision-capable LLM rather than only passing captions when image fidelity matters for the answer.

---

## What's Next

Lesson 2.6 covers incremental indexing and data freshness strategies — how to keep your index current as documents change, without re-indexing everything from scratch.