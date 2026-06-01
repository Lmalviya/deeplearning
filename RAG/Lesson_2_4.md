# Lesson 2.4 — Document Pre-processing Pipelines: OCR, Layout Parsing, and Table Extraction

---

## Why Pre-processing Is Where Most RAG Systems Secretly Fail

When a RAG system gives wrong answers, the debugging instinct is to look at retrieval algorithms, re-ranking, or the LLM prompt. These are the visible, tuneable parts. But a large fraction of production RAG failures trace back to something that happened much earlier: the document was parsed incorrectly, and the information the user is asking about was either lost, garbled, or stored in an unusable form.

Pre-processing is the most unglamorous part of RAG. It is also where the most silent failures happen. The system does not crash — it just indexes garbage and confidently retrieves garbage.

This lesson goes deep into the pre-processing pipeline: what the real challenges are, which tools handle them, and how to design a pipeline that is robust across the messy reality of real-world documents.

---

## The Pre-processing Pipeline

The goal of pre-processing is to convert raw documents in any format into clean, structured text that is ready for chunking and embedding. The pipeline has these stages:

```
Raw document (PDF, DOCX, HTML, image, scan...)
    ↓
Format detection
    ↓
Content extraction (text, tables, images, metadata)
    ↓
Layout analysis (reading order, structure identification)
    ↓
OCR (for image-based content)
    ↓
Table parsing and normalization
    ↓
Cleaning and normalization
    ↓
Output: structured text + extracted tables + figure captions + metadata
```

Each stage has tools, failure modes, and design decisions. We go through all of them.

---

## Stage 1 — Format Detection

Before parsing, identify what you are dealing with. Do not trust the file extension — a `.pdf` extension on a file does not tell you whether it contains embedded text or is a scanned image. A `.docx` may contain embedded PDFs. An `.html` file may be a JavaScript-rendered page with no static content.

```python
import magic  # python-magic library

def detect_format(file_path: str) -> dict:
    mime_type = magic.from_file(file_path, mime=True)
    
    # For PDFs specifically, detect if text-based or image-based
    if mime_type == "application/pdf":
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        
        text_pages = 0
        image_pages = 0
        
        for page in doc:
            text = page.get_text().strip()
            if len(text) > 50:  # Meaningful text content
                text_pages += 1
            else:
                image_pages += 1
        
        pdf_type = "digital" if text_pages > image_pages else "scanned"
        return {"mime_type": mime_type, "pdf_type": pdf_type, 
                "text_pages": text_pages, "image_pages": image_pages}
    
    return {"mime_type": mime_type}
```

Mixed PDFs (some digital pages, some scanned) are common in enterprise settings — a report where the main body is digital but appendices are scanned attachments. Your pipeline needs to handle per-page routing, not just per-document routing.

---

## Stage 2 — Text Extraction from Digital PDFs

For PDFs with embedded text, you have several parser options. They are not equivalent.

### PyMuPDF (fitz)

The fastest and most reliable for standard text extraction. Preserves text with reasonable reading order. Good Unicode handling.

```python
import fitz

def extract_text_pymupdf(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages = []
    
    for page_num, page in enumerate(doc):
        # Extract text blocks with position information
        blocks = page.get_text("blocks")  # returns (x0, y0, x1, y1, text, block_no, block_type)
        
        # Sort by vertical position (reading order)
        blocks_sorted = sorted(blocks, key=lambda b: (b[1], b[0]))
        
        page_text = "\n".join([b[4].strip() for b in blocks_sorted if b[4].strip()])
        
        pages.append({
            "page_number": page_num + 1,
            "text": page_text,
            "width": page.rect.width,
            "height": page.rect.height
        })
    
    return pages
```

**Limitation:** Multi-column layouts. PyMuPDF extracts text in the order it appears in the PDF's internal structure, which for multi-column layouts may interleave left and right column text rather than reading column-by-column.

### pdfplumber

Built on pdfminer, slower than PyMuPDF but better at extracting structured table data. The go-to when tables are important.

```python
import pdfplumber

def extract_with_pdfplumber(pdf_path: str) -> list[dict]:
    pages = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # Extract tables separately
            tables = page.extract_tables()
            
            # Extract text (with table regions cropped out to avoid duplication)
            if tables:
                # Crop out table regions before text extraction
                table_bboxes = [table.bbox for table in page.find_tables()]
                text_page = page
                for bbox in table_bboxes:
                    text_page = text_page.outside_bbox(bbox)
                text = text_page.extract_text() or ""
            else:
                text = page.extract_text() or ""
            
            pages.append({
                "page_number": page_num + 1,
                "text": text,
                "tables": tables  # list of 2D arrays (rows × columns)
            })
    
    return pages
```

### Multi-column Layout Handling

Multi-column PDFs are one of the hardest layout challenges. The naive approach interleaves columns. The correct approach: detect column boundaries and read each column top-to-bottom before moving to the next.

```python
import fitz

def extract_multicolumn(page) -> str:
    """Detect columns and extract text in correct reading order."""
    blocks = page.get_text("blocks")
    
    if not blocks:
        return ""
    
    page_width = page.rect.width
    
    # Estimate number of columns by clustering x-coordinates of block starts
    x_starts = [b[0] for b in blocks]
    
    # Simple 2-column detection: if there are blocks clustered around
    # both x < page_width/2 and x > page_width/2, it's 2-column
    left_blocks = [b for b in blocks if b[0] < page_width * 0.5]
    right_blocks = [b for b in blocks if b[0] >= page_width * 0.5]
    
    if len(right_blocks) > len(blocks) * 0.2:  # Significant right-column content
        # Sort each column top-to-bottom, then concatenate
        left_text = "\n".join(sorted([b[4] for b in left_blocks], 
                                      key=lambda b: b))
        right_text = "\n".join(sorted([b[4] for b in right_blocks], 
                                       key=lambda b: b))
        return left_text + "\n\n" + right_text
    else:
        # Single column
        return "\n".join([b[4] for b in sorted(blocks, key=lambda b: (b[1], b[0]))])
```

For complex layouts, dedicated layout analysis tools (described below) are more reliable than heuristics.

---

## Stage 3 — OCR for Scanned Documents

When a page has no embedded text (or very little), you need OCR to extract text from the page image.

### The OCR Quality Ladder

OCR tools vary enormously in quality, especially for complex layouts:

**Tesseract (open source):**
- Decent quality for clean, standard fonts on white backgrounds.
- Struggles with handwriting, low-resolution scans, unusual fonts, and complex layouts.
- Free, self-hosted, no data leaves your infrastructure.
- Good enough for internal documents with reasonable scan quality.

```python
import pytesseract
from PIL import Image
import fitz

def ocr_page_tesseract(pdf_path: str, page_num: int) -> str:
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    # Render page to high-resolution image (300 DPI for good OCR quality)
    mat = fitz.Matrix(300/72, 300/72)  # 300 DPI
    pix = page.get_pixmap(matrix=mat)
    
    # Convert to PIL Image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # Run Tesseract
    text = pytesseract.image_to_string(img, lang='eng', config='--psm 6')
    return text
```

**AWS Textract:**
- Significantly better than Tesseract for complex layouts, tables, and forms.
- Returns structured output: text blocks with bounding boxes, tables as proper row-column structures, form fields as key-value pairs.
- Managed API — no infrastructure to run, data goes to AWS.
- Cost: ~$1.50 per 1000 pages for text detection, ~$15 per 1000 pages for table/form analysis.

```python
import boto3

def ocr_with_textract(image_bytes: bytes) -> dict:
    textract = boto3.client('textract', region_name='us-east-1')
    
    response = textract.analyze_document(
        Document={'Bytes': image_bytes},
        FeatureTypes=['TABLES', 'FORMS']  # Also extract tables and form fields
    )
    
    # Response contains Blocks: each block is a word, line, table, cell, etc.
    # Parse the block structure to reconstruct text and tables
    return parse_textract_response(response['Blocks'])
```

**Google Document AI:**
- Comparable to Textract in quality. Specialized processors for invoices, contracts, identity documents.
- Better for multilingual documents.

**Azure Document Intelligence:**
- Strong for specific document types. Pre-built models for invoices, receipts, tax forms, business cards.
- When your documents fit a pre-built model type, it is the fastest path to structured extraction.

### Image Pre-processing Before OCR

OCR quality is heavily affected by image quality. Pre-processing significantly improves results on low-quality scans.

```python
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np

def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    # Convert to grayscale
    image = image.convert('L')
    
    # Increase contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    
    # Binarize (convert to black and white)
    # Otsu's thresholding is better than a fixed threshold
    img_array = np.array(image)
    threshold = img_array.mean()
    binary = img_array > threshold
    image = Image.fromarray((binary * 255).astype(np.uint8))
    
    # Deskew: correct rotation
    # Use pytesseract's OSD to detect skew angle
    import pytesseract
    osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
    angle = osd.get('rotate', 0)
    
    if abs(angle) > 0.5:  # Only rotate if significant skew
        image = image.rotate(angle, expand=True, fillcolor=255)
    
    # Denoise
    image = image.filter(ImageFilter.MedianFilter(size=3))
    
    return image
```

The order matters: grayscale → contrast → binarize → deskew → denoise.

---

## Stage 4 — Layout Analysis

Layout analysis goes beyond text extraction. It identifies the structural role of each region on the page: is this a heading, a paragraph, a table, a figure, a footer, a sidebar? This information guides chunking and prevents mixing structural elements that should be treated differently.

### Document Layout Models

**LayoutParser** with PaddleOCR or Detectron2 models:
- Detects regions on a page (text, title, table, figure, list) using computer vision.
- Returns bounding boxes with region type labels.
- Open source, self-hosted.

```python
import layoutparser as lp
import cv2

def analyze_layout(image_path: str) -> list[dict]:
    model = lp.Detectron2LayoutModel(
        config_path='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
    )
    
    image = cv2.imread(image_path)
    layout = model.detect(image)
    
    regions = []
    for block in layout:
        regions.append({
            "type": block.type,          # Text, Title, Table, Figure, List
            "bbox": block.block.coordinates,  # (x1, y1, x2, y2)
            "score": block.score         # Detection confidence
        })
    
    return regions
```

**Unstructured.io:**
A practical library that combines layout analysis, OCR routing, and content extraction into a single pipeline. It automatically detects document type, routes image pages to OCR, identifies tables and titles, and returns structured elements.

```python
from unstructured.partition.auto import partition

elements = partition(filename="document.pdf")

for element in elements:
    print(type(element).__name__, ":", element.text[:100])
    # Output types: Title, NarrativeText, Table, ListItem, Header, Footer, Image
```

Unstructured is one of the best practical choices for a general-purpose pre-processing pipeline because it handles the routing logic automatically. It supports PDF, DOCX, HTML, PPTX, images, and many other formats.

---

## Stage 5 — Table Extraction and Normalization

Tables are the hardest content type in document pre-processing. This deserves detailed treatment.

### Why Tables Are Hard

A table encodes information in two-dimensional space. The meaning of any cell depends on its row and column headers. When you linearize a table into text (which is what you must do to embed it), you must somehow preserve those two-dimensional relationships.

Consider this balance sheet table:

| | FY2023 | FY2022 |
|---|---|---|
| Revenue | $4.2B | $3.8B |
| Operating Income | $820M | $710M |
| Net Income | $601M | $522M |

Naive extraction produces: "Revenue $4.2B $3.8B Operating Income $820M $710M Net Income $601M $522M"

This is unusable. You cannot answer "what was net income in FY2022?" from this text.

### Approach 1: Markdown Table Serialization

Convert the table to Markdown format. This preserves row-column relationships in a format the LLM can parse.

```python
def table_to_markdown(table: list[list[str]]) -> str:
    """Convert a 2D table array to Markdown table format."""
    if not table or not table[0]:
        return ""
    
    # First row is headers
    headers = table[0]
    rows = table[1:]
    
    # Build markdown
    header_row = "| " + " | ".join(str(h) for h in headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_rows = ["| " + " | ".join(str(cell) for cell in row) + " |" 
                 for row in rows]
    
    return "\n".join([header_row, separator] + data_rows)
```

Resulting chunk:
```
| | FY2023 | FY2022 |
|---|---|---|
| Revenue | $4.2B | $3.8B |
| Operating Income | $820M | $710M |
| Net Income | $601M | $522M |
```

This embeds reasonably well and the LLM can read it. The main drawback: tables with many rows produce large chunks with diffuse embeddings.

### Approach 2: Natural Language Serialization

Convert each row into a natural language sentence. This produces much better embeddings for retrieval because the text resembles how people ask questions.

```python
def table_to_natural_language(table: list[list[str]], 
                               table_title: str = "") -> list[str]:
    """Convert table rows to natural language sentences for embedding."""
    if not table or len(table) < 2:
        return []
    
    headers = table[0]
    rows = table[1:]
    sentences = []
    
    for row in rows:
        # Create a sentence for each row
        parts = []
        row_label = row[0] if row else ""
        
        for col_idx, (header, value) in enumerate(zip(headers[1:], row[1:]), 1):
            if value and value.strip():
                parts.append(f"{header}: {value}")
        
        if parts:
            sentence = f"{table_title + ': ' if table_title else ''}"
            sentence += f"{row_label} — " + ", ".join(parts)
            sentences.append(sentence)
    
    return sentences
```

Output for the balance sheet:
```
Financial Summary: Revenue — FY2023: $4.2B, FY2022: $3.8B
Financial Summary: Operating Income — FY2023: $820M, FY2022: $710M
Financial Summary: Net Income — FY2023: $601M, FY2022: $522M
```

Each row becomes a separately embeddable chunk. A query "what was net income in FY2022?" will now retrieve the Net Income row directly.

### Approach 3: Hybrid — NL for Embedding, Markdown for Generation

This is the best of both worlds:
- Embed the natural language serialization (better retrieval).
- Store the Markdown table in metadata (better LLM reading).
- When a table row is retrieved, pass the full Markdown table to the LLM instead of just the row.

```python
{
  "text": "Financial Summary: Net Income — FY2023: $601M, FY2022: $522M",  # for embedding
  "metadata": {
    "content_type": "table_row",
    "table_markdown": "| | FY2023 | FY2022 |\n|---|---|---|\n...",  # full table
    "table_title": "Financial Summary",
    "row_label": "Net Income"
  }
}
```

At retrieval time, when a table row chunk is returned, substitute its `table_markdown` field as the context for the LLM rather than just the row text.

### Handling Complex Table Structures

**Merged cells:** Many tables have merged cells (a header spanning multiple columns, or a row label spanning multiple rows). Most parsers struggle with these. Textract and Document AI handle them better than pdfplumber.

Strategy: when a merged cell is detected, repeat the merged header/label in each cell it covers during serialization.

**Multi-page tables:** A table that spans two pages is split across page boundaries during extraction. Detection: if the last row of a page and the first row of the next page have the same column count and no header row, they are likely continuation rows of the same table. Merge them.

**Nested tables:** Rare but they exist — a table cell that contains another table. Extract the inner table separately and reference it from the outer table's cell.

---

## Stage 6 — Figure and Chart Extraction

Charts, diagrams, and figures are completely invisible to text extraction. The information in a revenue trend chart or a market share diagram is not in any text layer — it exists only as an image.

### Extraction Pipeline for Figures

```python
import fitz
import base64
from openai import OpenAI

def extract_and_caption_figures(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    client = OpenAI()
    figures = []
    
    for page_num, page in enumerate(doc):
        # Get all images on this page
        image_list = page.get_images(full=True)
        
        for img_idx, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            
            # Skip small images (likely icons, bullets, logos)
            if base_image["width"] < 200 or base_image["height"] < 200:
                continue
            
            image_bytes = base_image["image"]
            image_b64 = base64.b64encode(image_bytes).decode()
            
            # Use vision model to generate caption
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": """Describe this chart or figure in detail. Include:
1. What type of visualization this is (bar chart, line chart, pie chart, diagram, etc.)
2. What the axes represent (if applicable) with units
3. The key data points, values, and trends shown
4. The main insight or conclusion this figure communicates
Be specific about numbers and time periods. If this is not a data visualization 
(e.g., it's a logo or decorative image), say 'Not a data visualization'."""
                        }
                    ]
                }]
            )
            
            caption = response.choices[0].message.content
            
            if "Not a data visualization" not in caption:
                figures.append({
                    "text": caption,
                    "metadata": {
                        "content_type": "figure_caption",
                        "page_number": page_num + 1,
                        "figure_index": img_idx,
                        "source_type": "vision_model_caption"
                    }
                })
    
    return figures
```

### Cost Consideration

Vision model API calls are expensive (~$0.003–0.01 per image with GPT-4o). For a 100-page report with 20 charts, this adds $0.06–0.20 per document. For a corpus of 10,000 such documents, the cost becomes significant.

Optimization strategies:
- Filter out images below a minimum size (logos, icons, decorative elements).
- Use a cheap classifier first to determine if an image is a chart/diagram vs. a photo or logo before sending to the vision model.
- Batch processing: process figure captioning as an async background job, not in the critical path of document ingestion.

---

## Putting It Together: A Complete Pre-processing Pipeline

Here is how a production pre-processing pipeline looks end-to-end for a mixed-format enterprise document corpus:

```python
from dataclasses import dataclass
from typing import Optional
import fitz
from unstructured.partition.auto import partition

@dataclass
class ProcessedChunk:
    text: str
    metadata: dict
    embedding_text: str  # May differ from text for tables

def process_document(file_path: str, doc_metadata: dict) -> list[ProcessedChunk]:
    chunks = []
    
    # Step 1: Detect format
    format_info = detect_format(file_path)
    
    # Step 2: Route to appropriate parser
    if format_info['mime_type'] == 'application/pdf':
        if format_info['pdf_type'] == 'digital':
            # Use pdfplumber for rich table extraction
            raw_pages = extract_with_pdfplumber(file_path)
        else:
            # Route to Textract for OCR + layout
            raw_pages = extract_with_textract(file_path)
    else:
        # Use Unstructured for DOCX, HTML, PPTX, etc.
        elements = partition(filename=file_path)
        raw_pages = convert_unstructured_elements(elements)
    
    # Step 3: Process each page's content
    for page in raw_pages:
        # Process text content
        if page.get('text'):
            text_chunks = chunk_text(
                page['text'],
                base_metadata={**doc_metadata, "page_number": page['page_number']}
            )
            chunks.extend(text_chunks)
        
        # Process tables
        for table in page.get('tables', []):
            table_chunks = process_table(table, page['page_number'], doc_metadata)
            chunks.extend(table_chunks)
    
    # Step 4: Extract and caption figures (async, can be done in parallel)
    figure_chunks = extract_and_caption_figures(file_path)
    for fig in figure_chunks:
        fig['metadata'].update(doc_metadata)
        chunks.append(ProcessedChunk(
            text=fig['text'],
            metadata=fig['metadata'],
            embedding_text=fig['text']
        ))
    
    # Step 5: Quality filtering
    chunks = [c for c in chunks if is_quality_chunk(c)]
    
    return chunks

def is_quality_chunk(chunk: ProcessedChunk) -> bool:
    """Filter out low-quality chunks."""
    text = chunk.text.strip()
    
    # Too short
    if len(text.split()) < 10:
        return False
    
    # Mostly non-alphabetic (likely a garbled table or OCR noise)
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.4:
        return False
    
    # Repeating characters (OCR artifact)
    if any(char * 5 in text for char in 'abcdefghijklmnopqrstuvwxyz'):
        return False
    
    return True
```

---

## Monitoring Pre-processing Quality

Pre-processing failures are silent — the pipeline completes but produces bad output. Build quality checks in from the start.

**Extraction completeness:** For each document, estimate the expected number of pages and words. Alert when the extracted word count is significantly below expectations.

**Table detection rate:** If your documents are known to contain tables (financial reports), track how many tables are detected per document. A sudden drop indicates a parser change or a new document template.

**OCR confidence tracking:** AWS Textract and Google Document AI return confidence scores per word. Track the distribution. Low confidence → flag for human review or re-processing with a better OCR engine.

**Character error rate sampling:** Randomly sample 1% of chunks from scanned documents and manually verify OCR quality. Use this to calibrate your confidence score thresholds.

---

## Summary

- Format detection is the first step. Handle mixed PDFs (some digital pages, some scanned) at the page level, not the document level.
- Digital PDF text extraction with PyMuPDF is fast and reliable for standard layouts. Multi-column layouts require heuristic or model-based column detection.
- OCR quality varies enormously by tool. Tesseract for clean scans and budget constraints; Textract/Document AI/Azure Document Intelligence for complex layouts and forms.
- Image pre-processing (deskew, contrast, binarization) before OCR significantly improves results on low-quality scans.
- Layout analysis identifies the structural role of each page region. Unstructured.io provides a practical all-in-one pipeline.
- Tables require special handling. Natural language serialization per row gives the best retrieval quality. Store the full Markdown table in metadata for LLM consumption.
- Figures must be captioned by a vision model or they are invisible to the RAG system.
- Build quality filtering and monitoring into the pre-processing pipeline from the start. Silent failures are the norm without explicit checks.

---

## What's Next

Lesson 2.5 goes into multimodal RAG — systems that handle documents containing images, charts, and mixed content at a deeper level, including vision-language models, multimodal embeddings, and architectures that retrieve across text and image modalities simultaneously.