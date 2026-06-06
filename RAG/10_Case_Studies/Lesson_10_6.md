# Case Study 6 — Multimodal RAG: Invoices, Receipts, Scanned Forms — OCR + Layout + Retrieval

---

## Problem Statement

A global logistics company processes 85,000 documents per month: supplier invoices, customs declarations, bills of lading, proof-of-delivery receipts, and compliance certificates. Currently, 45 data entry clerks manually extract data from these documents, which takes 4-8 minutes per document and has a 2.3% error rate.

The goal: an AI system that (1) extracts structured data from incoming documents, (2) answers questions about historical documents, and (3) flags discrepancies and anomalies automatically.

The document challenge:
- Every supplier has a different invoice format (hundreds of templates).
- Many documents are scanned paper forms — image quality varies from excellent to barely legible.
- Critical fields: invoice number, vendor name, line items, unit prices, totals, tax, currency, payment terms.
- Legal requirement: extracted data must be traceable to the exact location in the source image.
- Anomaly detection: invoices that deviate from expected patterns (unusual amounts, new vendors, different payment terms) must be flagged.
- Multi-language: 12 languages across supplier base.

This case study is fundamentally different from the others. The primary challenge is not text retrieval — it is structured information extraction from images. RAG is the query interface on top of the extracted data, not the primary extraction mechanism.

---

## System Architecture Overview

```
Incoming Document (image/PDF)
         │
         ▼
┌─────────────────────────────────────────────────┐
│            EXTRACTION PIPELINE                   │
│                                                  │
│  Image preprocessing → Layout analysis → OCR    │
│         → Field extraction → Validation          │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐    ┌────────────────────────┐
│  Structured DB   │    │   Vector DB            │
│  (PostgreSQL)    │    │   (Qdrant)             │
│                  │    │                        │
│  invoices table  │    │  NL descriptions of    │
│  line_items table│    │  invoices + anomalies  │
│  vendors table   │    │  for semantic search   │
└─────────────────┘    └────────────────────────┘
         │                        │
         └────────────┬───────────┘
                      │
         ┌────────────▼────────────┐
         │     QUERY INTERFACE      │
         │   (RAG + SQL routing)    │
         └──────────────────────────┘
```

---

## Decision 1 — Document Triage and Routing

Not all documents require the same extraction approach. Route by document type and quality:

```python
from enum import Enum

class DocumentType(Enum):
    INVOICE = "invoice"
    RECEIPT = "receipt"
    CUSTOMS_DECLARATION = "customs_declaration"
    BILL_OF_LADING = "bill_of_lading"
    COMPLIANCE_CERT = "compliance_cert"
    UNKNOWN = "unknown"

class DocumentQuality(Enum):
    HIGH = "high"       # Clean scan or digital PDF
    MEDIUM = "medium"   # Acceptable scan, some noise
    LOW = "low"         # Poor quality, degraded, skewed
    UNREADABLE = "unreadable"

async def triage_document(
    image: bytes,
    file_metadata: dict
) -> dict:
    """
    Classify document type and quality before choosing extraction strategy.
    """
    from PIL import Image
    import io
    import numpy as np
    
    img = Image.open(io.BytesIO(image))
    img_array = np.array(img.convert('L'))  # Grayscale
    
    # Quality assessment
    std_dev = float(img_array.std())         # Low std = low contrast = low quality
    mean_brightness = float(img_array.mean())
    
    if std_dev < 30:
        quality = DocumentQuality.LOW
    elif std_dev < 50:
        quality = DocumentQuality.MEDIUM
    else:
        quality = DocumentQuality.HIGH
    
    # Type detection: use vision model for ambiguous documents
    # Use file metadata (filename, source system tag) for known formats
    if file_metadata.get("source_system") == "procurement_portal":
        doc_type = DocumentType.INVOICE
    elif file_metadata.get("source_system") == "customs_system":
        doc_type = DocumentType.CUSTOMS_DECLARATION
    else:
        doc_type = await classify_document_type_vision(image)
    
    return {
        "doc_type": doc_type,
        "quality": quality,
        "image_width": img.width,
        "image_height": img.height,
        "requires_preprocessing": quality in [DocumentQuality.LOW, DocumentQuality.MEDIUM],
        "extraction_strategy": choose_extraction_strategy(doc_type, quality)
    }

def choose_extraction_strategy(doc_type: DocumentType, quality: DocumentQuality) -> str:
    """Choose the appropriate extraction pipeline."""
    
    if quality == DocumentQuality.HIGH and doc_type == DocumentType.INVOICE:
        return "textract_forms"  # Fast, structured
    
    elif quality == DocumentQuality.LOW:
        return "preprocess_then_textract"  # Image cleanup first
    
    elif doc_type == DocumentType.UNKNOWN:
        return "vision_model_general"  # GPT-4o for unknown formats
    
    elif doc_type in [DocumentType.CUSTOMS_DECLARATION, DocumentType.BILL_OF_LADING]:
        return "textract_specialized"  # Custom Textract model for these types
    
    return "textract_forms"  # Default
```

---

## Decision 2 — Image Preprocessing Pipeline

Before OCR, preprocess to maximize text legibility:

```python
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import cv2

class DocumentPreprocessor:
    
    def preprocess(self, image: Image.Image, quality: str) -> Image.Image:
        """
        Apply preprocessing steps based on document quality.
        Order matters: deskew before denoise, enhance before binarize.
        """
        img = np.array(image)
        
        # Step 1: Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # Step 2: Deskew (correct rotation)
        angle = self._detect_skew(gray)
        if abs(angle) > 0.5:
            gray = self._rotate(gray, angle)
        
        # Step 3: Enhance contrast (adaptive histogram equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Step 4: Denoise (bilateral filter preserves edges)
        if quality == "low":
            denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        else:
            denoised = enhanced
        
        # Step 5: Binarize (Otsu's method)
        _, binary = cv2.threshold(
            denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Step 6: Remove borders/shadows
        cleaned = self._remove_borders(binary)
        
        return Image.fromarray(cleaned)
    
    def _detect_skew(self, gray_image: np.ndarray) -> float:
        """Detect document skew angle using Hough line transform."""
        edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        
        if lines is None:
            return 0.0
        
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            if -45 < angle < 45:
                angles.append(angle)
        
        return float(np.median(angles)) if angles else 0.0
    
    def _rotate(self, image: np.ndarray, angle: float) -> np.ndarray:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
    
    def _remove_borders(self, binary: np.ndarray) -> np.ndarray:
        """Remove dark borders that can confuse OCR."""
        # Find the bounding box of content
        coords = cv2.findNonZero(binary)
        if coords is None:
            return binary
        x, y, w, h = cv2.boundingRect(coords)
        # Add small padding
        padding = 10
        return binary[
            max(0, y-padding):min(binary.shape[0], y+h+padding),
            max(0, x-padding):min(binary.shape[1], x+w+padding)
        ]
```

---

## Decision 3 — Structured Data Extraction

The heart of the system: extracting structured fields from diverse invoice formats.

**Strategy: AWS Textract for layout + GPT-4o Vision for validation/correction**

```python
import boto3
import json

class InvoiceExtractor:
    
    def __init__(self):
        self.textract = boto3.client('textract', region_name='us-east-1')
        self.openai = OpenAI()
    
    async def extract_invoice(
        self,
        image: bytes,
        document_id: str
    ) -> dict:
        """
        Two-stage extraction:
        1. Textract: fast, structured extraction
        2. GPT-4o Vision: validation, gap-filling, low-confidence fields
        """
        
        # Stage 1: Textract for initial extraction
        textract_result = await self._textract_extract(image)
        
        # Parse Textract output into structured form
        parsed = self._parse_textract_response(textract_result)
        
        # Stage 2: GPT-4o Vision for validation and gap-filling
        validated = await self._vision_validate_and_fill(image, parsed)
        
        # Stage 3: Business logic validation
        final = self._apply_business_rules(validated)
        
        return {
            "document_id": document_id,
            "extracted_data": final,
            "confidence_scores": final.get("confidence"),
            "extraction_method": "textract+vision",
            "needs_human_review": self._requires_review(final)
        }
    
    async def _textract_extract(self, image: bytes) -> dict:
        """Run AWS Textract AnalyzeDocument for forms and tables."""
        
        response = self.textract.analyze_document(
            Document={"Bytes": image},
            FeatureTypes=["FORMS", "TABLES", "LAYOUT"]
        )
        
        return response
    
    def _parse_textract_response(self, response: dict) -> dict:
        """
        Parse Textract blocks into structured invoice fields.
        """
        blocks = response.get("Blocks", [])
        
        # Extract key-value pairs (form fields)
        key_value_pairs = {}
        key_blocks = {b["Id"]: b for b in blocks if b["BlockType"] == "KEY_VALUE_SET" and "KEY" in b.get("EntityTypes", [])}
        value_blocks = {b["Id"]: b for b in blocks if b["BlockType"] == "KEY_VALUE_SET" and "VALUE" in b.get("EntityTypes", [])}
        
        for key_id, key_block in key_blocks.items():
            # Find associated value
            for relationship in key_block.get("Relationships", []):
                if relationship["Type"] == "VALUE":
                    for value_id in relationship["Ids"]:
                        if value_id in value_blocks:
                            key_text = self._get_block_text(key_block, blocks)
                            value_text = self._get_block_text(value_blocks[value_id], blocks)
                            key_value_pairs[key_text] = {
                                "value": value_text,
                                "confidence": key_block.get("Confidence", 0)
                            }
        
        # Extract tables (line items)
        tables = self._extract_tables(blocks)
        
        # Map to invoice schema
        invoice = {
            "invoice_number": self._find_field(key_value_pairs, ["Invoice #", "Invoice No", "Invoice Number", "Inv #"]),
            "invoice_date": self._find_field(key_value_pairs, ["Date", "Invoice Date", "Issue Date"]),
            "vendor_name": self._find_field(key_value_pairs, ["Vendor", "From", "Supplier", "Bill From"]),
            "vendor_address": self._find_field(key_value_pairs, ["Address", "Vendor Address"]),
            "bill_to": self._find_field(key_value_pairs, ["Bill To", "Customer", "Client"]),
            "subtotal": self._find_field(key_value_pairs, ["Subtotal", "Sub Total", "Net Amount"]),
            "tax": self._find_field(key_value_pairs, ["Tax", "VAT", "GST", "Tax Amount"]),
            "total": self._find_field(key_value_pairs, ["Total", "Amount Due", "Grand Total", "Total Due"]),
            "currency": self._detect_currency(key_value_pairs),
            "payment_terms": self._find_field(key_value_pairs, ["Payment Terms", "Terms", "Due Date"]),
            "line_items": tables[0] if tables else [],
            "_raw_key_value_pairs": key_value_pairs,
            "_textract_confidence": self._compute_avg_confidence(key_value_pairs)
        }
        
        return invoice
    
    async def _vision_validate_and_fill(
        self,
        image: bytes,
        parsed_invoice: dict
    ) -> dict:
        """
        Use GPT-4o Vision to validate Textract output and fill missing fields.
        Only called when Textract confidence is below threshold or fields are missing.
        """
        
        missing_critical = any(
            parsed_invoice.get(field) is None
            for field in ["invoice_number", "vendor_name", "total"]
        )
        
        low_confidence = parsed_invoice.get("_textract_confidence", 0) < 0.85
        
        if not missing_critical and not low_confidence:
            return parsed_invoice  # Textract result is good enough
        
        # Build a targeted prompt based on what's missing
        missing_fields = [
            field for field in ["invoice_number", "invoice_date", "vendor_name", "total", "currency", "payment_terms"]
            if parsed_invoice.get(field) is None or parsed_invoice.get(field, {}).get("confidence", 1) < 0.7
        ]
        
        import base64
        image_b64 = base64.b64encode(image).decode()
        
        prompt = f"""Extract the following fields from this invoice image.

FIELDS NEEDED: {', '.join(missing_fields)}

ALREADY EXTRACTED (verify these are correct):
{json.dumps({k: v for k, v in parsed_invoice.items() if not k.startswith('_') and v is not None}, indent=2)}

Return ONLY a JSON object with the missing fields and any corrections to existing fields.
For monetary values, include both the amount and currency.
If a field is genuinely absent from the document, set it to null."""
        
        response = self.openai.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": prompt}
                ]
            }],
            response_format={"type": "json_object"},
            max_tokens=500,
            temperature=0.0
        )
        
        corrections = json.loads(response.choices[0].message.content)
        
        # Merge corrections into parsed invoice (corrections win)
        for field, value in corrections.items():
            if value is not None:
                parsed_invoice[field] = {"value": value, "confidence": 0.95, "source": "vision_model"}
        
        return parsed_invoice
    
    def _apply_business_rules(self, invoice: dict) -> dict:
        """
        Apply validation rules and compute derived fields.
        """
        
        # Compute line item sum and compare to stated total
        if invoice.get("line_items"):
            computed_subtotal = sum(
                float(item.get("amount", 0))
                for item in invoice["line_items"]
                if item.get("amount")
            )
            
            stated_total = self._parse_amount(invoice.get("total", {}).get("value", "0"))
            
            if abs(computed_subtotal - stated_total) > 0.02 * stated_total:
                invoice["_validation_flags"] = invoice.get("_validation_flags", [])
                invoice["_validation_flags"].append({
                    "type": "amount_mismatch",
                    "computed": computed_subtotal,
                    "stated": stated_total,
                    "difference": abs(computed_subtotal - stated_total)
                })
        
        # Flag unusual payment terms
        payment_terms = invoice.get("payment_terms", {}).get("value", "")
        if payment_terms and not self._is_standard_payment_term(payment_terms):
            invoice["_validation_flags"] = invoice.get("_validation_flags", [])
            invoice["_validation_flags"].append({
                "type": "unusual_payment_terms",
                "value": payment_terms
            })
        
        return invoice
```

---

## Decision 4 — Storage Architecture

Extracted invoice data goes into both a structured database (for precise queries) and a vector database (for semantic search):

```sql
-- Structured storage for precise queries
CREATE TABLE invoices (
    invoice_id UUID PRIMARY KEY,
    document_id VARCHAR(100) UNIQUE,
    source_image_path TEXT,
    
    -- Extracted fields
    invoice_number VARCHAR(100),
    invoice_date DATE,
    vendor_name VARCHAR(200),
    vendor_id UUID REFERENCES vendors(vendor_id),
    
    subtotal DECIMAL(15, 2),
    tax_amount DECIMAL(15, 2),
    total_amount DECIMAL(15, 2),
    currency CHAR(3),
    payment_terms VARCHAR(200),
    due_date DATE,
    
    -- Quality metadata
    extraction_confidence FLOAT,
    extraction_method VARCHAR(50),
    needs_human_review BOOLEAN DEFAULT FALSE,
    review_reason TEXT,
    
    -- Audit
    extracted_at TIMESTAMPTZ DEFAULT NOW(),
    reviewed_by UUID,
    reviewed_at TIMESTAMPTZ
);

CREATE TABLE line_items (
    item_id UUID PRIMARY KEY,
    invoice_id UUID REFERENCES invoices(invoice_id),
    description TEXT,
    quantity DECIMAL(10, 3),
    unit_price DECIMAL(15, 4),
    amount DECIMAL(15, 2),
    tax_rate DECIMAL(5, 2),
    item_index INTEGER  -- Position in invoice
);

-- Indexes for common query patterns
CREATE INDEX idx_invoices_vendor ON invoices(vendor_name, invoice_date);
CREATE INDEX idx_invoices_total ON invoices(total_amount);
CREATE INDEX idx_invoices_date ON invoices(invoice_date);
CREATE INDEX idx_invoices_review ON invoices(needs_human_review) WHERE needs_human_review = TRUE;
```

**Vector database for semantic search:**

```python
# Generate a natural language description of each invoice for semantic indexing
def generate_invoice_description(invoice: dict) -> str:
    """
    Convert extracted invoice data to a searchable natural language description.
    """
    line_item_summary = ""
    if invoice.get("line_items"):
        items = invoice["line_items"][:5]  # Top 5 items
        line_item_summary = "Items: " + "; ".join([
            f"{item.get('description', 'Unknown')} x{item.get('quantity', 1)} @ {item.get('unit_price', 0)}"
            for item in items
        ])
    
    flags_summary = ""
    if invoice.get("_validation_flags"):
        flags_summary = "Flags: " + ", ".join([
            f["type"] for f in invoice["_validation_flags"]
        ])
    
    description = (
        f"Invoice {invoice.get('invoice_number', 'Unknown')} "
        f"from {invoice.get('vendor_name', 'Unknown vendor')} "
        f"dated {invoice.get('invoice_date', 'Unknown date')}. "
        f"Total: {invoice.get('currency', 'USD')} {invoice.get('total_amount', 0):,.2f}. "
        f"Payment terms: {invoice.get('payment_terms', 'Standard')}. "
        f"{line_item_summary}. "
        f"{flags_summary}"
    ).strip()
    
    return description
```

---

## Decision 5 — Anomaly Detection

Flag invoices that deviate from established patterns:

```python
class InvoiceAnomalyDetector:
    
    async def detect_anomalies(
        self,
        invoice: dict,
        db_client
    ) -> list[dict]:
        """
        Detect anomalies by comparing to historical patterns for this vendor.
        """
        anomalies = []
        vendor_name = invoice.get("vendor_name")
        
        if not vendor_name:
            return [{"type": "unknown_vendor", "severity": "high"}]
        
        # Get historical stats for this vendor
        stats = await db_client.fetchrow("""
            SELECT 
                AVG(total_amount) as avg_total,
                STDDEV(total_amount) as stddev_total,
                COUNT(*) as invoice_count,
                MAX(invoice_date) as last_invoice_date
            FROM invoices
            WHERE vendor_name = $1
            AND invoice_date > NOW() - INTERVAL '12 months'
        """, vendor_name)
        
        current_total = float(invoice.get("total_amount", 0))
        
        # Anomaly 1: Amount > 3 standard deviations from vendor average
        if stats and stats["stddev_total"] and stats["invoice_count"] > 3:
            z_score = abs(current_total - stats["avg_total"]) / stats["stddev_total"]
            if z_score > 3:
                anomalies.append({
                    "type": "unusual_amount",
                    "severity": "high",
                    "detail": f"Amount ${current_total:,.2f} is {z_score:.1f} std devs from vendor average ${stats['avg_total']:,.2f}"
                })
        
        # Anomaly 2: New vendor (no history)
        if not stats or stats["invoice_count"] == 0:
            anomalies.append({
                "type": "new_vendor",
                "severity": "medium",
                "detail": f"First invoice from vendor: {vendor_name}"
            })
        
        # Anomaly 3: Duplicate invoice number
        existing = await db_client.fetchrow("""
            SELECT invoice_id FROM invoices
            WHERE vendor_name = $1 AND invoice_number = $2
        """, vendor_name, invoice.get("invoice_number"))
        
        if existing:
            anomalies.append({
                "type": "duplicate_invoice",
                "severity": "critical",
                "detail": f"Invoice number {invoice.get('invoice_number')} already exists for {vendor_name}"
            })
        
        # Anomaly 4: Amount mismatch (from business rules validation)
        for flag in invoice.get("_validation_flags", []):
            if flag["type"] == "amount_mismatch":
                anomalies.append({
                    "type": "amount_calculation_error",
                    "severity": "high",
                    "detail": f"Line item sum ({flag['computed']:,.2f}) differs from stated total ({flag['stated']:,.2f})"
                })
        
        return anomalies
```

---

## Decision 6 — Query Interface (RAG + SQL)

```python
async def answer_invoice_query(
    query: str,
    user_context: dict,
    vector_db,
    structured_db,
    embedding_model,
    llm_client
) -> dict:
    """
    Route invoice queries to SQL or vector search based on query type.
    """
    
    query_type_prompt = f"""Classify this invoice query:
"{query}"

Types:
- LOOKUP: find a specific invoice by number, vendor, date
- AGGREGATE: sum/count/average across multiple invoices
- SEMANTIC: find invoices matching a description
- ANOMALY: find unusual or flagged invoices
- COMPARISON: compare invoices or vendor patterns

Return JSON: {{"type": "TYPE", "sql_feasible": true/false}}"""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": query_type_prompt}],
        response_format={"type": "json_object"},
        max_tokens=50,
        temperature=0.0
    )
    
    import json
    query_info = json.loads(response.choices[0].message.content)
    
    if query_info.get("sql_feasible"):
        # Generate SQL from natural language
        sql_query = await generate_sql_from_query(query, llm_client)
        sql_results = await structured_db.fetch(sql_query)
        
        return {
            "source": "structured_db",
            "results": sql_results,
            "query_used": sql_query
        }
    else:
        # Semantic search over invoice descriptions
        query_embedding = await embedding_model.embed(query)
        vector_results = await vector_db.search(
            query_vector=query_embedding,
            limit=10
        )
        
        return {
            "source": "vector_db",
            "results": [r.payload for r in vector_results]
        }
```

---

## Results

| Metric | Before (Manual) | After (AI) |
|---|---|---|
| Documents/day processed | 2,800 | 4,200 (auto) |
| Processing time per invoice | 4-8 min | 12 sec |
| Extraction accuracy | 97.7% | 99.1% |
| Staff required | 45 clerks | 8 reviewers |
| Anomalies detected | ~70% | 96% |
| Duplicate invoice detection | Manual audit | Real-time |
| Query time for historical lookup | Hours | Seconds |

---

## Lessons Learned

**Lesson 1:** Two-stage extraction (Textract + Vision) is the right architecture. Textract handles 80% of cases accurately and cheaply. Vision model handles the remaining 20% that Textract struggles with (rotated text, unusual layouts, handwritten fields). Running vision on everything is 10× more expensive without meaningful quality improvement on clean documents.

**Lesson 2:** Business rule validation is as important as field extraction. Textract accurately extracts line items and totals. The system only became reliable after adding the validation step that checks whether they arithmetically agree.

**Lesson 3:** Anomaly detection required significantly more historical data than expected. Detecting "unusual amounts" with statistical significance requires at least 10 invoices per vendor. For new vendors, the system falls back to a rule-based check (amount > $10,000 always requires review).

**Lesson 4:** The semantic vector search is the most valuable query interface for ad-hoc analysis that cannot be anticipated in advance. "Find all invoices from vendors who provide logistics services in Southeast Asia" is easy to ask but impossible to answer with predefined SQL — it requires semantic understanding of vendor descriptions.

---

## Interview Questions This Case Study Prepares You For

**"How do you build RAG for scanned documents?"**
Answer: Three-stage pipeline — (1) image preprocessing (deskew, denoise, binarize) to maximize OCR quality, (2) layout-aware OCR (AWS Textract) for structure recovery, (3) vision model validation for low-confidence fields. Store extracted structured data in a relational DB and generate NL descriptions for vector indexing.

**"How do you handle documents where precise field extraction matters more than retrieval?"**
Answer: Treat it as an information extraction problem first, not a retrieval problem. Build a structured extraction pipeline, store in a relational database, and layer semantic search on top. Route queries to SQL for precise structured queries and to vector search for semantic/descriptive queries.

**"How do you detect anomalies in a document processing system?"**
Answer: Z-score on vendor-specific historical distributions for amount anomalies. Duplicate detection via exact field matching. Business rule validation (line item sum vs. stated total). New vendor detection. Each anomaly type has a severity level that determines whether it goes to auto-approval, review queue, or immediate escalation.