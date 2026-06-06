# Case Study 2 — Financial Report Analysis System: Tables, Charts, Earnings Calls, Regulatory Filings

---

## Problem Statement

A mid-sized investment management firm wants to build an analyst assistant that answers questions over their financial document library. Analysts spend hours manually reading earnings calls, 10-K/10-Q filings, analyst reports, and ESG disclosures to answer questions like:

- "What was Apple's gross margin trend over the last 8 quarters?"
- "Which of our portfolio companies have revenue concentration risk above 30%?"
- "What did the CEO say about supply chain risks in the Q3 2024 earnings call?"
- "Compare the debt-to-equity ratios of our top 5 holdings as of their latest filing."

The corpus:
- 12,000 SEC filings (10-K, 10-Q, 8-K) across 200 portfolio companies, last 5 years.
- 4,000 earnings call transcripts.
- 800 analyst research reports (PDF, 20-60 pages each).
- 200 ESG/sustainability reports.
- New documents arrive daily (filings, transcripts).

The document challenge:
- Dense financial tables spanning multiple pages.
- Charts and graphs (revenue trends, margin waterfall charts, geographic breakdowns).
- Precise numerical data where accuracy is critical — a wrong revenue figure has real consequences.
- Cross-document comparison is the primary use case (compare Company A to Company B).
- Documents follow standard structures (MD&A, Risk Factors, Financial Statements sections in 10-Ks) but formatting varies by company and year.

---

## The Unique Challenge: Financial Data Precision

Financial RAG has a constraint that general RAG does not: numerical precision is non-negotiable. In a customer support chatbot, a slightly imprecise answer is annoying. In a financial analysis tool, a slightly imprecise answer — "$4.2 billion" when the actual figure is "$4.02 billion" — can affect investment decisions worth millions.

This shapes every design decision.

---

## Architecture Design Decisions

### Decision 1 — Document Pre-Processing: Structured Extraction First

Financial documents are not "text with some tables." Financial data *is* the document. The tables in an income statement contain the primary information; the prose explains and contextualizes it.

Strategy: extract structured data first, build text on top of it.

**For SEC filings (digital PDF):**

SEC EDGAR provides XBRL-tagged financial data for all filings since 2009. XBRL is machine-readable structured financial data — it is the gold standard source for financial figures.

```python
from sec_api import XbrlApi

xbrl_api = XbrlApi(api_key="your-key")

async def extract_financial_data_xbrl(filing_url: str) -> dict:
    """
    Extract structured financial data from SEC XBRL tags.
    Far more reliable than parsing PDF tables.
    """
    xbrl_data = xbrl_api.xbrl_frames(filing_url)
    
    return {
        "income_statement": {
            "revenue": xbrl_data.get("Revenues"),
            "gross_profit": xbrl_data.get("GrossProfit"),
            "operating_income": xbrl_data.get("OperatingIncomeLoss"),
            "net_income": xbrl_data.get("NetIncomeLoss"),
            "eps_diluted": xbrl_data.get("EarningsPerShareDiluted")
        },
        "balance_sheet": {
            "total_assets": xbrl_data.get("Assets"),
            "total_liabilities": xbrl_data.get("Liabilities"),
            "stockholders_equity": xbrl_data.get("StockholdersEquity"),
            "cash": xbrl_data.get("CashAndCashEquivalentsAtCarryingValue")
        },
        "cash_flow": {
            "operating_cash_flow": xbrl_data.get("NetCashProvidedByUsedInOperatingActivities"),
            "capex": xbrl_data.get("PaymentsToAcquirePropertyPlantAndEquipment"),
            "free_cash_flow": None  # Computed: operating_cf - capex
        },
        "filing_date": xbrl_data.get("filing_date"),
        "period": xbrl_data.get("period_of_report")
    }
```

For pre-2009 filings or companies that do not file XBRL, fall back to pdfplumber table extraction with verification.

**Financial table serialization:**

Unlike general documents where NL serialization per row is sufficient, financial tables require preserving the full table with column headers because users often ask about multi-period comparisons.

```python
def serialize_financial_table(
    table: dict,
    table_name: str,
    company: str,
    period: str
) -> list[dict]:
    """
    Generate both a row-level embedding chunk AND a full table summary chunk.
    """
    chunks = []
    
    # Full table chunk (for queries that need to see the whole table)
    full_table_text = f"{company} {table_name} for {period}:\n"
    for metric, value in table.items():
        if value is not None:
            full_table_text += f"  {metric}: {value:,.2f}\n"
    
    chunks.append({
        "text": full_table_text,
        "metadata": {
            "content_type": "financial_table_full",
            "table_name": table_name,
            "company": company,
            "period": period
        }
    })
    
    # Individual metric chunks (for precise metric lookup)
    for metric, value in table.items():
        if value is not None:
            metric_text = f"{company} {metric} ({table_name}) for {period}: {value:,.2f}"
            chunks.append({
                "text": metric_text,
                "metadata": {
                    "content_type": "financial_metric",
                    "metric_name": metric,
                    "metric_value": float(value),  # Store as float for range queries
                    "table_name": table_name,
                    "company": company,
                    "period": period
                }
            })
    
    return chunks
```

### Decision 2 — Chart and Figure Extraction

Financial charts carry critical information not in the text:
- Revenue growth trajectory (line chart).
- Margin waterfall (the breakdown from gross to operating to net margin).
- Geographic revenue split (pie chart).
- Market share evolution over time.

**Vision model captioning for financial charts:**

```python
async def caption_financial_chart(
    image: bytes,
    surrounding_text: str,
    company: str,
    filing_period: str,
    llm_client
) -> str:
    """
    Generate a detailed, data-rich caption for a financial chart.
    """
    import base64
    
    image_b64 = base64.b64encode(image).decode()
    
    prompt = f"""You are analyzing a financial chart from {company}'s {filing_period} filing.

Surrounding document text (for context): {surrounding_text[:500]}

Describe this chart with extreme precision:
1. Chart type and what it measures
2. ALL visible data points with exact values (do not say "approximately")
3. Time periods shown
4. Trend direction and magnitude (e.g., "revenue grew from $X.X billion in YYYY to $X.X billion in YYYY, a X% CAGR")
5. Any notable inflection points, anomalies, or management callouts
6. Units (billions, millions, percentages)

This description will be used by financial analysts — precision is critical."""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                {"type": "text", "text": prompt}
            ]
        }],
        max_tokens=600,
        temperature=0.0  # Zero temperature for numerical precision
    )
    
    return response.choices[0].message.content
```

### Decision 3 — Hybrid Storage Architecture

Pure vector search is insufficient for financial data. Precise numerical queries ("companies with revenue > $10B") require structured database queries, not semantic similarity.

**Architecture: Vector DB + Relational DB**

```
┌─────────────────────────────────────────────────────────────┐
│                     Query Router                             │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌─────────────────┐   ┌─────────────────────────────┐
│   Vector DB     │   │   Structured Financial DB    │
│   (Qdrant)      │   │   (PostgreSQL)               │
│                 │   │                              │
│ - Text chunks   │   │ - income_statements table    │
│ - Chart captions│   │ - balance_sheets table       │
│ - MD&A sections │   │ - cash_flows table           │
│ - Risk factors  │   │ - key_metrics table          │
│ - Earnings call │   │   (EPS, margins, ratios)     │
│   transcripts   │   │                              │
└─────────────────┘   └─────────────────────────────┘
```

```sql
CREATE TABLE financial_metrics (
    id SERIAL PRIMARY KEY,
    company VARCHAR(10),           -- Ticker symbol
    company_name VARCHAR(200),
    period VARCHAR(20),            -- "2024-Q3", "2023-FY"
    period_end_date DATE,
    filing_type VARCHAR(20),       -- "10-K", "10-Q"
    metric_name VARCHAR(100),      -- "revenue", "gross_margin", "debt_equity"
    metric_value DECIMAL(20, 4),
    unit VARCHAR(20),              -- "USD_millions", "percentage", "ratio"
    source_chunk_id VARCHAR(100),  -- Links back to the vector DB chunk
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_metrics_company_period ON financial_metrics(company, period);
CREATE INDEX idx_metrics_name_value ON financial_metrics(metric_name, metric_value);
```

### Decision 4 — Query Router

Financial queries fall into distinct types requiring different retrieval strategies:

```python
async def route_financial_query(
    query: str,
    llm_client
) -> dict:
    """
    Classify financial query type to determine retrieval strategy.
    """
    
    prompt = f"""Classify this financial analyst query.

Query: {query}

Types:
1. NUMERICAL_LOOKUP: asking for a specific financial figure for one company/period
   Example: "What was Apple's Q3 2024 revenue?"
   
2. TIME_SERIES: asking about trends across multiple periods
   Example: "What was Apple's gross margin over the last 8 quarters?"
   
3. CROSS_COMPANY: comparing financial metrics across multiple companies
   Example: "Compare debt-to-equity ratios of Apple, Microsoft, and Google"
   
4. SCREENING: finding companies that meet numerical criteria
   Example: "Which portfolio companies have revenue growth > 20% YoY?"
   
5. QUALITATIVE: asking about narrative content (management discussion, risks, strategy)
   Example: "What did Apple's CEO say about AI investment in Q3 2024?"
   
6. HYBRID: requires both numerical data and qualitative context

Return JSON: {{"type": "TYPE", "companies": ["list"], "metrics": ["list"], "periods": ["list"]}}"""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=150,
        temperature=0.0
    )
    
    import json
    return json.loads(response.choices[0].message.content)


async def retrieve_financial(query: str, user_context: dict) -> dict:
    """
    Route to appropriate retrieval based on query type.
    """
    
    query_info = await route_financial_query(query, llm_client)
    query_type = query_info["type"]
    
    if query_type == "NUMERICAL_LOOKUP":
        # Direct SQL query — most precise
        sql_results = await sql_lookup(
            company=query_info["companies"][0] if query_info["companies"] else None,
            metrics=query_info["metrics"],
            periods=query_info["periods"]
        )
        return {"source": "sql", "data": sql_results}
    
    elif query_type == "TIME_SERIES":
        # SQL for the time series data
        sql_results = await sql_time_series(
            company=query_info["companies"][0],
            metric=query_info["metrics"][0] if query_info["metrics"] else None,
            n_periods=8  # Default: last 8 quarters
        )
        # Also retrieve text context about the trend
        text_results = await vector_retrieve(query)
        return {"source": "hybrid", "data": sql_results, "context": text_results}
    
    elif query_type == "CROSS_COMPANY":
        # SQL for each company's metrics
        comparative_data = {}
        for company in query_info["companies"]:
            comparative_data[company] = await sql_lookup(
                company=company,
                metrics=query_info["metrics"],
                periods=query_info["periods"]
            )
        return {"source": "sql_comparative", "data": comparative_data}
    
    elif query_type == "SCREENING":
        # SQL WHERE clause with metric filters
        screened = await sql_screen(
            metric=query_info["metrics"][0],
            operator=query_info.get("operator", ">"),
            threshold=query_info.get("threshold"),
            universe=user_context.get("portfolio_companies")
        )
        return {"source": "sql_screen", "data": screened}
    
    elif query_type in ["QUALITATIVE", "HYBRID"]:
        # Vector retrieval for text content
        text_results = await vector_retrieve(query)
        
        if query_type == "HYBRID" and query_info["companies"]:
            # Also fetch relevant metrics
            sql_results = await sql_lookup(
                company=query_info["companies"][0],
                metrics=query_info.get("metrics", []),
                periods=query_info.get("periods", [])
            )
            return {"source": "hybrid", "data": sql_results, "context": text_results}
        
        return {"source": "vector", "context": text_results}
```

### Decision 5 — Earnings Call Transcript Handling

Earnings call transcripts have a unique structure:
- Prepared remarks (CEO, CFO) — authoritative, carefully worded.
- Q&A section — more candid, analyst questions may reveal concerns.
- Speaker attribution matters ("what did the CFO say" vs "what did an analyst ask").

```python
def chunk_earnings_transcript(transcript: str, metadata: dict) -> list[dict]:
    """
    Speaker-aware chunking for earnings call transcripts.
    Preserves attribution and separates Q&A from prepared remarks.
    """
    import re
    
    chunks = []
    
    # Detect section boundaries
    prepared_remarks_end = re.search(
        r'(question.and.answer|q.?&.?a|operator.*questions)',
        transcript,
        re.IGNORECASE
    )
    
    if prepared_remarks_end:
        prepared_section = transcript[:prepared_remarks_end.start()]
        qa_section = transcript[prepared_remarks_end.start():]
    else:
        prepared_section = transcript
        qa_section = ""
    
    # Chunk prepared remarks by speaker turn
    speaker_turns = re.split(r'\n([A-Z][A-Z\s]+):\s*\n', prepared_section)
    
    current_speaker = None
    for i, turn in enumerate(speaker_turns):
        if re.match(r'^[A-Z][A-Z\s]+$', turn.strip()):
            current_speaker = turn.strip()
        elif current_speaker and turn.strip():
            chunks.append({
                "text": f"{current_speaker}: {turn.strip()[:800]}",
                "metadata": {
                    **metadata,
                    "content_type": "earnings_prepared_remarks",
                    "speaker": current_speaker,
                    "speaker_role": classify_speaker_role(current_speaker),
                    "section": "prepared_remarks"
                }
            })
    
    # Chunk Q&A by exchange (question + answer pair)
    if qa_section:
        qa_exchanges = parse_qa_exchanges(qa_section)
        for exchange in qa_exchanges:
            chunks.append({
                "text": exchange["text"],
                "metadata": {
                    **metadata,
                    "content_type": "earnings_qa",
                    "analyst_firm": exchange.get("analyst_firm"),
                    "topic": exchange.get("topic"),
                    "section": "qa"
                }
            })
    
    return chunks
```

### Decision 6 — Numerical Verification Layer

Given the precision requirements, add a post-generation numerical verification step:

```python
async def verify_numerical_claims(
    query: str,
    generated_answer: str,
    sql_data: dict,
    llm_client
) -> dict:
    """
    Check that any numerical claims in the answer match the source data.
    Critical for financial analysis where numbers matter.
    """
    
    # Extract numerical claims from the answer
    import re
    numerical_patterns = [
        r'\$[\d,]+(?:\.\d+)?(?:\s?(?:billion|million|thousand|B|M|K))?',
        r'[\d,]+(?:\.\d+)?%',
        r'[\d,]+(?:\.\d+)?\s?(?:billion|million|thousand|B|M)'
    ]
    
    extracted_numbers = []
    for pattern in numerical_patterns:
        matches = re.findall(pattern, generated_answer, re.IGNORECASE)
        extracted_numbers.extend(matches)
    
    if not extracted_numbers:
        return {"verified": True, "no_numbers_to_verify": True}
    
    # Cross-check against SQL data
    prompt = f"""Verify whether these numbers from an AI answer match the source data.

Question: {query}

AI Answer extract (numbers to verify): {extracted_numbers}

Source financial data (ground truth):
{format_sql_data_for_verification(sql_data)}

For each number in the AI answer:
1. Find it in the source data
2. Confirm it matches (within rounding tolerance)
3. Flag any discrepancies

Return JSON:
{{
    "all_verified": true/false,
    "verified_numbers": ["list of numbers confirmed correct"],
    "discrepancies": [
        {{"ai_said": "...", "source_says": "...", "context": "..."}}
    ]
}}"""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=500,
        temperature=0.0
    )
    
    import json
    verification = json.loads(response.choices[0].message.content)
    
    if not verification.get("all_verified") and verification.get("discrepancies"):
        # Trigger re-generation with the discrepancy noted
        corrected_answer = await regenerate_with_correction(
            query, generated_answer, verification["discrepancies"], llm_client
        )
        return {
            "verified": False,
            "discrepancies": verification["discrepancies"],
            "corrected_answer": corrected_answer
        }
    
    return {"verified": True}
```

### Decision 7 — Prompt Design for Financial Analysis

```python
FINANCIAL_ANALYSIS_PROMPT = """You are a financial analyst assistant with access to SEC filings, 
earnings call transcripts, and analyst reports.

PRECISION RULES (non-negotiable):
- Report ALL financial figures exactly as stated in the source. Do not round, estimate, or paraphrase numbers.
- Always specify: the company, the metric, the exact period (quarter/year), and the unit ($ millions, %, ratio).
- When citing a figure, include the source type: "Per the 10-K filed [date]" or "Per the Q3 2024 earnings call".
- If data spans multiple periods for comparison, show each period explicitly.

WHEN NUMBERS CONFLICT:
- If XBRL data and text differ, note both and flag the discrepancy.
- Restated figures: always use the most recent restatement and note if restated.

UNCERTAINTY:
- If asked about projections or guidance: clearly label as "management guidance" not "reported results".
- If the data is unavailable: say so explicitly. Do not estimate.
- For calculated ratios (not directly reported): show the calculation.

CONTEXT DATA:
{formatted_context}

FINANCIAL DATA FROM DATABASE:
{formatted_sql_data}"""
```

---

## Key Failure Modes and Mitigations

### Failure Mode 1: Table Parsing Creates Wrong Figures

10-K financial statements often span 3 pages. When split across chunk boundaries, headers get separated from data rows, creating incorrect associations.

**Mitigation:** XBRL extraction for all post-2009 SEC filings eliminates this problem entirely — XBRL data is pre-structured with correct metric-value associations. For pre-XBRL documents, use a specialized financial table parser (Camelot, pdfplumber) with explicit validation (column sums must balance, year-over-year changes should be reasonable).

### Failure Mode 2: LLM Hallucinates Financial Figures

The LLM may generate plausible but incorrect financial figures, particularly for numbers not explicitly in the context window.

**Mitigation:** The numerical verification layer (Decision 6) catches most cases. Additionally, the system prompt explicitly prohibits estimation. For critical queries, always retrieve from the structured SQL database rather than relying on text chunks.

### Failure Mode 3: Outdated Figures After Restatement

Companies restate prior-period financials. A retrieved chunk from 2022 may contain figures that were later restated in 2023.

**Mitigation:** Store restatement flags in metadata. When a restatement is detected (via 8-K filing or 10-K note disclosure), mark prior-period chunks with `has_restatement: true`. Retrieval filter by default excludes chunks with pending restatement reviews.

---

## Evaluation Setup

```python
FINANCIAL_EVAL_QUERIES = [
    # Numerical precision tests
    {"query": "What was Apple's revenue in Q3 FY2024?", 
     "expected_answer": "$85.78 billion",
     "tolerance": 0.01},  # 1% tolerance for rounding
    
    # Time series
    {"query": "Apple gross margin last 4 quarters",
     "expected_answer": "Q3 2024: 46.3%, Q2 2024: 46.6%, Q1 2024: 45.9%, Q4 2023: 45.2%"},
    
    # Cross-company comparison
    {"query": "Compare P/E ratios of Apple, Microsoft, Google",
     "expected_format": "table_or_list_with_all_three"},
    
    # Qualitative
    {"query": "What risks did Apple mention related to China in their 2024 10-K?",
     "requires_citation": True},
    
    # IDK test
    {"query": "What is Apple's Q1 2026 revenue forecast?",
     "expected": "idk_or_guidance_disclaimer"}
]
```

---

## Lessons Learned

**Lesson 1:** XBRL data is a huge unlock. Teams that rely solely on PDF parsing for financial figures spend enormous time debugging table extraction errors. Investing in XBRL API access early pays back immediately.

**Lesson 2:** The query router is essential. A financial RAG system without routing sends every numerical query to vector search, which returns text chunks that may contain the right number but in a paragraph, requiring the LLM to parse it. SQL lookup is 10× more reliable for precise figures.

**Lesson 3:** Chart captions need zero-temperature generation. When temperature > 0.1, the LLM occasionally "rounds" numbers in chart descriptions. "~$4.2 billion" becomes "$4 billion" and analysts notice. Use temperature=0.0 for all financial figure generation.

**Lesson 4:** Earnings call Q&A sections are gold for sentiment and risk signal, but analysts often want to know *who asked* a question (the analyst firm) as much as what was answered. Speaker attribution in chunking metadata unlocked a whole class of queries: "What did Goldman Sachs analysts ask Apple about?"

---

## Interview Questions This Case Study Prepares You For

**"How would you handle financial data where precise numbers matter?"**
Answer: Multi-pronged: (1) XBRL structured extraction for SEC filings — bypass PDF parsing entirely, (2) store metrics in a relational database for exact lookup, (3) query router to send numerical queries to SQL not vector search, (4) post-generation numerical verification layer.

**"How would you build RAG for documents with many tables and charts?"**
Answer: Tables — NL serialization per row for retrieval embedding, store markdown table in metadata for LLM reading, use XBRL when available. Charts — extract as images, caption with vision model at temperature=0, index captions as text chunks.

**"How would you handle a query that requires comparing data across many documents?"**
Answer: Query router classifies it as cross-company comparison. SQL query retrieves the structured metrics for all requested companies from the financial_metrics table. Results passed as structured data to LLM for comparison and narrative generation. Vector search used only for qualitative context.