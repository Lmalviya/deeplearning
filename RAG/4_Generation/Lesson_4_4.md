# Lesson 4.4 — Structured Output Generation from Retrieved Context

---

## Why Structured Output Matters in RAG

Most RAG tutorials assume the output is free-form prose. In production, this is rarely the case. Applications need:

- A JSON object with specific fields extracted from a contract.
- A table comparing features across multiple product documents.
- A structured report with predefined sections.
- A list of action items extracted from meeting notes.
- A classification label with confidence score derived from policy documents.
- A filled form where each field traces back to a specific source document.

Free-form prose is hard to process downstream. Structured outputs enable your RAG system to feed into databases, APIs, UIs, and downstream automation pipelines reliably.

The challenge: LLMs are probabilistic text generators, not reliable JSON serializers. Getting them to produce valid, schema-compliant structured output consistently — especially when the structure must be grounded in retrieved context rather than invented — requires deliberate engineering.

---

## The Three Failure Modes of Structured Output from RAG

Before designing solutions, understand what breaks:

**Failure 1 — Schema non-compliance.** The LLM produces output that does not match the requested schema. Missing required fields, wrong data types, extra fields not in the schema, invalid enum values, nested objects where flat values are expected.

**Failure 2 — Hallucinated field values.** The LLM fills required fields with plausible-sounding values not present in the retrieved context. A "contract_value" field gets filled with "$500,000" because that sounds like a reasonable contract value, even though the actual retrieved context says "$1.2 million."

**Failure 3 — Field omission vs. null.** When a field value is genuinely not present in the context, the LLM must choose between leaving it null/empty (correct) or filling it with a guess (wrong). LLMs default to filling rather than nulling, which produces confident wrong structured data.

Solving all three requires: explicit schema enforcement, source-tracing instructions, and null-handling rules.

---

## Approach 1 — Structured Output with JSON Mode

Most LLM APIs provide a JSON mode that guarantees the output is valid JSON, even if it does not match your specific schema.

```python
async def extract_structured_from_context(
    query: str,
    context: str,
    schema: dict,  # JSON Schema definition
    llm_client
) -> dict:
    """
    Extract structured data from retrieved context using JSON mode.
    """
    
    schema_description = format_schema_for_prompt(schema)
    
    system_prompt = f"""You extract structured information from documents.
    
Output ONLY a JSON object matching this schema:
{schema_description}

Rules:
- Only include values explicitly stated in the provided context
- Use null for any field not found in the context
- Do not infer, calculate, or guess values
- For list fields, include all instances found in the context
- Preserve exact figures, dates, and names as they appear in the source"""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nExtract the requested information."}
        ],
        response_format={"type": "json_object"},  # JSON mode
        max_tokens=2000,
        temperature=0.0
    )
    
    import json
    raw_output = response.choices[0].message.content
    parsed = json.loads(raw_output)  # Guaranteed to be valid JSON in JSON mode
    
    return parsed


def format_schema_for_prompt(schema: dict) -> str:
    """Convert JSON schema to a human-readable description for the prompt."""
    import json
    return json.dumps(schema, indent=2)
```

**JSON mode guarantees:** Valid JSON. It does NOT guarantee your specific schema is followed — the LLM may still produce fields not in your schema or omit required ones.

---

## Approach 2 — Structured Outputs with Function Calling / Tool Use

Function calling is a stronger mechanism. You define the exact schema as a function signature, and the LLM must produce output conforming to that signature. The API validates the output against the schema before returning it.

```python
from openai import OpenAI
import json

client = OpenAI()

# Define the extraction schema as a function/tool
contract_extraction_tool = {
    "type": "function",
    "function": {
        "name": "extract_contract_details",
        "description": "Extract key details from a contract document",
        "parameters": {
            "type": "object",
            "properties": {
                "parties": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "role": {"type": "string", "enum": ["buyer", "seller", "licensor", "licensee", "service_provider", "client"]}
                        },
                        "required": ["name", "role"]
                    },
                    "description": "All parties named in the contract"
                },
                "effective_date": {
                    "type": ["string", "null"],
                    "description": "Contract effective date in ISO 8601 format (YYYY-MM-DD), or null if not specified"
                },
                "expiry_date": {
                    "type": ["string", "null"],
                    "description": "Contract expiry/termination date in ISO 8601 format, or null if not specified"
                },
                "contract_value": {
                    "type": ["string", "null"],
                    "description": "Total contract value as stated in the document (e.g., '$500,000'), or null if not stated"
                },
                "payment_terms": {
                    "type": ["string", "null"],
                    "description": "Payment terms as described in the contract, or null if not specified"
                },
                "termination_conditions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of conditions under which the contract can be terminated"
                },
                "governing_law": {
                    "type": ["string", "null"],
                    "description": "The jurisdiction and governing law, or null if not specified"
                },
                "extraction_confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "Overall confidence in the extraction quality"
                },
                "missing_fields_reason": {
                    "type": "string",
                    "description": "Explanation of why any fields are null — state which information was absent from the provided context"
                }
            },
            "required": ["parties", "effective_date", "expiry_date", "contract_value", 
                        "termination_conditions", "governing_law", "extraction_confidence",
                        "missing_fields_reason"]
        }
    }
}


async def extract_contract_with_tool_use(
    context: str
) -> dict:
    """
    Extract contract details using function calling for schema enforcement.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """Extract contract details from the provided context.
Only include values explicitly stated in the context.
Use null for any field not found. Do not infer or calculate values.
In missing_fields_reason, explain what information was absent."""
            },
            {
                "role": "user",
                "content": f"Contract context:\n{context}"
            }
        ],
        tools=[contract_extraction_tool],
        tool_choice={"type": "function", "function": {"name": "extract_contract_details"}},
        temperature=0.0
    )
    
    # Extract the function call arguments
    tool_call = response.choices[0].message.tool_calls[0]
    return json.loads(tool_call.function.arguments)
```

Function calling produces schema-compliant output because the API validates it before returning. It is the most reliable approach for structured extraction from RAG.

### The `null` vs. Hallucination Design

Note the design of the schema: every optional field has `["string", "null"]` type and explicit "or null if not specified" in the description. This pattern is critical.

Without explicit null guidance, LLMs default to filling fields. With it, they correctly return null for absent information.

Additionally, the `missing_fields_reason` required field forces the LLM to explicitly acknowledge what was missing from the context — turning silent omission into an auditable explanation.

---

## Approach 3 — Pydantic Validation with Retry

For Python applications, use Pydantic to validate the LLM's output against a schema and automatically retry on validation failure.

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List
import json

class Party(BaseModel):
    name: str
    role: str

class ContractExtraction(BaseModel):
    parties: List[Party]
    effective_date: Optional[str] = None
    expiry_date: Optional[str] = None
    contract_value: Optional[str] = None
    payment_terms: Optional[str] = None
    termination_conditions: List[str] = Field(default_factory=list)
    governing_law: Optional[str] = None
    extraction_confidence: str
    
    @validator('effective_date', 'expiry_date')
    def validate_date_format(cls, v):
        if v is None:
            return v
        import re
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', v):
            raise ValueError(f"Date must be in YYYY-MM-DD format, got: {v}")
        return v
    
    @validator('extraction_confidence')
    def validate_confidence(cls, v):
        if v not in ['high', 'medium', 'low']:
            raise ValueError(f"Confidence must be high/medium/low, got: {v}")
        return v


async def extract_with_validation_retry(
    context: str,
    llm_client,
    max_retries: int = 3
) -> ContractExtraction:
    """
    Extract structured data with Pydantic validation and automatic retry.
    On validation failure, feed the error back to the LLM for correction.
    """
    
    previous_attempt = None
    previous_error = None
    
    for attempt in range(max_retries):
        prompt_parts = [f"Context:\n{context}"]
        
        if previous_attempt and previous_error:
            prompt_parts.append(f"""
Your previous response had validation errors:
Previous response: {previous_attempt}
Errors: {previous_error}

Please correct these specific errors and try again.""")
        
        prompt_parts.append("Extract contract details as a JSON object.")
        
        response = await llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Extract contract details as JSON. Use null for missing fields."},
                {"role": "user", "content": "\n\n".join(prompt_parts)}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        raw_json = response.choices[0].message.content
        previous_attempt = raw_json
        
        try:
            data = json.loads(raw_json)
            validated = ContractExtraction(**data)
            return validated
        except (json.JSONDecodeError, Exception) as e:
            previous_error = str(e)
            if attempt == max_retries - 1:
                raise ValueError(f"Failed to extract valid structure after {max_retries} attempts: {e}")
    
    raise ValueError("Extraction failed")
```

The retry-with-error-feedback pattern significantly improves reliability. The LLM sees its exact error and can correct it specifically.

---

## Source Tracing for Structured Outputs

A structured extraction is only trustworthy if you can trace each field value back to the specific text in the source document that supports it. Without source tracing, structured extraction is a black box — you cannot audit why a field has a particular value.

```python
contract_extraction_with_sources_tool = {
    "type": "function",
    "function": {
        "name": "extract_contract_details_with_sources",
        "parameters": {
            "type": "object",
            "properties": {
                "contract_value": {
                    "type": "object",
                    "properties": {
                        "value": {"type": ["string", "null"]},
                        "source_quote": {
                            "type": ["string", "null"],
                            "description": "Exact quote from the document supporting this value"
                        },
                        "source_ref": {
                            "type": ["string", "null"],
                            "description": "Reference number [N] of the source document"
                        }
                    }
                },
                "effective_date": {
                    "type": "object",
                    "properties": {
                        "value": {"type": ["string", "null"]},
                        "source_quote": {"type": ["string", "null"]},
                        "source_ref": {"type": ["string", "null"]}
                    }
                }
                # ... same pattern for all fields
            }
        }
    }
}
```

With source tracing, each extracted field includes:
- The extracted value.
- The exact quote from the source document that contains this value.
- The reference number of the source document.

This enables auditing, debugging, and user-facing citations in structured extraction outputs. If a user questions why "contract_value" is $1.2M, you can show them the exact sentence in the contract that contains that figure.

---

## Table Generation from Multiple Documents

A common structured output pattern: generate a comparison table from multiple retrieved documents.

```python
async def generate_comparison_table(
    query: str,
    documents: list[dict],   # Each is {title, text}
    comparison_dimensions: list[str],  # What to compare across documents
    llm_client
) -> dict:
    """
    Generate a structured comparison table from multiple documents.
    
    Example: Compare vendor contracts across:
    - Payment terms
    - SLA commitments
    - Termination clauses
    - Governing law
    """
    
    # First pass: extract each dimension from each document independently
    extraction_results = {}
    
    for doc in documents:
        dimensions_schema = {
            dim: {"type": ["string", "null"], "description": f"Value of '{dim}' as stated in this document, or null if not found"}
            for dim in comparison_dimensions
        }
        
        extraction_prompt = f"""From the following document, extract these specific attributes:
{', '.join(comparison_dimensions)}

Document ({doc['title']}):
{doc['text']}

For each attribute, provide the exact value as stated in the document.
Use null if the attribute is not mentioned."""
        
        response = await llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": extraction_prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=500
        )
        
        try:
            extraction_results[doc['title']] = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            extraction_results[doc['title']] = {dim: None for dim in comparison_dimensions}
    
    # Build table structure
    table = {
        "headers": ["Attribute"] + [doc["title"] for doc in documents],
        "rows": []
    }
    
    for dimension in comparison_dimensions:
        row = [dimension]
        for doc in documents:
            value = extraction_results.get(doc["title"], {}).get(dimension)
            row.append(value if value is not None else "—")
        table["rows"].append(row)
    
    # Second pass: identify key differences and agreements
    table_text = format_table_as_text(table)
    
    analysis_prompt = f"""Based on this comparison table:
{table_text}

Provide a brief analysis:
1. Where do all documents agree?
2. Where are there significant differences?
3. Which document has the most favorable terms for a buyer?

Be specific, citing exact values from the table."""
    
    analysis_response = await llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": analysis_prompt}],
        max_tokens=600,
        temperature=0.1
    )
    
    return {
        "table": table,
        "analysis": analysis_response.choices[0].message.content,
        "raw_extractions": extraction_results
    }


def format_table_as_text(table: dict) -> str:
    """Format table dict as markdown table string."""
    headers = table["headers"]
    rows = table["rows"]
    
    header_row = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_rows = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    
    return "\n".join([header_row, separator] + data_rows)
```

---

## Report Generation with Predefined Sections

For generating structured reports (executive summaries, audit reports, due diligence memos), define the report schema explicitly and fill each section from relevant retrieved context.

```python
REPORT_SCHEMA = {
    "executive_summary": {
        "description": "2-3 paragraph overview of key findings",
        "max_tokens": 400
    },
    "key_findings": {
        "description": "Bulleted list of the most important findings",
        "format": "list",
        "max_items": 10
    },
    "risk_factors": {
        "description": "Identified risks with severity ratings",
        "format": "structured_list",
        "fields": ["risk", "severity", "mitigation"]
    },
    "recommendations": {
        "description": "Actionable recommendations based on findings",
        "format": "list",
        "max_items": 5
    }
}

async def generate_structured_report(
    topic: str,
    retrieved_chunks: list[dict],
    report_schema: dict,
    llm_client
) -> dict:
    """
    Generate a structured report by filling each section from retrieved context.
    Each section is generated independently to maintain focus.
    """
    context = "\n\n".join(c["text"] for c in retrieved_chunks)
    report = {}
    
    for section_name, section_config in report_schema.items():
        section_prompt = f"""Based on the following context, write the '{section_name}' 
section of a report about: {topic}

Section requirements: {section_config['description']}
{f"Format: {section_config.get('format', 'prose')}" if 'format' in section_config else ""}
{f"Maximum items: {section_config['max_items']}" if 'max_items' in section_config else ""}

Context:
{context}

Write ONLY the {section_name} section content. 
Only include information present in the context.
Use null or omit items where context is insufficient."""
        
        response = await llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": section_prompt}],
            max_tokens=section_config.get("max_tokens", 800),
            temperature=0.1
        )
        
        report[section_name] = response.choices[0].message.content
    
    return report
```

Section-by-section generation has an important advantage: each section gets the full context window focused on one task, rather than the LLM trying to produce all sections simultaneously. This reduces cross-section confusion and improves per-section quality.

---

## Validation Pipeline for Structured RAG Outputs

Production structured extraction needs validation beyond schema compliance. The extracted values must also be factually grounded in the retrieved context.

```python
async def validate_structured_extraction(
    extracted_data: dict,
    source_context: str,
    llm_client
) -> dict:
    """
    Validate that extracted field values are actually present in the source context.
    Flag any values that appear to be hallucinated.
    """
    
    # Check non-null fields against source context
    fields_to_check = {k: v for k, v in extracted_data.items() 
                       if v is not None and isinstance(v, (str, int, float))}
    
    validation_prompt = f"""Check whether each of these extracted values is actually 
present in or directly derivable from the source context.

Source context:
{source_context}

Extracted values to verify:
{json.dumps(fields_to_check, indent=2)}

For each value, determine:
- VERIFIED: The value appears explicitly in the context
- INFERRED: The value can be directly inferred from context (e.g., end date calculated from start + duration)
- NOT_FOUND: The value is not in the context — likely hallucinated

Respond with JSON: {{"field_name": "VERIFIED|INFERRED|NOT_FOUND", ...}}"""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": validation_prompt}],
        response_format={"type": "json_object"},
        max_tokens=500,
        temperature=0.0
    )
    
    validation_results = json.loads(response.choices[0].message.content)
    
    # Flag potentially hallucinated values
    hallucinated_fields = [
        field for field, status in validation_results.items()
        if status == "NOT_FOUND"
    ]
    
    if hallucinated_fields:
        # Null out hallucinated fields
        for field in hallucinated_fields:
            extracted_data[field] = None
        
        extracted_data["_validation_flags"] = {
            "hallucinated_fields_nulled": hallucinated_fields,
            "validation_results": validation_results
        }
    
    return extracted_data
```

This post-extraction validation catches and nulls hallucinated field values before they propagate downstream.

---

## Summary

- Structured outputs from RAG fail in three ways: schema non-compliance, hallucinated field values, and incorrect null handling. All three require deliberate engineering to prevent.
- JSON mode guarantees valid JSON but not schema compliance. Function calling / tool use enforces schema at the API level — the strongest reliability guarantee.
- Design schemas with explicit `null` types and "or null if not found" descriptions for all optional fields. Add a `missing_fields_reason` field to make absent data explicit and auditable.
- Pydantic validation with error-feedback retry handles schema compliance failures gracefully by feeding the exact error back to the LLM for correction.
- Source tracing attaches exact quotes and document references to each extracted field, enabling auditing and user-facing citations.
- For multi-document comparison tables, extract dimensions from each document independently (map step) then assemble the table structure (reduce step).
- For structured reports, generate each section independently to keep each LLM call focused on one task.
- Post-extraction validation checks whether extracted values actually appear in the source context. Null out values that cannot be verified.

---

## What's Next

Lesson 4.5 covers hallucination — causes, detection methods, and the layered mitigation strategies that go beyond just "write a better prompt."