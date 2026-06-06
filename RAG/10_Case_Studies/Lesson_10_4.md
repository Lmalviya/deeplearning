# Case Study 4 — Codebase Assistant: Code Chunking, Cross-File Context, and Tool Use

---

## Problem Statement

A software company with 800 engineers wants a codebase assistant that answers questions about their monorepo. Engineers spend hours reading code they did not write — understanding unfamiliar services, debugging issues in legacy code, understanding how APIs are used across the codebase.

The codebase:
- 2.3 million lines of code across 18,000 files.
- Primary languages: Python (60%), TypeScript (25%), Go (10%), SQL (5%).
- Monorepo structure: 45 services, each with its own directory.
- Git history: 8 years, 95,000 commits.
- Documentation: sparse. 30% of functions have docstrings. Service-level README files are often outdated.
- Dependencies: complex cross-service dependencies via internal APIs and shared libraries.

The use cases:
- "How does the authentication service handle JWT refresh tokens?"
- "Which services depend on the UserPaymentService API?"
- "Show me all the places we handle rate limit errors from the payment gateway."
- "What does this function do, and what are its callers?"
- "How should I add a new field to the UserProfile model?"

---

## Why Code RAG Is Different

Text documents are independent — a policy document makes sense on its own. Code is a deeply interconnected graph. A function's meaning depends on the functions it calls, the classes it inherits from, the interfaces it implements, and the constants it references. Chunking code naively destroys this graph structure.

Three specific challenges:

**1. Chunk boundaries must respect code structure.** Splitting a function in the middle of its body makes the chunk meaningless. Splitting a class away from its methods loses the relationship. Code must be chunked at AST (Abstract Syntax Tree) boundaries.

**2. Cross-file context is essential.** "How does `process_payment()` work?" requires understanding `process_payment()` itself, plus the helper functions it calls, plus the data models it operates on — which may span 5 different files.

**3. Code and documentation are different retrieval targets.** A question about "how to add a new API endpoint" should retrieve documentation and existing endpoint examples. A question about "what does `validate_user()` return" should retrieve the actual function implementation.

---

## Architecture Design Decisions

### Decision 1 — AST-Based Chunking

The fundamental requirement: never split a syntactic unit across chunks. Each chunk is one complete, meaningful code unit.

```python
import ast
import tree_sitter
from pathlib import Path

class ASTChunker:
    """
    Chunks code files at AST boundaries.
    Each chunk is a complete syntactic unit: function, class, method.
    """
    
    def chunk_python_file(self, file_path: str) -> list[dict]:
        with open(file_path, 'r') as f:
            source = f.read()
        
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # Fallback to recursive character splitting for unparseable files
            return self._fallback_chunk(source, file_path)
        
        chunks = []
        module_docstring = ast.get_docstring(tree)
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                
                # Only process top-level and class-level definitions
                # (not nested functions unless they're significant)
                if not self._is_significant(node, source):
                    continue
                
                chunk_text = ast.get_source_segment(source, node)
                if not chunk_text:
                    continue
                
                # Build rich context prefix
                context_header = self._build_context_header(
                    node=node,
                    file_path=file_path,
                    source=source
                )
                
                # Extract metadata
                metadata = self._extract_metadata(node, file_path, source)
                
                chunks.append({
                    "text": f"{context_header}\n\n{chunk_text}",
                    "raw_code": chunk_text,
                    "metadata": metadata
                })
        
        return chunks
    
    def _build_context_header(self, node, file_path: str, source: str) -> str:
        """
        Build a natural language header describing the code unit.
        This improves embedding quality — the header provides semantic context
        that the raw code alone may not.
        """
        file_name = Path(file_path).name
        service = self._extract_service_name(file_path)
        
        if isinstance(node, ast.ClassDef):
            bases = [b.id for b in node.bases if hasattr(b, 'id')]
            base_str = f" (extends: {', '.join(bases)})" if bases else ""
            header = f"# File: {file_name} | Service: {service}\n# Class: {node.name}{base_str}"
        
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            is_async = isinstance(node, ast.AsyncFunctionDef)
            args = [a.arg for a in node.args.args]
            return_annotation = ""
            if node.returns:
                return_annotation = f" -> {ast.unparse(node.returns)}"
            
            docstring = ast.get_docstring(node) or ""
            header = (
                f"# File: {file_name} | Service: {service}\n"
                f"# {'async ' if is_async else ''}Function: {node.name}({', '.join(args)}){return_annotation}\n"
                f"# Docstring: {docstring[:200]}" if docstring else ""
            )
        
        return header
    
    def _extract_metadata(self, node, file_path: str, source: str) -> dict:
        """Extract metadata for filtering and relationship building."""
        
        # Find all function calls within this node
        called_functions = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    called_functions.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    called_functions.append(f"{child.func.attr}")
        
        # Find imports used
        imports_used = self._extract_imports(source, node)
        
        return {
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "service": self._extract_service_name(file_path),
            "language": "python",
            "unit_type": type(node).__name__,   # "FunctionDef", "ClassDef", etc.
            "unit_name": node.name,
            "line_start": node.lineno,
            "line_end": node.end_lineno,
            "called_functions": list(set(called_functions))[:20],  # Cap for storage
            "imports_used": imports_used[:10],
            "is_test": "test" in file_path.lower() or node.name.startswith("test_"),
            "is_async": isinstance(node, ast.AsyncFunctionDef),
            "has_docstring": bool(ast.get_docstring(node)),
            "token_count": len(source[node.col_offset:node.end_col_offset].split())
        }
    
    def _extract_service_name(self, file_path: str) -> str:
        """Extract service name from file path."""
        parts = Path(file_path).parts
        # Assumes monorepo structure: /repo/services/{service_name}/...
        if "services" in parts:
            idx = list(parts).index("services")
            if idx + 1 < len(parts):
                return parts[idx + 1]
        return "unknown"
```

**TypeScript and Go chunking:**

```python
from tree_sitter import Language, Parser

class TreeSitterChunker:
    """
    Uses tree-sitter for multi-language AST parsing.
    Handles TypeScript, Go, Java, Rust, and more.
    """
    
    def __init__(self):
        # Build languages (install tree-sitter grammars)
        self.parsers = {
            "typescript": self._build_parser("typescript"),
            "go": self._build_parser("go"),
            "javascript": self._build_parser("javascript")
        }
    
    def chunk_typescript_file(self, file_path: str) -> list[dict]:
        with open(file_path, 'rb') as f:
            source = f.read()
        
        parser = self.parsers["typescript"]
        tree = parser.parse(source)
        
        chunks = []
        
        # Query for function and class declarations
        query = self.parsers["typescript"].language.query("""
            (function_declaration name: (identifier) @name) @func
            (class_declaration name: (type_identifier) @class_name) @class
            (method_definition name: (property_identifier) @method_name) @method
            (arrow_function) @arrow
        """)
        
        matches = query.matches(tree.root_node)
        
        for pattern_idx, match in matches:
            for capture_name, nodes in match.items():
                for node in nodes:
                    if "func" in capture_name or "class" in capture_name or "method" in capture_name:
                        chunk_text = source[node.start_byte:node.end_byte].decode('utf-8')
                        
                        if len(chunk_text.split()) < 5:
                            continue  # Skip trivial chunks
                        
                        chunks.append({
                            "text": chunk_text,
                            "metadata": {
                                "file_path": file_path,
                                "language": "typescript",
                                "unit_type": capture_name,
                                "line_start": node.start_point[0],
                                "line_end": node.end_point[0],
                                "service": self._extract_service_name(file_path)
                            }
                        })
        
        return chunks
```

### Decision 2 — Embedding Strategy for Code

Code embedding requires a different approach than text embedding.

**Model choice:** `code-search-net` family models or OpenAI's `text-embedding-3-large` (which was trained on code). Dedicated code models like `CodeBERT` and `UniXcoder` are specifically trained for code retrieval tasks.

**Asymmetric encoding:** Natural language questions and code have very different distributions. Some models support asymmetric encoding with different instructions for query and document:

```python
# For indexing code
code_prefix = "Represent this code for retrieval: "

# For encoding natural language queries about code
query_prefix = "Represent this query for searching code documentation: "

# E.g., with e5-mistral-7b-instruct
chunk_embedding = model.encode(f"{code_prefix}{chunk['text']}")
query_embedding = model.encode(f"{query_prefix}{user_query}")
```

**Hybrid: embed both code and docstring separately:**

For functions with docstrings, create two chunks:
1. The code implementation (for "how does X work" queries).
2. The docstring/comment (for "what does X do" queries).

Link them via shared `chunk_family_id`. At retrieval, if either returns, fetch both.

### Decision 3 — Call Graph Index for Cross-File Context

The key to answering "which services use this function" and "what does this function depend on" questions is a call graph — a graph of function call relationships.

```python
import networkx as nx

class CallGraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()
    
    def build_from_chunks(self, all_chunks: list[dict]):
        """
        Build a call graph from all indexed code chunks.
        """
        # Add all functions as nodes
        for chunk in all_chunks:
            if chunk["metadata"].get("unit_type") in ["FunctionDef", "AsyncFunctionDef"]:
                func_id = self._make_func_id(chunk)
                self.graph.add_node(func_id, **chunk["metadata"])
        
        # Add edges for call relationships
        for chunk in all_chunks:
            caller_id = self._make_func_id(chunk)
            
            for called_name in chunk["metadata"].get("called_functions", []):
                # Find the chunk that defines this called function
                callee_chunk = self._find_definition(called_name, all_chunks)
                
                if callee_chunk:
                    callee_id = self._make_func_id(callee_chunk)
                    self.graph.add_edge(
                        caller_id,  # caller → callee
                        callee_id,
                        relationship="calls"
                    )
    
    def get_callers(self, func_id: str, max_depth: int = 2) -> list[str]:
        """Find all functions that call this function."""
        callers = []
        for depth in range(1, max_depth + 1):
            ancestors = nx.ancestors(self.graph, func_id)
            callers.extend(list(ancestors))
        return list(set(callers))
    
    def get_callees(self, func_id: str, max_depth: int = 2) -> list[str]:
        """Find all functions this function calls."""
        return list(nx.descendants(self.graph, func_id))
    
    def find_cross_service_dependencies(self) -> list[dict]:
        """Find all cross-service function calls."""
        cross_service = []
        for caller_id, callee_id in self.graph.edges():
            caller_service = self.graph.nodes[caller_id].get("service")
            callee_service = self.graph.nodes[callee_id].get("service")
            
            if caller_service and callee_service and caller_service != callee_service:
                cross_service.append({
                    "caller": caller_id,
                    "callee": callee_id,
                    "caller_service": caller_service,
                    "callee_service": callee_service
                })
        
        return cross_service
```

**Using the call graph at retrieval time:**

```python
async def retrieve_with_call_graph_expansion(
    query: str,
    retriever,
    call_graph: CallGraphBuilder,
    k: int = 5
) -> list[dict]:
    """
    Retrieve code chunks and expand context using the call graph.
    """
    
    # Step 1: Standard vector retrieval
    initial_results = await retriever.retrieve(query, k=k)
    
    expanded_chunks = list(initial_results)
    seen_ids = {r["chunk_id"] for r in initial_results}
    
    # Step 2: For each retrieved function, also fetch its direct callees
    for result in initial_results[:3]:  # Only expand top 3 to avoid bloat
        func_id = result["metadata"].get("unit_name")
        service = result["metadata"].get("service")
        
        if not func_id:
            continue
        
        node_id = f"{service}.{func_id}"
        
        # Get direct callees (one hop in call graph)
        callees = call_graph.get_callees(node_id, max_depth=1)
        
        for callee_id in callees[:3]:  # Limit to 3 callees per function
            callee_chunk = await fetch_chunk_by_func_id(callee_id)
            
            if callee_chunk and callee_chunk["chunk_id"] not in seen_ids:
                expanded_chunks.append({
                    **callee_chunk,
                    "retrieved_as": "call_graph_expansion",
                    "expansion_reason": f"Called by {func_id}"
                })
                seen_ids.add(callee_chunk["chunk_id"])
    
    return expanded_chunks
```

### Decision 4 — Agentic Code Q&A for Complex Questions

Some questions cannot be answered with a single retrieval pass:

- "Which services will be affected if I change the signature of `validate_user()`?"
- "Walk me through the complete flow of a payment request from API to database."

These require multi-hop reasoning across the call graph — exactly what an agent excels at.

```python
CODE_AGENT_TOOLS = [
    {
        "name": "search_code",
        "description": "Search for code functions, classes, or patterns. Use for semantic questions about what code does.",
        "parameters": {
            "query": "string — natural language description of what to find",
            "language": "string — optional: python, typescript, go",
            "service": "string — optional: limit to a specific service"
        }
    },
    {
        "name": "get_function_definition",
        "description": "Get the implementation of a specific function by name.",
        "parameters": {
            "function_name": "string",
            "service": "string — optional"
        }
    },
    {
        "name": "find_callers",
        "description": "Find all functions that call a specific function. Use for impact analysis.",
        "parameters": {
            "function_name": "string",
            "max_depth": "int — default 2"
        }
    },
    {
        "name": "find_callees",
        "description": "Find all functions called by a specific function. Use to understand dependencies.",
        "parameters": {
            "function_name": "string"
        }
    },
    {
        "name": "search_by_pattern",
        "description": "Search for a code pattern using regex across the codebase.",
        "parameters": {
            "pattern": "string — regex pattern",
            "file_pattern": "string — optional glob pattern"
        }
    },
    {
        "name": "get_file_structure",
        "description": "Get the list of functions and classes in a file.",
        "parameters": {
            "file_path": "string"
        }
    }
]
```

**Example agent trace for "walk me through a payment request flow":**

```
Thought: I need to find the entry point for payment requests — likely an API endpoint.
Action: search_code(query="payment request API endpoint handler")
Observation: Found PaymentController.create_payment() in services/payments/api/controllers.py

Thought: Let me see the full implementation of this entry point.
Action: get_function_definition(function_name="create_payment", service="payments")
Observation: [function implementation — calls validate_payment_request() and PaymentService.process()]

Thought: I need to follow the call to PaymentService.process()
Action: get_function_definition(function_name="process", service="payments")
Observation: [calls PaymentGatewayClient.charge() and writes to PaymentRepository]

Thought: And the repository write — let me see that too.
Action: get_function_definition(function_name="save", service="payments")
Observation: [SQL insert to payments table via SQLAlchemy]

Thought: I now have the complete flow. Let me synthesize.
Answer: The payment flow is: 
1. PaymentController.create_payment() — validates request, authenticates user
2. PaymentService.process() — business logic, rate limiting
3. PaymentGatewayClient.charge() — external API call to Stripe
4. PaymentRepository.save() — persists to PostgreSQL payments table
```

### Decision 5 — Code-Specific Prompt Design

```python
CODE_ASSISTANT_SYSTEM_PROMPT = """You are a senior software engineer assistant with full 
access to the company's codebase. You help engineers understand, navigate, and modify code.

WHEN ANSWERING CODE QUESTIONS:
- Reference actual code. Quote function signatures, variable names, and class names exactly.
- Specify file paths: "In services/payments/api/controllers.py, line 142..."
- When explaining a flow, use the actual function names in the sequence.
- Note important dependencies: "This requires the PaymentGatewayConfig to be initialized."
- Flag deprecated patterns: "Note: UserV1 is deprecated. Use UserV2 for new code."

WHEN SUGGESTING CODE CHANGES:
- Show the change in context (surrounding code, not just the changed lines).
- Note which other functions/services the change may affect.
- Mention relevant tests: "The existing tests in test_payment_service.py should cover this."

WHEN YOU DON'T KNOW:
- Say so explicitly: "I don't see a definition for X in the indexed codebase."
- Suggest where to look: "This may be in a vendor package or dynamically generated."

CONTEXT CODE:
{formatted_context}"""
```

### Decision 6 — Git History as a Retrieval Source

Git commit history contains invaluable context: why was this code changed, what bug was it fixing, who knows about this area.

```python
import subprocess

def extract_git_metadata(file_path: str, repo_root: str) -> dict:
    """
    Extract relevant git history for a file.
    """
    
    # Last modification: who, when, why
    last_commit = subprocess.run(
        ["git", "log", "--follow", "-1", "--format=%H|%an|%ae|%ad|%s", file_path],
        capture_output=True, text=True, cwd=repo_root
    ).stdout.strip()
    
    if last_commit:
        parts = last_commit.split("|")
        last_commit_info = {
            "hash": parts[0] if len(parts) > 0 else None,
            "author": parts[1] if len(parts) > 1 else None,
            "email": parts[2] if len(parts) > 2 else None,
            "date": parts[3] if len(parts) > 3 else None,
            "message": parts[4] if len(parts) > 4 else None
        }
    else:
        last_commit_info = {}
    
    # Most frequent contributors (code owners)
    contributors = subprocess.run(
        ["git", "log", "--follow", "--format=%an", file_path],
        capture_output=True, text=True, cwd=repo_root
    ).stdout.strip().split('\n')
    
    from collections import Counter
    contributor_counts = Counter(contributors)
    top_contributors = [name for name, _ in contributor_counts.most_common(3)]
    
    return {
        "last_commit": last_commit_info,
        "top_contributors": top_contributors,
        "code_owners": top_contributors[:2]  # Primary owners for routing
    }
```

Store git metadata in chunk metadata. Enables queries like "who wrote the authentication service?" and routes escalations to code owners.

---

## Indexing Pipeline

```
Source code files (git clone or filesystem mount)
    ↓
Language detection (by file extension)
    ↓
AST parsing (Python: ast module, Others: tree-sitter)
    ↓
Chunk at AST boundaries (function, class, method)
    ↓
Context header generation (file path, service, signature, docstring)
    ↓
Git metadata extraction (last commit, code owners)
    ↓
Call relationship extraction (called_functions list)
    ↓
Embedding (code-specific model, asymmetric encoding)
    ↓
Upsert to Qdrant
    ↓
Build/update call graph (NetworkX, stored separately)
```

**Index update on git push:**

```python
# CI/CD webhook: trigger re-indexing of changed files on every push to main
async def handle_git_push(push_event: dict):
    changed_files = push_event["commits_changed_files"]
    
    for file_path in changed_files:
        if is_code_file(file_path):
            # Re-index this file
            await index_file(file_path, priority="high")
            
            # Update call graph edges for this file
            await update_call_graph_for_file(file_path)
            
            # Invalidate any cached answers that referenced this file
            await invalidate_cache_for_file(file_path)
```

---

## Evaluation

Code RAG evaluation requires code-specific metrics:

```python
CODE_EVAL_QUERIES = [
    # Function lookup
    {"query": "How does JWT token refresh work?",
     "expected_chunk_ids": ["auth_service.refresh_token", "auth_service.verify_token"],
     "query_type": "implementation_lookup"},
    
    # Impact analysis
    {"query": "What will break if I change the signature of validate_user()?",
     "expected_answer_contains": ["callers", "services"],
     "query_type": "impact_analysis"},
    
    # Pattern search
    {"query": "Where do we handle database connection timeouts?",
     "expected_chunk_ids": ["db.connection_pool", "db.retry_handler"],
     "query_type": "pattern_search"},
    
    # Cross-service
    {"query": "How does the analytics service get user data?",
     "expected_answer_mentions_services": ["analytics", "user"],
     "query_type": "cross_service"}
]
```

---

## Lessons Learned

**Lesson 1:** The context header on each chunk is the most impactful change over naive chunking. Simply prepending "File: auth/service.py | Function: validate_token(user_id: str, token: str) -> bool" dramatically improves retrieval — natural language queries about "token validation" now match the right chunks even without knowing the exact function name.

**Lesson 2:** Test files must be either excluded or clearly tagged. Engineers asking "how do I use the PaymentService" do not want to see test mocks and fixtures. Filter test files out by default; include them only when the query explicitly asks about testing.

**Lesson 3:** The call graph becomes stale the moment code changes. A fast call graph update pipeline (triggered on every push to main) is necessary. Stale call graphs return wrong callers/callees.

**Lesson 4:** Engineers trust the tool more when it gives them the exact file path and line number. "See services/auth/token.py line 142" builds more trust than "I found something relevant." Always include file path and line range in the response.

---

## Interview Questions This Case Study Prepares You For

**"How do you chunk code for RAG?"**
Answer: AST-based chunking at function/class/method boundaries using Python's ast module or tree-sitter for multi-language support. Never split within a syntactic unit. Prepend a context header (file path, service, signature, docstring) to each chunk to improve embedding quality for natural language queries.

**"How do you handle cross-file dependencies in code RAG?"**
Answer: Build a call graph from the extracted `called_functions` metadata. At retrieval time, expand context by fetching direct callees (one hop) for the top retrieved functions. For complex multi-hop questions, use an agent with call graph traversal tools.

**"How do you keep a codebase index fresh?"**
Answer: Git push webhooks trigger re-indexing of changed files. Each re-index also updates the call graph edges for that file and invalidates cached answers referencing that file.