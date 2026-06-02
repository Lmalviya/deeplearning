# Lesson 5.3 — Agentic RAG: Tool-Calling, Multi-Step Reasoning, and ReAct Loops

---

## The Fundamental Shift: From Pipeline to Agent

Every RAG system we have built so far follows a fixed pipeline. The sequence of operations is predetermined: query understanding → retrieval → re-ranking → generation. The system follows this path for every query, regardless of what the query actually requires.

This works well for queries that fit the pipeline's assumptions: there is a single retrieval step needed, the answer lives in the local knowledge base, and the question can be answered in one generation pass.

It breaks down for a class of queries that require adaptive, multi-step information gathering:

- "Compare the revenue growth of our top three clients over the last two years, then identify which grew fastest relative to their industry."
- "Find all contracts with Company X, check if any have expired, and summarize the renewal terms for those that are still active."
- "What does our internal policy say about remote work, and how does it compare to what other tech companies are currently doing?"

These queries need: multiple retrieval passes with different queries, external data sources, intermediate reasoning steps, and decisions at runtime about what to look for next based on what was found.

An agent handles this by treating retrieval as one tool among many, reasoning about which tools to use and in what order, observing the results of each tool call, and iterating until the question is fully answered.

---

## The ReAct Framework

ReAct (Reason + Act, Yao et al., 2022) is the foundational pattern for agentic RAG. The agent interleaves reasoning steps and action steps in an alternating loop.

At each step, the agent:
1. **Reasons:** Thinks about what it knows, what it still needs, and what action to take next.
2. **Acts:** Executes a tool call (retrieval, web search, calculation, API call, etc.).
3. **Observes:** Receives the tool's output.
4. Repeats until it has enough information to produce a final answer.

A ReAct trace for a multi-step query:

```
Question: "What is the total contract value of all active vendor agreements 
with Acme Corp, and when does the earliest one expire?"

Thought 1: I need to find all active vendor agreements with Acme Corp.
Action 1: search_documents(query="vendor agreement Acme Corp active", filter={"status": "active"})
Observation 1: Found 3 contracts: [Contract A: $500K, expires 2025-06, Contract B: $1.2M, expires 2026-01, Contract C: $800K, expires 2024-12]

Thought 2: I have all three contracts. I can now calculate the total and find the earliest expiry.
Action 2: calculate(operation="sum", values=[500000, 1200000, 800000])
Observation 2: 2500000

Thought 3: Total is $2.5M. Earliest expiry is Contract C at 2024-12. I have enough to answer.
Final Answer: The total contract value of all active Acme Corp vendor agreements is $2.5M. 
The earliest expiration is Contract C, expiring in December 2024.
```

Each Thought is the agent's internal reasoning. Each Action is a tool call. Each Observation is the tool's output. The agent continues until it reaches a final answer.

---

## Building an Agentic RAG System

### Defining the Tool Set

The agent's capabilities are defined by the tools it has access to. For a document RAG agent, a minimal useful tool set:

```python
from openai import OpenAI
import json

client = OpenAI()

# Tool definitions (OpenAI function calling format)
RAG_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search the internal document knowledge base for relevant information. Use this for questions about internal policies, contracts, procedures, and company-specific information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query. Be specific and use keywords likely to appear in relevant documents."
                    },
                    "document_type": {
                        "type": "string",
                        "enum": ["policy", "contract", "invoice", "report", "procedure", "any"],
                        "description": "Filter by document type. Use 'any' if unsure."
                    },
                    "date_after": {
                        "type": "string",
                        "description": "Only return documents modified after this date (YYYY-MM-DD format). Optional."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default 5, max 20).",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the public web for information not available in internal documents. Use for general knowledge, industry benchmarks, competitor information, and recent news.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The web search query."
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5).",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform mathematical calculations. Use for summing values, computing percentages, date arithmetic, and other numerical operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A mathematical expression to evaluate (e.g., '500000 + 1200000 + 800000', '(45 - 38) / 38 * 100')."
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_by_id",
            "description": "Retrieve the full content of a specific document by its ID. Use when you have a document ID from a previous search and need to read the full content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "The document ID to retrieve."
                    }
                },
                "required": ["doc_id"]
            }
        }
    }
]
```

### Tool Execution Functions

```python
async def execute_tool(tool_name: str, tool_args: dict, tool_registry: dict) -> str:
    """
    Execute a tool call and return the result as a string.
    """
    if tool_name not in tool_registry:
        return f"Error: Unknown tool '{tool_name}'"
    
    try:
        result = await tool_registry[tool_name](**tool_args)
        
        # Convert result to string for inclusion in conversation
        if isinstance(result, (dict, list)):
            return json.dumps(result, indent=2)
        return str(result)
    
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"


# Tool implementations
async def search_documents_tool(
    query: str,
    document_type: str = "any",
    date_after: str = None,
    max_results: int = 5,
    retriever=None
) -> list[dict]:
    """Execute document search using the RAG retriever."""
    
    metadata_filter = {}
    if document_type != "any":
        metadata_filter["document_type"] = document_type
    if date_after:
        metadata_filter["modified_date"] = {"$gte": date_after}
    
    results = await retriever.retrieve(
        query=query,
        metadata_filter=metadata_filter if metadata_filter else None,
        k=max_results
    )
    
    return [
        {
            "chunk_id": r["chunk_id"],
            "doc_id": r["metadata"]["doc_id"],
            "doc_title": r["metadata"].get("doc_title", "Unknown"),
            "section": r["metadata"].get("heading_path", ""),
            "text": r["text"][:500],  # Truncate for agent's observation window
            "relevance_score": r.get("rerank_score", 0)
        }
        for r in results
    ]


async def calculate_tool(expression: str) -> dict:
    """Safely evaluate a mathematical expression."""
    import ast
    import operator
    
    # Safe evaluation — only allow basic math operations
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg
    }
    
    def safe_eval(node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            return allowed_operators[type(node.op)](safe_eval(node.left), safe_eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            return allowed_operators[type(node.op)](safe_eval(node.operand))
        else:
            raise ValueError(f"Unsupported operation: {type(node)}")
    
    try:
        tree = ast.parse(expression, mode='eval')
        result = safe_eval(tree.body)
        return {"result": result, "expression": expression}
    except Exception as e:
        return {"error": str(e), "expression": expression}
```

### The Agent Loop

```python
class RAGAgent:
    def __init__(
        self,
        llm_client,
        tool_registry: dict,
        max_iterations: int = 10,
        model: str = "gpt-4o"
    ):
        self.llm = llm_client
        self.tools = tool_registry
        self.max_iterations = max_iterations
        self.model = model
        
        self.system_prompt = """You are a helpful research assistant with access to 
tools for searching internal documents, the web, and performing calculations.

Guidelines:
- Use tools to gather information before answering
- Be methodical: think through what you need, then gather it step by step
- For complex questions, break them into sub-questions and address each
- Always cite your sources when providing information from documents
- If you cannot find information after searching, say so clearly
- Calculate rather than estimate when precise numbers are needed"""
    
    async def run(self, query: str) -> dict:
        """
        Run the agent loop until a final answer is produced or max iterations reached.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query}
        ]
        
        tool_calls_made = []
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            
            # Call LLM — may generate tool calls or a final answer
            response = await self.llm.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=RAG_AGENT_TOOLS,
                tool_choice="auto",  # LLM decides whether to call tools
                max_tokens=1000
            )
            
            message = response.choices[0].message
            messages.append(message)  # Add assistant response to history
            
            # Check if the LLM made tool calls
            if message.tool_calls:
                # Execute each tool call
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    # Execute the tool
                    result = await execute_tool(
                        tool_name, tool_args, self.tools
                    )
                    
                    tool_calls_made.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "result_preview": result[:200]
                    })
                    
                    # Add tool result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
            
            else:
                # No tool calls — LLM produced a final answer
                return {
                    "answer": message.content,
                    "iterations": iteration,
                    "tool_calls": tool_calls_made,
                    "status": "completed"
                }
        
        # Max iterations reached without a final answer
        return {
            "answer": "I reached the maximum number of steps without completing this task. Please try rephrasing the question or breaking it into smaller parts.",
            "iterations": iteration,
            "tool_calls": tool_calls_made,
            "status": "max_iterations_reached"
        }
```

---

## Agent Failure Modes

Agentic RAG introduces failure modes that fixed pipelines do not have.

### Looping

The agent calls the same tool with the same arguments repeatedly, not making progress. Common cause: the first tool call returned insufficient results and the agent does not recognize this.

```python
def detect_looping(tool_calls_made: list[dict], window: int = 3) -> bool:
    """
    Check if the agent is repeating the same tool calls.
    """
    if len(tool_calls_made) < window:
        return False
    
    recent = tool_calls_made[-window:]
    
    # Check if all recent calls are the same tool with the same args
    first = recent[0]
    is_looping = all(
        t["tool"] == first["tool"] and t["args"] == first["args"]
        for t in recent
    )
    
    return is_looping
```

**Fix:** Add a loop detection check in the agent loop. If looping is detected, inject a message prompting the agent to try a different approach.

### Hallucinated Tool Results

The LLM "hallucinates" a tool call by generating text that looks like a tool result without actually calling the tool. This happens when the model's training data contained examples of tool use and the model pattern-matches without actually executing.

**Fix:** Ensure tool calls go through the proper API mechanism (function calling), not as text generation. Never parse tool call results from the LLM's text output — only accept results through the official tool call response structure.

### Tool Result Misinterpretation

The agent calls the right tool but misinterprets the result. A search returns 0 results, and the agent concludes "the document says X" rather than "no documents were found."

```python
def validate_tool_result(tool_name: str, raw_result: str) -> dict:
    """
    Validate and enrich tool results before adding to conversation.
    """
    if tool_name == "search_documents":
        try:
            results = json.loads(raw_result)
            if isinstance(results, list) and len(results) == 0:
                return {
                    "status": "no_results",
                    "message": "No documents found matching this query.",
                    "suggestion": "Try different keywords or broaden the search."
                }
            return {"status": "success", "results": results, "count": len(results)}
        except json.JSONDecodeError:
            return {"status": "error", "raw": raw_result}
    
    return {"status": "success", "raw": raw_result}
```

### Runaway Tool Use

The agent makes far too many tool calls, either by excessive exploration or by not recognizing when it has sufficient information.

**Fix:** Hard cap on iterations (`max_iterations`). Also add a tool call budget per tool type:

```python
class BudgetedRAGAgent(RAGAgent):
    def __init__(self, *args, tool_budgets: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tool_budgets = tool_budgets or {
            "search_documents": 10,
            "web_search": 5,
            "calculate": 20,
            "get_document_by_id": 5
        }
        self.tool_usage = {k: 0 for k in self.tool_budgets}
    
    def is_tool_allowed(self, tool_name: str) -> bool:
        budget = self.tool_budgets.get(tool_name, float('inf'))
        used = self.tool_usage.get(tool_name, 0)
        return used < budget
    
    def record_tool_use(self, tool_name: str):
        self.tool_usage[tool_name] = self.tool_usage.get(tool_name, 0) + 1
```

---

## Multi-Agent RAG

For very complex tasks, a single agent may not be sufficient. Multi-agent architectures split the work across specialized agents.

```
Orchestrator Agent
    ├── Research Agent (specialized in document retrieval)
    ├── Analysis Agent (specialized in data analysis and calculation)
    └── Synthesis Agent (specialized in writing and summarization)
```

```python
class OrchestratorAgent:
    """
    Routes subtasks to specialized agents and synthesizes results.
    """
    
    def __init__(self, research_agent, analysis_agent, llm_client):
        self.research = research_agent
        self.analysis = analysis_agent
        self.llm = llm_client
    
    async def run(self, query: str) -> dict:
        # Step 1: Decompose the query into subtasks
        decomposition = await self._decompose_query(query)
        
        # Step 2: Route subtasks to appropriate agents
        subtask_results = {}
        
        for subtask in decomposition["subtasks"]:
            if subtask["type"] == "research":
                result = await self.research.run(subtask["query"])
                subtask_results[subtask["id"]] = result
            
            elif subtask["type"] == "analysis":
                # Pass research results as context to analysis agent
                context = {
                    k: v["answer"] for k, v in subtask_results.items()
                    if k in subtask.get("depends_on", [])
                }
                result = await self.analysis.run(subtask["query"], context=context)
                subtask_results[subtask["id"]] = result
        
        # Step 3: Synthesize all subtask results into final answer
        final_answer = await self._synthesize(query, subtask_results)
        
        return {
            "answer": final_answer,
            "subtask_results": subtask_results,
            "decomposition": decomposition
        }
    
    async def _decompose_query(self, query: str) -> dict:
        prompt = f"""Break this complex query into subtasks for specialized agents.

Query: {query}

Return JSON with subtasks, each having:
- id: unique identifier
- type: "research" (document/web search) or "analysis" (calculation/comparison)
- query: the specific question for this subtask
- depends_on: list of subtask IDs this depends on (empty list if independent)

Keep subtasks minimal — only create separate tasks when truly needed."""
        
        response = await self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=500,
            temperature=0.0
        )
        
        return json.loads(response.choices[0].message.content)
    
    async def _synthesize(self, original_query: str, subtask_results: dict) -> str:
        results_text = "\n\n".join([
            f"Subtask {k}: {v['answer']}"
            for k, v in subtask_results.items()
        ])
        
        response = await self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": f"Original question: {original_query}\n\nSubtask results:\n{results_text}\n\nSynthesize a complete answer."
                }
            ],
            max_tokens=1000,
            temperature=0.1
        )
        
        return response.choices[0].message.content
```

---

## When to Use Agentic RAG

Agentic RAG adds significant complexity. Do not use it by default.

**Use agentic RAG when:**
- Queries require multiple distinct retrieval operations.
- Intermediate results determine what to retrieve next.
- The answer requires both internal documents and external data.
- The query requires tool use beyond text retrieval (calculation, API calls, code execution).
- Query complexity is genuinely high and unpredictable.

**Do NOT use agentic RAG when:**
- Queries are well-structured and predictable — a fixed pipeline handles them well.
- Latency is critical — agents are inherently multi-round and slow.
- Determinism is required — agents are non-deterministic and hard to test.
- The corpus is the only data source needed — no external tools required.
- You need guaranteed behavior — agents can take unexpected paths.

> **Interview note:** "Would you use an agent or a pipeline for your RAG system?" — The answer: "It depends on query complexity. For predictable, well-defined queries over a single knowledge base, a fixed pipeline is faster, cheaper, and more reliable. Agents are justified when queries are genuinely multi-step, require multiple data sources, or need adaptive retrieval strategies based on intermediate results. I would default to a pipeline and add agentic capabilities only where the pipeline demonstrably fails."

---

## Evaluation Challenges for Agentic RAG

Standard RAG evaluation measures: was the final answer correct? Agentic RAG requires evaluating the entire reasoning trace.

```python
async def evaluate_agent_trace(
    query: str,
    expected_answer: str,
    agent_result: dict,
    llm_client
) -> dict:
    """
    Evaluate agent performance across multiple dimensions.
    """
    
    tool_calls = agent_result.get("tool_calls", [])
    answer = agent_result.get("answer", "")
    
    # 1. Answer correctness
    correctness_prompt = f"""Does this answer correctly address the question?
Question: {query}
Expected: {expected_answer}
Agent answer: {answer}
Score 0-10 and explain."""
    
    # 2. Tool use efficiency — did the agent use the minimum necessary tools?
    efficiency_note = (
        "efficient" if len(tool_calls) <= 5
        else "excessive" if len(tool_calls) > 10
        else "acceptable"
    )
    
    # 3. Reasoning quality — did the agent reason correctly between steps?
    # This requires inspecting the full message trace
    
    return {
        "answer_length": len(answer.split()),
        "tool_calls_count": len(tool_calls),
        "tool_efficiency": efficiency_note,
        "iterations": agent_result.get("iterations", 0),
        "completed": agent_result.get("status") == "completed",
        "tools_used": list(set(t["tool"] for t in tool_calls))
    }
```

---

## Summary

- Agentic RAG replaces fixed pipelines with a reasoning agent that decides what tools to use and in what order at runtime. This handles queries that require multiple retrieval steps, external data, or adaptive strategy.
- ReAct (Reason + Act) is the foundational pattern: alternating thought, action, and observation steps until the question is answered.
- Define tools explicitly (function calling), implement safe execution, and handle errors gracefully. The tool set defines the agent's capabilities.
- Agent-specific failure modes: looping (repeating same calls), hallucinated tool results, result misinterpretation, and runaway tool use. Each requires explicit detection and handling.
- Multi-agent architectures split complex tasks across specialized agents coordinated by an orchestrator. Increases parallelism and specialization at the cost of coordination complexity.
- Use agents selectively — only when queries are genuinely multi-step, require multiple data sources, or need adaptive retrieval. Default to fixed pipelines for predictable query patterns.
- Evaluation of agentic RAG requires assessing the entire reasoning trace, not just the final answer.

---

## What's Next

Lesson 5.4 covers Graph RAG — building knowledge graphs from document corpora, entity linking, community summaries, and retrieving relational information that vector search fundamentally cannot handle.