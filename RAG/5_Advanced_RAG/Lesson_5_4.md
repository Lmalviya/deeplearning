# Lesson 5.4 — Graph RAG: Knowledge Graph Construction, Entity Linking, and Community Summaries

---

## What Vector Search Cannot Do

Vector search finds documents that are semantically similar to a query. It is fundamentally a document-to-query similarity operation. This works when the answer lives in one document or a small cluster of related documents.

It breaks down for a different class of questions — ones where the answer requires understanding the relationships between entities across many documents:

- "What are all the companies that have both a vendor relationship and a legal dispute with us?"
- "Who are the key decision-makers that appear in both our Q3 board minutes and our major contract negotiations?"
- "How did the relationship between Entity X and Entity Y evolve from 2019 to 2024 across all our documents?"
- "What are the dominant themes across our entire 50,000-document corpus?"

These questions are about the graph of relationships in your data, not about any individual document. Vector search of individual documents will return fragmented, incomplete answers because the relational structure exists across the corpus, not within any single document.

Graph RAG (Edge et al., Microsoft, 2024) addresses this by extracting a knowledge graph from the corpus and using graph-level structures — entity relationships, community clusters, community summaries — as the retrieval unit rather than document chunks.

---

## The Knowledge Graph Structure

A knowledge graph represents information as nodes (entities) and edges (relationships between entities).

**Nodes** are entities: people, organizations, products, concepts, locations, events, policies, contracts, dates.

**Edges** are relationships: "Company A acquired Company B", "Person X signed Contract Y", "Policy Z supersedes Policy W", "Event A caused Event B".

Each node and edge can have properties: a Company node might have properties `{name, industry, founded_year, headquarters}`. A "signed" edge might have properties `{date, role, document_id}`.

For a document corpus, you build this graph by extracting entities and relationships from every document, then merging entities that refer to the same real-world thing (entity resolution).

---

## Step 1 — Entity and Relationship Extraction

For each document chunk, extract entities and the relationships between them.

```python
async def extract_entities_and_relations(
    chunk_text: str,
    doc_metadata: dict,
    llm_client
) -> dict:
    """
    Extract a knowledge graph fragment from a document chunk.
    Returns entities and relationships found in this chunk.
    """
    
    prompt = f"""Extract all entities and relationships from the following text.

Text: {chunk_text}

Extract:
1. ENTITIES: Named entities (people, organizations, products, locations, dates, concepts)
2. RELATIONSHIPS: Explicit relationships between entities

Return JSON:
{{
    "entities": [
        {{
            "name": "entity name as it appears in text",
            "type": "PERSON | ORG | PRODUCT | LOCATION | DATE | CONCEPT | POLICY | CONTRACT",
            "description": "brief description based on context",
            "mentions": ["exact phrases used to refer to this entity"]
        }}
    ],
    "relationships": [
        {{
            "source": "entity name",
            "relation": "relationship type (e.g., 'acquired', 'employs', 'signed', 'supersedes', 'located_in')",
            "target": "entity name",
            "description": "brief description of this relationship",
            "evidence": "exact quote from text supporting this relationship"
        }}
    ]
}}

Only extract relationships that are explicitly stated in the text.
Do not infer relationships."""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=1000,
        temperature=0.0
    )
    
    import json
    result = json.loads(response.choices[0].message.content)
    
    # Attach source metadata to all extracted elements
    for entity in result.get("entities", []):
        entity["source_doc_id"] = doc_metadata.get("doc_id")
        entity["source_chunk_id"] = doc_metadata.get("chunk_id")
    
    for relation in result.get("relationships", []):
        relation["source_doc_id"] = doc_metadata.get("doc_id")
        relation["source_chunk_id"] = doc_metadata.get("chunk_id")
    
    return result
```

This must run on every chunk across the entire corpus. For a 10,000-chunk corpus at ~200ms per LLM call, sequential processing takes 33 minutes. Parallelize with a rate-limited async worker:

```python
async def extract_from_corpus(
    chunks: list[dict],
    llm_client,
    max_concurrent: int = 20
) -> list[dict]:
    """Extract KG fragments from all chunks with controlled concurrency."""
    from asyncio import Semaphore
    
    semaphore = Semaphore(max_concurrent)
    
    async def extract_one(chunk: dict) -> dict:
        async with semaphore:
            result = await extract_entities_and_relations(
                chunk["text"], chunk["metadata"], llm_client
            )
            return {"chunk_id": chunk["chunk_id"], **result}
    
    tasks = [extract_one(chunk) for chunk in chunks]
    return await asyncio.gather(*tasks)
```

---

## Step 2 — Entity Resolution

Across a large corpus, the same entity appears under many names: "Apple Inc.", "Apple", "AAPL", "the company" (in context referring to Apple). Without merging these, you get a fragmented graph where Apple has dozens of disconnected nodes instead of one.

Entity resolution identifies which mentions refer to the same real-world entity and merges them.

```python
async def resolve_entities(
    all_entities: list[dict],
    llm_client,
    similarity_threshold: float = 0.85
) -> dict:
    """
    Cluster entity mentions that refer to the same real-world entity.
    Returns a mapping from mention variants to canonical entity names.
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    
    # Step 1: Embed all entity names
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    
    entity_names = [e["name"] for e in all_entities]
    entity_types = [e["type"] for e in all_entities]
    
    # Embed with type context for better disambiguation
    embed_inputs = [f"{t}: {n}" for n, t in zip(entity_names, entity_types)]
    embeddings = embedder.encode(embed_inputs, normalize_embeddings=True)
    
    # Step 2: Cluster similar entities
    # Only cluster entities of the same type (don't merge a Person with an Org)
    entity_to_canonical = {}
    
    for entity_type in set(entity_types):
        type_indices = [i for i, t in enumerate(entity_types) if t == entity_type]
        
        if len(type_indices) < 2:
            # Single entity of this type — no clustering needed
            for idx in type_indices:
                entity_to_canonical[entity_names[idx]] = entity_names[idx]
            continue
        
        type_embeddings = embeddings[type_indices]
        
        # Agglomerative clustering with cosine distance
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - similarity_threshold,
            metric='cosine',
            linkage='average'
        )
        labels = clustering.fit_predict(type_embeddings)
        
        # For each cluster, pick the canonical name
        for cluster_id in set(labels):
            cluster_indices = [type_indices[i] for i, l in enumerate(labels) if l == cluster_id]
            cluster_names = [entity_names[i] for i in cluster_indices]
            
            # Use LLM to pick the best canonical name from cluster members
            canonical = await pick_canonical_entity(cluster_names, entity_type, llm_client)
            
            for name in cluster_names:
                entity_to_canonical[name] = canonical
    
    return entity_to_canonical


async def pick_canonical_entity(
    names: list[str],
    entity_type: str,
    llm_client
) -> str:
    """
    Given a cluster of entity name variants, pick the best canonical form.
    """
    if len(names) == 1:
        return names[0]
    
    prompt = f"""These are different ways the same {entity_type} entity is referred to 
in documents: {names}

Pick the most complete, formal, and unambiguous name as the canonical form.
Return ONLY the canonical name, nothing else."""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.0
    )
    
    return response.choices[0].message.content.strip()
```

---

## Step 3 — Building the Graph

With entities resolved, construct the knowledge graph.

```python
import networkx as nx

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()  # Directed multigraph: multiple edges between nodes
    
    def add_entity(self, canonical_name: str, entity_type: str, properties: dict):
        if not self.graph.has_node(canonical_name):
            self.graph.add_node(
                canonical_name,
                entity_type=entity_type,
                **properties
            )
        else:
            # Merge properties from multiple mentions
            existing = self.graph.nodes[canonical_name]
            # Merge source_doc_ids into a list
            existing_sources = existing.get("source_doc_ids", [])
            new_source = properties.get("source_doc_id")
            if new_source and new_source not in existing_sources:
                existing_sources.append(new_source)
            self.graph.nodes[canonical_name]["source_doc_ids"] = existing_sources
    
    def add_relationship(
        self,
        source: str,
        relation: str,
        target: str,
        properties: dict
    ):
        self.graph.add_edge(
            source,
            target,
            relation=relation,
            **properties
        )
    
    def get_entity_context(self, entity_name: str, hops: int = 1) -> dict:
        """
        Get an entity's neighborhood in the graph up to N hops.
        Returns all connected entities and relationships.
        """
        if not self.graph.has_node(entity_name):
            return {"entity": entity_name, "found": False}
        
        # Get all nodes within N hops
        subgraph_nodes = {entity_name}
        frontier = {entity_name}
        
        for _ in range(hops):
            new_frontier = set()
            for node in frontier:
                # Successors (outgoing edges)
                new_frontier.update(self.graph.successors(node))
                # Predecessors (incoming edges)
                new_frontier.update(self.graph.predecessors(node))
            subgraph_nodes.update(new_frontier)
            frontier = new_frontier
        
        subgraph = self.graph.subgraph(subgraph_nodes)
        
        # Format as readable context
        relationships = []
        for u, v, data in subgraph.edges(data=True):
            relationships.append({
                "source": u,
                "relation": data.get("relation", "related_to"),
                "target": v,
                "description": data.get("description", ""),
                "source_doc_id": data.get("source_doc_id")
            })
        
        return {
            "entity": entity_name,
            "entity_type": self.graph.nodes[entity_name].get("entity_type"),
            "connected_entities": list(subgraph_nodes - {entity_name}),
            "relationships": relationships,
            "found": True
        }
    
    def find_path(self, source: str, target: str) -> list[dict]:
        """
        Find the shortest relationship path between two entities.
        """
        try:
            path_nodes = nx.shortest_path(self.graph.to_undirected(), source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
        
        path_relationships = []
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            # Get edge data (take first edge if multiple)
            edge_data = dict(list(self.graph.get_edge_data(u, v, {}).values())[0]) if self.graph.has_edge(u, v) else {}
            path_relationships.append({
                "from": u,
                "relation": edge_data.get("relation", "related_to"),
                "to": v
            })
        
        return path_relationships
```

---

## Step 4 — Community Detection and Summary Generation

The full power of Graph RAG comes from community detection — identifying clusters of closely related entities and generating summaries of each community.

### Why Communities Matter

A query like "what are the key themes across our entire contract corpus?" cannot be answered by retrieving individual chunks. It requires a global view of the data. Community summaries provide this global view at a manageable granularity — each community summary describes a cluster of related entities and their relationships.

```python
def detect_communities(kg: KnowledgeGraph) -> dict:
    """
    Detect communities (clusters) in the knowledge graph.
    Uses the Leiden algorithm for high-quality community detection.
    """
    try:
        import leidenalg
        import igraph as ig
        
        # Convert networkx graph to igraph
        g = ig.Graph.from_networkx(kg.graph.to_undirected())
        
        # Run Leiden algorithm
        partition = leidenalg.find_partition(
            g,
            leidenalg.ModularityVertexPartition
        )
        
        communities = {}
        for community_id, community_members in enumerate(partition):
            node_names = [g.vs[i]["_nx_name"] for i in community_members]
            communities[community_id] = node_names
        
        return communities
    
    except ImportError:
        # Fallback to networkx's Louvain implementation
        import networkx.algorithms.community as nx_comm
        
        undirected = kg.graph.to_undirected()
        communities_gen = nx_comm.louvain_communities(undirected)
        
        return {
            i: list(community)
            for i, community in enumerate(communities_gen)
        }


async def generate_community_summary(
    community_id: int,
    community_nodes: list[str],
    kg: KnowledgeGraph,
    llm_client,
    max_relationships: int = 50
) -> dict:
    """
    Generate a natural language summary of a community in the knowledge graph.
    """
    
    # Collect all relationships within this community
    community_set = set(community_nodes)
    relationships = []
    
    for u, v, data in kg.graph.edges(data=True):
        if u in community_set and v in community_set:
            relationships.append(f"{u} {data.get('relation', 'relates to')} {v}: {data.get('description', '')}")
    
    # Get entity descriptions
    entity_descriptions = []
    for node in community_nodes[:20]:  # Limit for prompt length
        node_data = kg.graph.nodes[node]
        entity_descriptions.append(
            f"{node} ({node_data.get('entity_type', 'unknown')})"
        )
    
    # Build context for LLM
    relationships_text = "\n".join(relationships[:max_relationships])
    entities_text = ", ".join(entity_descriptions)
    
    prompt = f"""You are analyzing a cluster of related entities from a document corpus.

Entities in this cluster:
{entities_text}

Relationships within the cluster:
{relationships_text}

Generate a comprehensive summary of this cluster that:
1. Describes what the entities in this cluster have in common
2. Explains the key relationships between them
3. Identifies the main themes or topics this cluster represents
4. Notes any significant patterns or structures

Write 2-4 paragraphs. This summary will be used to answer high-level questions about the corpus."""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.1
    )
    
    summary = response.choices[0].message.content
    
    return {
        "community_id": community_id,
        "entities": community_nodes,
        "entity_count": len(community_nodes),
        "relationship_count": len(relationships),
        "summary": summary,
        "summary_embedding": None  # Will be computed later
    }
```

---

## Step 5 — Graph RAG Retrieval

With the knowledge graph and community summaries built, retrieval works differently depending on query type.

### Local Search (Entity-Centric Queries)

For queries about specific entities or relationships between named entities:

```python
async def graph_local_search(
    query: str,
    kg: KnowledgeGraph,
    community_summaries: list[dict],
    llm_client,
    embedding_model,
    k_communities: int = 3
) -> str:
    """
    Local search: find relevant entities and their neighborhood.
    Best for: "tell me about X", "how are X and Y related?", "what did X do?"
    """
    
    # Step 1: Extract entities from the query
    entity_extraction = await extract_query_entities(query, llm_client)
    query_entities = entity_extraction.get("entities", [])
    
    # Step 2: Find these entities in the graph
    context_parts = []
    
    for entity_name in query_entities:
        # Try exact match first
        if kg.graph.has_node(entity_name):
            entity_context = kg.get_entity_context(entity_name, hops=2)
            context_parts.append(format_entity_context(entity_context))
        else:
            # Try fuzzy match using embedding similarity
            similar_entities = await find_similar_entities(
                entity_name, kg, embedding_model
            )
            for entity in similar_entities[:2]:
                entity_context = kg.get_entity_context(entity, hops=1)
                context_parts.append(format_entity_context(entity_context))
    
    # Step 3: Find relevant community summaries
    query_embedding = await embedding_model.embed(query)
    
    # Score communities by relevance (embed summaries at index time)
    community_scores = []
    for comm in community_summaries:
        if comm.get("summary_embedding") is not None:
            score = cosine_similarity(query_embedding, comm["summary_embedding"])
            community_scores.append((score, comm))
    
    top_communities = sorted(community_scores, key=lambda x: x[0], reverse=True)[:k_communities]
    for _, comm in top_communities:
        context_parts.append(f"Community context:\n{comm['summary']}")
    
    # Step 4: Generate answer from graph context
    graph_context = "\n\n---\n\n".join(context_parts)
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "Answer questions using the provided knowledge graph context. Be specific about relationships and entities."
            },
            {
                "role": "user",
                "content": f"Knowledge graph context:\n{graph_context}\n\nQuestion: {query}"
            }
        ],
        max_tokens=800,
        temperature=0.1
    )
    
    return response.choices[0].message.content


async def extract_query_entities(query: str, llm_client) -> dict:
    """Extract named entities from a query for graph lookup."""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"Extract named entities from: '{query}'\nReturn JSON: {{\"entities\": [\"entity1\", \"entity2\"]}}"
        }],
        response_format={"type": "json_object"},
        max_tokens=100,
        temperature=0.0
    )
    import json
    return json.loads(response.choices[0].message.content)
```

### Global Search (Corpus-Wide Queries)

For queries that require understanding patterns across the entire corpus:

```python
async def graph_global_search(
    query: str,
    community_summaries: list[dict],
    llm_client,
    embedding_model,
    k_communities: int = 10
) -> str:
    """
    Global search using community summaries.
    Best for: "what are the main themes?", "summarize the corpus", 
              "what patterns exist across all documents?"
    """
    
    # Score all community summaries against the query
    query_embedding = await embedding_model.embed(query)
    
    scored_communities = []
    for comm in community_summaries:
        if comm.get("summary_embedding") is not None:
            score = cosine_similarity(query_embedding, comm["summary_embedding"])
            scored_communities.append((score, comm))
    
    # Take top K communities
    top_k = sorted(scored_communities, key=lambda x: x[0], reverse=True)[:k_communities]
    
    # Map step: ask each community to answer the query from its perspective
    async def map_community(score: float, comm: dict) -> str:
        response = await llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Based on this community summary:
{comm['summary']}

What is relevant to answering: {query}

Provide a brief answer (2-3 sentences) from this community's perspective.
If not relevant, respond: NOT_RELEVANT"""
            }],
            max_tokens=200,
            temperature=0.1
        )
        result = response.choices[0].message.content
        return result if "NOT_RELEVANT" not in result else ""
    
    map_tasks = [map_community(score, comm) for score, comm in top_k]
    map_results = await asyncio.gather(*map_tasks)
    
    # Reduce step: synthesize community answers into final response
    relevant_results = [r for r in map_results if r.strip()]
    
    combined = "\n\n---\n\n".join(relevant_results)
    
    reduce_response = await llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"""Synthesize these perspectives to answer: {query}

Perspectives from different document clusters:
{combined}

Provide a comprehensive, well-organized answer that identifies the main themes 
and patterns across the entire corpus."""
        }],
        max_tokens=1000,
        temperature=0.1
    )
    
    return reduce_response.choices[0].message.content
```

---

## Hybrid Graph + Vector RAG

Graph RAG and vector RAG are complementary. Build a system that routes to the appropriate retrieval method based on query type.

```python
async def hybrid_graph_vector_rag(
    query: str,
    kg: KnowledgeGraph,
    community_summaries: list[dict],
    vector_retriever,
    llm_client,
    embedding_model
) -> dict:
    """
    Route query to graph search, vector search, or both based on query type.
    """
    
    # Classify query type
    query_type = await classify_query_for_graph_rag(query, llm_client)
    
    results = {}
    
    if query_type in ["relational", "global_theme"]:
        if query_type == "relational":
            graph_answer = await graph_local_search(
                query, kg, community_summaries, llm_client, embedding_model
            )
        else:
            graph_answer = await graph_global_search(
                query, community_summaries, llm_client, embedding_model
            )
        results["graph_answer"] = graph_answer
    
    if query_type in ["factual", "procedural", "relational"]:
        # Also run vector retrieval for specific facts
        vector_chunks = await vector_retriever.retrieve(query)
        results["vector_chunks"] = vector_chunks
    
    # Combine if both were used
    if "graph_answer" in results and "vector_chunks" in results:
        return await combine_graph_and_vector(
            query, results["graph_answer"], results["vector_chunks"], llm_client
        )
    
    if "graph_answer" in results:
        return {"answer": results["graph_answer"], "source": "graph"}
    
    # Pure vector fallback
    context = format_context(results["vector_chunks"])
    answer = await generate_from_context(query, context, llm_client)
    return {"answer": answer, "source": "vector"}


async def classify_query_for_graph_rag(query: str, llm_client) -> str:
    prompt = f"""Classify this query:
"{query}"

Types:
- relational: asks about connections between specific named entities
- global_theme: asks about patterns, themes, or summaries across many documents
- factual: asks about a specific fact in a document
- procedural: asks how to do something

Return only the type name."""
    
    response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20,
        temperature=0.0
    )
    return response.choices[0].message.content.strip().lower()
```

---

## Graph RAG Cost and When to Use It

Graph RAG is significantly more expensive to build and maintain than vector RAG.

**Build cost:**
- Entity extraction: ~1 LLM call per chunk × corpus size.
- Community summary generation: ~1 LLM call per community × number of communities.
- Entity resolution: embedding + clustering across all entities.

For a 10,000-chunk corpus at $0.0005 per gpt-4o-mini call: ~$5 for extraction. For 200 communities: ~$2 for summaries. Total: ~$7 one-time build cost. For a 1M-chunk corpus, $700 just for extraction.

**Maintenance cost:** Every new document must have entities extracted and integrated into the graph. Community detection must re-run periodically as the graph grows. Community summaries must be regenerated when their member entities change significantly.

**When Graph RAG is worth it:**

Use Graph RAG when your use case is fundamentally relational or global:
- Legal and compliance: "Show all cases where Company X is involved as both a plaintiff and a defendant."
- Knowledge management: "What are the key themes across all our research reports from the last 3 years?"
- Due diligence: "Map all relationships between the target company's officers and other entities we have encountered."
- Biomedical: "What proteins interact with both Drug A and Drug B pathways?"

**Do NOT use Graph RAG when:**
- Queries are primarily factual lookups — vector search is faster, cheaper, and sufficient.
- The corpus changes frequently — maintaining the graph is expensive.
- You do not have clear entity-relationship questions — the complexity is not justified.

---

## Summary

- Vector search finds similar documents. Graph RAG retrieves relational information — connections between entities across documents — that vector search fundamentally cannot handle.
- Knowledge graph construction requires: entity extraction per chunk (LLM), entity resolution (embedding + clustering), relationship merging, and graph construction.
- Community detection clusters related entities. Community summaries provide a global view of the corpus, enabling thematic and cross-corpus questions.
- Local search: entity neighborhood traversal for specific entity-centric queries. Global search: community summary map-reduce for corpus-wide thematic queries.
- Hybrid graph + vector RAG routes queries to the right retrieval mechanism based on query type. Most production systems benefit from having both.
- Graph RAG has high build and maintenance costs. Justify it only when your use case is fundamentally relational or requires global corpus understanding.

---

## What's Next

Lesson 5.5 covers multi-hop and multi-document reasoning — how to handle queries that require chaining information across multiple documents where no single document contains the complete answer.