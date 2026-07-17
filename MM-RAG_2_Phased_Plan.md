# MM-RAG 2.0 — Phased Implementation Plan

## Overview

MM-RAG 2.0 evolves the existing MM-RAG pipeline into a production-quality, multimodal
retrieval-augmented generation system capable of indexing documents, audio, and video while
remaining fully edge-compatible on Apple Silicon. The architecture is inspired by three
open-source projects from HKUDS — **LightRAG**, **RAG-Anything**, and **VideoRAG** — and
maps their ideas onto an MLX-first runtime with a dual vector-backend design: embedded
**sqlite-vec** for edge/offline use and **Qdrant** for server-scale deployments.

---

## Dependency Versions (pinned at plan date: June 2026)

| Dependency | Version | Role |
|---|---|---|
| GRDB.swift | 7.10.0 | SQLite ORM for Swift (Android/Linux/Windows/SQLCipher support added) |
| sqlite-vec | 0.1.6 (PyPI) / 0.1.7 (GitHub releases) | Embedded vector search SQLite extension |
| Qdrant server | 1.18.1 | Optional server vector backend |
| qdrant-client (Python) | 1.18.0 | Python SDK for Qdrant |
| mlx-swift | 0.21.x (May 2026) | Swift array framework for Apple Silicon LLM/VLM inference |
| MLX-VLM | latest main | Vision-language model inference on MLX |
| mlx-whisper / lightning-whisper-mlx | latest | Apple Silicon Whisper transcription |
| LanceDB | 0.33.0 | Optional alternative embedded multimodal vector store |
| LlamaIndex | latest stable | Optional Python orchestration layer |
| ffmpeg | 7.x | Video/audio preprocessing |
| PyMuPDF / Docling | latest | Document parsing |
| Python | 3.13.x | Runtime for ingestion/pipeline services |

---

## Open-Source Inspiration Mapping

Understanding what is borrowed from each project clarifies the architectural decisions.

### LightRAG → Dual Graph + Vector Retrieval

LightRAG introduces a **knowledge graph + vector index hybrid**, where entities and
relationships extracted from documents are stored as a graph alongside dense embeddings.
At query time, it supports three modes: `local` (vector-only), `global` (graph-first),
and `hybrid` (vector search + graph expansion). This "mix mode" retrieval substantially
improves multi-hop and relational queries compared to naive chunk retrieval.

**MM-RAG 2.0 mapping:**
- Adopt the three query modes as first-class API options.
- Implement the graph as three lightweight SQLite tables (`nodes`, `edges`,
  `node_embeddings`) rather than LightRAG's assumed Postgres/OpenSearch/Milvus backends.
- Use smaller local MLX models (7–9B) for entity extraction instead of LightRAG's
  recommended 32B+ LLMs with 32–64K context.
- Graph construction runs offline/async, not inline with ingestion.

### RAG-Anything → Multimodal Document Pipeline + `content_list` Abstraction

RAG-Anything generalizes RAG to arbitrary file types by normalizing all parsed content
into a typed **`content_list`**: an ordered sequence of records where each item carries
a `type` (`text`, `image`, `table`, `equation`, `generic`), source metadata, and
text/caption payload. Modal-specific processors enrich each type before indexing.
Vector-graph fusion retrieval and modality-aware ranking then operate over the unified
index.

**MM-RAG 2.0 mapping:**
- Adopt `content_list` as the canonical internal representation for all source types
  (documents, audio, video). Every ingestion path outputs to this format before the
  common indexer.
- Implement pluggable modal processors: caption images, summarize tables, explain
  equations, transcribe audio — all optional and configurable for edge resource
  constraints.
- Share a single indexing and retrieval layer across all modalities.

### VideoRAG → Graph-Driven Video Indexing + Hierarchical Context

VideoRAG addresses long-video RAG (tested on 134+ hours on a single RTX 3090) by
building a **hierarchical segment graph**: scenes → sub-scenes → frames, connected via
`NEXT_SCENE`, `PART_OF`, and topical edges. Cross-video links (`SIMILAR_SCENE`) enable
retrieval that spans multiple video files. Dual-channel retrieval combines transcript
text and visual/frame content.

**MM-RAG 2.0 mapping:**
- Implement hierarchical video nodes (scene → sub-scene → frame) as graph edges in the
  same SQLite graph layer.
- Add `SIMILAR_SCENE` cross-video edges computed from embedding similarity at index time.
- Dual-channel video retrieval: transcript segment search + frame/caption search, unified
  at the graph expansion step.
- Scale down to laptop hardware: no 100+ hour benchmarks required; the architecture
  generalizes.

---

## Vector Backend Design

The dual-backend design is the key edge-compatibility decision. A `VectorBackend` protocol
abstracts all vector operations so the rest of the pipeline is backend-agnostic.

```
┌────────────────────────────────────────────┐
│           VectorBackend Protocol           │
│  upsert(id, vector, payload)               │
│  search(query_vector, k, filter)           │
│  delete(id)                                │
│  batch_upsert([...])                       │
└────────────┬───────────────────────────────┘
             │
     ┌───────┴────────┐
     ▼                ▼
SqliteVecBackend  QdrantBackend
(default, edge)   (server)
```

### SqliteVecBackend (default)
- sqlite-vec 0.1.6+ embedded in-process with the graph SQLite file.
- No daemon, no network, runs on M5 MacBook Pro and Raspberry Pi 5 equally.
- Stores all modality embeddings in `vec_items` virtual table alongside `nodes`/`edges`.
- Suitable for corpora up to ~50–100 hours of video or thousands of documents.

### QdrantBackend (optional, server mode)
- Qdrant 1.18.1 server deployed via Docker Compose on a self-hosted server.
- Python client: `qdrant-client==1.18.0`.
- Supports named sparse + dense vector collections for hybrid dense/sparse retrieval.
- Enables multi-device access (laptop + edge nodes over a private network).
- Switched via a single config key: `vector_backend: "qdrant"` + `qdrant_url`.
- `qdrant-client` also supports an in-memory / local-file mode
  (`QdrantClient(path="path/to/db")`) for development without a running server.

### Switching backends
```python
# config.yaml
vector_backend: "sqlite_vec"   # or "qdrant"
qdrant_url: "http://your-server:6333"
qdrant_api_key: null  # local deployment
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        MM-RAG 2.0 Runtime                       │
│                                                                 │
│  ┌────────────┐  ┌────────────────┐  ┌──────────────────────┐  │
│  │  Ingestion │  │   Enrichment   │  │     Storage Layer    │  │
│  │  Pipeline  │→ │  & Graph Build │→ │  SQLite + VectorBack │  │
│  └────────────┘  └────────────────┘  └──────────────────────┘  │
│         ↑                                        ↓              │
│  ┌──────┴──────┐                    ┌────────────────────────┐  │
│  │   Sources   │                    │   Retrieval & QA       │  │
│  │ Video/Audio │                    │  local/global/hybrid   │  │
│  │ Docs/Images │                    │  + MLX LLM/VLM         │  │
│  └─────────────┘                    └────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

All content flows through a `content_list` abstraction before indexing. The graph layer
sits alongside the vector store in SQLite (edge mode) or as payload metadata in Qdrant
(server mode).

---

## Phase 0 — Pipeline Refactor and `content_list` Foundation

**Goal:** Decompose the existing MM-RAG monolithic ingestion into explicit, testable
pipeline stages and introduce the `content_list` abstraction as the canonical internal
format.

### Tasks

1. **Audit current MM-RAG stages** — document each step: download → ffmpeg → scene
   detection → STT → frame sampling → OCR → embedding → SQLite store.
2. **Define `ContentItem` dataclass:**
   ```python
   @dataclass
   class ContentItem:
       type: Literal["text", "image", "table", "equation",
                     "video_segment", "audio_segment", "generic"]
       source_id: str          # video/audio/doc identifier
       chunk_idx: int
       page_idx: Optional[int]
       scene_idx: Optional[int]
       start_ts: Optional[float]   # seconds
       end_ts: Optional[float]
       text: Optional[str]
       caption: Optional[str]
       file_path: Optional[str]    # for image/frame items
       metadata: dict
   ```
3. **Wrap each existing step** as a `PipelineStage` with `process(source) → List[ContentItem]`.
4. **Define `VectorBackend` protocol** with `SqliteVecBackend` as default implementation
   using sqlite-vec 0.1.6.
5. **Write unit tests** for each stage in isolation.

### Changes to existing MM-RAG
- No user-facing changes; this is a pure internal refactor.
- All output still lands in the same SQLite schema — extend it minimally.
- Existing retrieval still works through `SqliteVecBackend`.

### Key files to add/modify
- `mm_rag/pipeline/content_item.py` (new)
- `mm_rag/pipeline/stages/*.py` (new — wrap existing logic)
- `mm_rag/storage/vector_backend.py` (new protocol)
- `mm_rag/storage/sqlite_vec_backend.py` (new implementation)

---

## Phase 1 — Multimodal Document Ingestion (RAG-Anything-inspired)

**Goal:** Add full document ingestion (PDF, Office, Markdown, HTML) alongside existing
video/audio, outputting unified `content_list` items.

### Inspiration from RAG-Anything
RAG-Anything uses MinerU or Docling to parse documents into typed content blocks, then
runs modal-specific processors (VLM captioning for images, LLM summarization for tables,
OCR for equations) before indexing. MM-RAG 2.0 adapts this with local MLX models.

### Tasks

1. **Add `DocumentIngestor`** using PyMuPDF (primary) and Docling (for complex layouts):
   - Extract text blocks → `ContentItem(type="text", ...)`
   - Extract embedded images → `ContentItem(type="image", file_path=..., ...)`
   - Extract tables → `ContentItem(type="table", text=markdown_table, ...)`
   - Extract equations → `ContentItem(type="equation", text=latex, ...)`
2. **Add modal processors** (all optional, configured per profile):
   - `ImageCaptionProcessor`: calls MLX-VLM to generate a caption for each extracted image.
   - `TableSummaryProcessor`: calls local LLM to produce a plain-language table summary.
   - `EquationProcessor`: stores LaTeX as-is; optionally generates a description.
3. **Add `AudioIngestor`** that wraps existing Whisper/MLX transcription and emits
   `audio_segment` items with timestamps and speaker metadata.
4. **Update the common indexer** to accept `content_list` from any ingestor and upsert
   into `VectorBackend`.
5. **Add resource profiles** in config:
   ```yaml
   profiles:
     edge_m5:
       image_captioning: true
       table_summary: false
       equation_processing: false
       vlm_model: "mlx-community/Qwen2-VL-7B-4bit"
     edge_rpi5:
       image_captioning: false
       table_summary: false
       equation_processing: false
   ```

### Changes to existing MM-RAG
- New ingestor classes alongside existing video pipeline.
- Config schema extended with `profiles`.
- MCP API gains a `ingest_document(path, profile)` endpoint.

### Swift considerations (if building native macOS client)
- Use GRDB.swift 7.10.0 to read/write the SQLite DB from Swift.
- GRDB 7.10.0 supports Swift 6 concurrency, Android/Linux/Windows, and SQLCipher via SPM.
- Modal processor calls route through the MLX-swift 0.21.x runtime.

---

## Phase 2 — Graph Layer (LightRAG-inspired)

**Goal:** Add a lightweight knowledge graph alongside the vector store, enabling
entity-aware and relational retrieval without external graph databases.

### Inspiration from LightRAG
LightRAG extracts entities and relationships from text using an LLM, builds a graph
(originally backed by Postgres/OpenSearch/Milvus), and queries it in `local`, `global`,
or `hybrid` mode. MM-RAG 2.0 re-implements this on SQLite with smaller local models.

### SQLite Schema Additions

```sql
-- Graph nodes: entities, scenes, documents, topics
CREATE TABLE nodes (
    id          TEXT PRIMARY KEY,
    type        TEXT NOT NULL,   -- 'entity' | 'scene' | 'document' | 'topic'
    label       TEXT NOT NULL,
    payload     TEXT,            -- JSON
    source_id   TEXT,
    created_at  REAL DEFAULT (unixepoch('subsec'))
);

-- Graph edges
CREATE TABLE edges (
    id           TEXT PRIMARY KEY,
    src_id       TEXT NOT NULL REFERENCES nodes(id),
    dst_id       TEXT NOT NULL REFERENCES nodes(id),
    relation     TEXT NOT NULL,  -- 'MENTIONS' | 'NEXT_SCENE' | 'PART_OF' |
                                 --  'SAME_TOPIC' | 'SIMILAR_SCENE' | 'FOLLOWS'
    weight       REAL DEFAULT 1.0,
    metadata     TEXT            -- JSON
);
CREATE INDEX idx_edges_src ON edges(src_id);
CREATE INDEX idx_edges_dst ON edges(dst_id);

-- Optional: node-level embeddings for entity search
CREATE VIRTUAL TABLE node_vec USING vec0(
    embedding float[768]
);
```

### Graph Builder

1. **Entity extraction** per `ContentItem(type="text")`:
   - First pass: lightweight NER (spaCy or a small local model) to identify entities fast.
   - Optional second pass: local LLM call for relation extraction on high-value chunks.
   - Create `nodes` rows for entities; create `MENTIONS` edges to source document/scene nodes.
2. **Scene graph** for video (VideoRAG-inspired):
   - Each `video_segment` ContentItem → one `scene` node.
   - Sequential scenes → `NEXT_SCENE` edges.
   - Scenes with embedding cosine similarity > threshold → `SAME_TOPIC` edges.
3. **Cross-video links** (VideoRAG `SIMILAR_SCENE`):
   - At index-time, compare new scene embeddings to all existing scene node embeddings.
   - Create `SIMILAR_SCENE` edges above a configurable threshold.
4. **Document nodes**: each ingested document/video → one `document` node; all its
   `ContentItem` chunks → `PART_OF` edges to that document node.

### Graph Builder runs async
- Triggered after `VectorBackend.upsert()` completes.
- Uses a task queue (asyncio-based) so ingestion UX is not blocked.
- Configurable: can be disabled entirely on Raspberry Pi 5 profile.

### Changes to existing MM-RAG
- Three new tables in the SQLite schema (migration via GRDB's `DatabaseMigrator` in Swift,
  or a Python migration script).
- `GraphBuilder` class added to `mm_rag/graph/`.
- New config keys: `graph.enabled`, `graph.entity_extraction_model`,
  `graph.similarity_threshold`.

---

## Phase 3 — Hybrid Retrieval Modes (LightRAG + VideoRAG-inspired)

**Goal:** Implement `local`, `global`, and `hybrid` query modes and dual-channel video
retrieval, giving agents and users control over retrieval depth vs. speed.

### Query Modes

#### `local` (current MM-RAG behavior)
- Pure vector search over `content_list` chunks.
- Fastest; best for simple factual lookups.

#### `global` (LightRAG-inspired)
- Identify entities in the query via NER/embedding match against `node_vec`.
- Expand graph neighborhood: find all nodes reachable within N hops.
- Retrieve `ContentItem` chunks associated with those nodes.
- Best for "what does this person say about X across all videos?" style queries.

#### `hybrid` (LightRAG mix mode)
- Run `local` vector search → top-K chunks.
- Expand graph neighborhood of those K chunks' associated nodes.
- Re-rank combined candidate set by relevance score.
- Default recommended mode; balances precision and graph-aware coverage.

### Video-aware retrieval (VideoRAG dual-channel)
- Transcript channel: vector search over `audio_segment` / `video_segment` text.
- Visual channel: vector search over frame captions and OCR text.
- Merge ranked results, deduplicate by scene, return time-bounded segments.
- For cross-video queries: traverse `SIMILAR_SCENE` edges to pull related segments from
  other indexed videos.

### Modality-aware ranking
Inspired by RAG-Anything's modality-aware re-ranking:
- Boost `image` and `table` items when the query contains visual keywords
  ("diagram", "chart", "show me", "what does X look like").
- Boost `video_segment` items when the query contains temporal keywords
  ("when", "at what point", "timestamp").
- Boost `equation` items for mathematical/formula queries.

### MCP API changes
```python
# New query parameters
class QueryRequest:
    query: str
    mode: Literal["local", "global", "hybrid"] = "hybrid"
    modalities: Optional[List[str]] = None   # filter by type
    source_filter: Optional[List[str]] = None  # filter by source_id
    top_k: int = 10
    cross_video: bool = True
```

---

## Phase 4 — Qdrant Backend + Dual-Backend Switching

**Goal:** Implement `QdrantBackend` as a drop-in replacement for `SqliteVecBackend`,
enabling server-scale deployments with multi-device access.

### When to use Qdrant
- Corpora larger than ~100 hours of video or tens of thousands of documents.
- Multi-device access (laptop + edge devices over a private network).
- Sparse + dense hybrid search (Qdrant's `sparse_vectors` support).
- Need for advanced filtering on rich payload metadata at scale.

### Qdrant deployment
```yaml
# docker-compose.yml (server)
services:
  qdrant:
    image: qdrant/qdrant:v1.18.1
    ports:
      - "6333:6333"
      - "6334:6334"   # gRPC
    volumes:
      - ./qdrant_storage:/qdrant/storage
    environment:
      QDRANT__SERVICE__GRPC_PORT: "6334"
```

### QdrantBackend implementation
```python
# qdrant-client==1.18.0
from qdrant_client import AsyncQdrantClient, models

class QdrantBackend(VectorBackend):
    """
    Uses two collections per deployment:
    - mm_rag_content: dense vectors for ContentItem chunks
    - mm_rag_nodes: dense vectors for graph entity nodes
    Graph edges are stored as payload metadata on points.
    """
    def __init__(self, url: str, api_key: Optional[str] = None):
        self.client = AsyncQdrantClient(url=url, api_key=api_key)

    async def upsert(self, id: str, vector: list[float], payload: dict):
        await self.client.upsert(
            collection_name="mm_rag_content",
            points=[models.PointStruct(id=id, vector=vector, payload=payload)]
        )

    async def search(self, query_vector, k: int, filter: Optional[dict] = None):
        return await self.client.query_points(
            collection_name="mm_rag_content",
            query=query_vector,
            limit=k,
            query_filter=filter
        )
```

### Local development mode (no server required)
```python
# qdrant-client supports in-process local mode
client = QdrantClient(path="./qdrant_local_db")  # persists to disk, no server needed
```
This lets developers use Qdrant API locally without Docker during development, then switch
to the server by changing the URL.

### Config switching
```yaml
# edge mode (default)
vector_backend: "sqlite_vec"

# server mode
vector_backend: "qdrant"
qdrant_url: "http://your-server:6333"
qdrant_api_key: null

# local dev mode with Qdrant API (no server)
vector_backend: "qdrant_local"
qdrant_local_path: "./qdrant_dev_db"
```

### Graph layer in Qdrant mode
- `nodes` and `edges` tables stay in SQLite (they are small, structured, and benefit from
  relational queries).
- `node_embeddings` are stored in a `mm_rag_nodes` Qdrant collection instead of
  `node_vec` virtual table.
- This hybrid SQLite-graph + Qdrant-vectors design avoids duplicating graph traversal
  logic.

---

## Phase 5 — MLX Runtime Integration

**Goal:** Ensure all LLM, VLM, and embedding calls route through a local MLX runtime by
default, with OpenRouter/Ollama as configurable fallbacks.

### LLMBackend protocol
```python
class LLMBackend(Protocol):
    async def generate(self, prompt: str, system: str) -> str: ...
    async def embed_text(self, texts: List[str]) -> List[List[float]]: ...
    async def caption_image(self, image_path: str, prompt: str) -> str: ...
    async def describe_video_segment(self, frames: List[str], transcript: str) -> str: ...
```

### Implementations
| Backend | Model | Use case |
|---|---|---|
| `MLXTextBackend` | Qwen2.5-7B-4bit or Gemma-3-9B-4bit via mlx-swift | Text generation, entity extraction, query rewriting |
| `MLXVLMBackend` | Qwen2-VL-7B-4bit via MLX-VLM | Image captioning, video frame description |
| `MLXAudioBackend` | Whisper large-v3 via mlx-whisper | Audio/video transcription |
| `MLXEmbedBackend` | mlx-embeddings (e.g. nomic-embed-text) | Text + image embeddings |
| `OllamaBackend` | Any Ollama model | Fallback if MLX model unavailable |
| `OpenRouterBackend` | Any cloud model | Optional fallback for heavy tasks |

### M5 MacBook Pro resource profile
```yaml
profiles:
  edge_m5:
    llm_backend: "mlx_text"
    vlm_backend: "mlx_vlm"
    audio_backend: "mlx_audio"
    embed_backend: "mlx_embed"
    text_model: "mlx-community/Qwen2.5-7B-Instruct-4bit"
    vlm_model: "mlx-community/Qwen2-VL-7B-Instruct-4bit"
    embed_model: "mlx-community/nomic-embed-text-v1.5"
    whisper_model: "large-v3"
    max_concurrent_inferences: 2
    graph_build_async: true
```

### Batching and streaming
- All LLM/VLM calls use async queuing to avoid starving the MCP API.
- Batch STT processing: segment audio before calling Whisper to keep per-call latency low.
- Stream LLM responses where the client supports it.

---

## Phase 6 — Evaluation and Tuning

**Goal:** Establish a local evaluation harness to measure retrieval quality and latency
across all phases, and tune thresholds/models based on results.

### Evaluation approach
- Inspired by LightRAG's use of RAGAS metrics (context precision, faithfulness,
  answer relevance) and VideoRAG's LongerVideos-style evaluation.
- Build a small local evaluation set:
  - 5–10 long-form lecture recordings (audio/video).
  - 1–2 documentary-style long videos (testing cross-video and scene graph retrieval).
  - 20–50 PDF/Office documents (testing multimodal document pipeline).
  - 50–100 ground-truth QA pairs with source citations.

### Metrics to track
| Metric | Tool | Target |
|---|---|---|
| Context precision | RAGAS | ≥0.80 |
| Answer faithfulness | RAGAS | ≥0.85 |
| Retrieval latency (local mode) | Python timeit | <200ms p95 |
| Retrieval latency (hybrid mode) | Python timeit | <500ms p95 |
| Ingestion throughput | custom | ≥1 hr video/min on M5 |

### Tuning levers
- `graph.similarity_threshold` — controls density of `SIMILAR_SCENE` edges.
- `retrieval.top_k` — chunk count before graph expansion.
- `retrieval.graph_hops` — neighborhood expansion depth in global/hybrid modes.
- Modal processor on/off per profile.
- VLM captioning frequency (every frame vs. keyframes only).

---

## Migration Guide for Existing MM-RAG Users

| Existing behavior | MM-RAG 2.0 equivalent | Migration action |
|---|---|---|
| Video-only ingestion | Still fully supported via `VideoIngestor` | None required |
| sqlite-vec vector store | `SqliteVecBackend` (default) | Transparent; existing DB compatible |
| Single retrieval mode | Now `mode="local"` | Default query mode is `"hybrid"` — set `mode="local"` to preserve old behavior |
| OpenRouter/Ollama LLM calls | `OllamaBackend` / `OpenRouterBackend` | Set `llm_backend: "ollama"` in config |
| No graph layer | Graph is opt-in | `graph.enabled: false` preserves old behavior |

---

## Summary: Phase-by-Phase Inspiration Map

| Phase | Primary Inspiration | What is borrowed |
|---|---|---|
| 0 — Refactor | Internal | Pipeline modularity, `content_list` abstraction |
| 1 — Documents | RAG-Anything | `content_list` typed records, modal processors, multimodal indexing |
| 2 — Graph | LightRAG | Dual vector+graph index, entity/relation extraction, async graph build |
| 2 — Video Graph | VideoRAG | Hierarchical scene nodes, `SIMILAR_SCENE` cross-video edges |
| 3 — Retrieval | LightRAG + VideoRAG | `local/global/hybrid` modes, dual-channel video retrieval, modality-aware ranking |
| 4 — Qdrant | Internal | Dual-backend abstraction, server scaling |
| 5 — MLX Runtime | Internal | Edge-first inference, M5 resource profiles |
| 6 — Evaluation | LightRAG + VideoRAG | RAGAS metrics, long-video evaluation sets |

