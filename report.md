# Track A: Novel-Backstory Consistency Classifier - Technical Report

## System Overview

Hypothesis-based causal consistency system using the Pathway Python framework to evaluate logical coherence between backstories and novels (100k+ words).

## Ingestion Design

### Native Pathway IO Approach

The ingestion module leverages **native Pathway file system connectors** (`pw.io.fs.read`) for direct streaming data ingestion. This design eliminates intermediate file formats and enables declarative data processing.

**Key Design Decisions:**

1. **Glob Pattern Matching**: Uses file patterns (`*/novel.txt`, `*/backstory.txt`) to automatically discover and load all stories from directory structure
2. **Binary Format Reading**: Reads files as raw bytes with metadata, providing full control over encoding and error handling
3. **Static Mode**: Processes complete dataset in batch mode suitable for hackathon evaluation

### Schema Design

Both novels and backstories use enriched schemas with metadata fields:

**LongContextNovelSchema / LongContextBackstorySchema:**
- `story_id`: Unique story identifier extracted from directory name
- `doc_id`: Composite unique document ID (e.g., `story_1_novel`, `story_1_backstory`)
- `document_type`: Document classification (`"novel"` or `"backstory"`)
- `text`: Full document content (supports 100k+ words)
- `word_count`: Total word count for analytics
- `char_count`: Character count for chunking calculations
- `file_path`: Original file path for traceability

### Pathway Transform Pipeline

All processing logic is implemented as **Pathway transformations** using `pw.select()` and `pw.apply()`:

```python
novels_raw = pw.io.fs.read(novel_pattern, format="binary", mode="static", with_metadata=True)

novels_table = novels_raw.select(
    text=pw.apply(lambda x: x.decode('utf-8', errors='ignore'), novels_raw.data),
    file_path=pw.apply(lambda m: m["path"], novels_raw._metadata),
).select(
    pw.this.text,
    pw.this.file_path,
    story_id=pw.apply(lambda p: os.path.basename(os.path.dirname(p)), pw.this.file_path),
    doc_id=pw.apply(lambda p: f"{os.path.basename(os.path.dirname(p))}_novel", pw.this.file_path),
    document_type=pw.apply(lambda _: "novel", pw.this.file_path),
    word_count=pw.apply(lambda t: len(t.split()), pw.this.text),
    char_count=pw.apply(lambda t: len(t), pw.this.text),
)
```

**Transform Stages:**
1. Decode binary to UTF-8 text
2. Extract file path from metadata
3. Derive story_id from directory structure
4. Generate unique doc_id
5. Set document_type label
6. Compute word/character counts

### Benefits

- **No intermediate files**: Direct file → Pathway table flow
- **Lazy evaluation**: Computations deferred until needed
- **Type safety**: Schemas enforce data contracts
- **Metadata enrichment**: All metadata computed within Pathway DAG
- **Scalability**: Handles 100k+ word novels efficiently
- **Traceability**: Full lineage from file path to processed record

## Chunking Strategy

### Design Rationale

Long novels (100k+ words) exceed LLM context windows. The chunking module splits novels into manageable segments while preserving context through overlap.

### Token-Based Chunking

**Why Tokens Over Words/Characters:**
- LLM costs and limits measured in tokens, not words
- Token boundaries respect semantic units better than arbitrary character splits
- Using `tiktoken` with `cl100k_base` encoding ensures accurate token counting

**Parameters:**
- **Chunk Size**: 1000 tokens (configurable)
  - Balances context richness with retrieval granularity
  - ~750 words average, sufficient for coherent passages
- **Overlap**: 200 tokens (configurable)
  - 20% overlap prevents loss of context at boundaries
  - Ensures entities/events spanning chunks aren't split

### Chunk Metadata

Each chunk includes detailed provenance metadata:

| Field | Description | Example |
|-------|-------------|---------|
| `chunk_id` | Unique identifier | `story_1_chunk_42` |
| `chunk_index` | Sequential position (0-based) | `42` |
| `story_id` | Parent story identifier | `story_1` |
| `text` | Chunk content | "The detective entered..." |
| `start_char` | **Absolute** starting position in original text | `45230` |
| `end_char` | **Absolute** ending position in original text | `46890` |
| `token_count` | Actual tokens in chunk | `987` |
| `overlap_size` | Number of overlapping tokens with previous chunk | `200` |
| `temporal_position` | Normalized position in document (0.0-1.0) | `0.42` |
| `has_temporal_markers` | Contains temporal/causal keywords | `true` |

### Temporal Heuristic

**Simple Causal/Temporal Feature**: Chunk metadata includes `has_temporal_markers`, a boolean flag indicating presence of temporal or causal keywords.

**Marker Set** (regex word boundary matching):
- Temporal: `before`, `after`, `then`, `when`, `while`, `during`, `until`, `since`, `later`, `earlier`
- Sequential: `first`, `next`, `finally`, `initially`, `subsequently`, `eventually`, `previously`

**Rationale**:
- Backstory consistency requires temporal coherence
- Chunks with temporal markers are often critical for causal reasoning
- Retrieval can prioritize temporally-marked chunks
- **No deep NLP** — simple keyword matching only

### Benefits

1. **Debuggability**: `start_char`/`end_char` enable tracing chunks back to source
2. **Deduplication**: `chunk_id` prevents duplicate processing
3. **Context Preservation**: Overlap maintains narrative continuity
4. **Accuracy**: Token-based splitting aligns with LLM processing
5. **Flexibility**: Parameterized sizes support experimentation

### Implementation

```python
chunker = TextChunker(chunk_size=1000, overlap=200)
chunks = chunker.chunk_text(novel_text, story_id="story_1")
# Returns list of chunks with full metadata
```

Validation ensures `overlap < chunk_size` and positive values.

## Pipeline Stages

1. **Ingestion**: `ingest_long_context_stories()` - Load with Pathway IO
2. **Chunking**: Split novels into 1000-token chunks with 200-token overlap
3. **Indexing**: Embed chunks using sentence-transformers
4. **Retrieval**: Semantic search for top-15 relevant chunks per backstory
5. **Classification**: LLM scores hypotheses, Python selects highest-scoring label (0/1 output)
6. **Output**: Generate `results.csv` with `story_id,prediction` format

## Key Technologies

- **Pathway**: Declarative data processing framework
- **OpenAI GPT-4o-mini**: Hypothesis scoring (not authoritative analysis)
- **Sentence Transformers**: Semantic embeddings
- **Tiktoken**: Token-accurate chunking

## Evidence-Driven Reasoning

The system retrieves relevant novel chunks for each backstory, providing focused evidence for hypothesis scoring. This approach handles long contexts efficiently while staying within token limits.

## Pathway Framework Role

### Why Pathway?

Pathway is a **declarative data processing framework** designed for streaming and batch data pipelines. We leverage Pathway for:

1. **Document Ingestion**: Native file system connectors with glob pattern matching
2. **Data Joins**: Automatic pairing of novels and backstories by `story_id`
3. **Schema Enforcement**: Type-safe data contracts with validation
4. **Transform Pipeline**: Declarative metadata enrichment (word counts, file paths, etc.)
5. **Iteration Control**: Pathway manages document iteration while keeping heavy compute (embeddings, LLM) downstream

### Pathway Data Flow

```
File System (data/*/novel.txt, data/*/backstory.txt)
    ↓
pw.io.fs.read() — Native IO connector
    ↓
Pathway Tables (novels, backstories)
    ↓
pw.join() — Automatic story_id matching
    ↓
Joined Table (story pairs with metadata)
    ↓
Extract to Python — Pathway hands off documents
    ↓
Downstream Processing (chunking, embedding, retrieval, classification)
    ↓
Structured Output (results.csv + detailed metadata)
```

### Key Design Decisions

**Pathway Controls**:
- File discovery and loading
- Schema validation and type checking
- Novel-backstory pairing via joins
- Document iteration order

**Python/LLM Downstream**:
- Chunking (compute-intensive tokenization)
- Embedding (sentence-transformers)
- Vector similarity search
- LLM classification (OpenAI API)

**Why This Split?**
- Pathway excels at data orchestration, not ML inference
- LLM calls and embeddings need specialized libraries (OpenAI SDK, sentence-transformers)
- Keeps Pathway pipeline declarative and minimal
- Enables easy swapping of ML components without touching data layer

### Evidence Diversity Metrics

To handle **long-context novels (100k+ words)**, we track evidence quality:

| Metric | Description | Purpose |
|--------|-------------|---------|
| `temporal_spread` | Range of chunk positions (0.0-1.0) | Ensures evidence covers full narrative |
| `temporal_marker_ratio` | % chunks with causal keywords | Prioritizes temporally-relevant passages |
| `chunk_coverage` | Unique chunks retrieved | Measures evidence redundancy |
| `avg_temporal_position` | Mean chunk position | Identifies bias toward start/end |

These metrics help validate that retrieval provides **diverse, representative evidence** from the full novel, not just clustered passages.

### Structured Output

**Submission Format** (`results.csv`):
```csv
story_id,prediction
story_1,1
story_2,0
```

**Detailed Format** (`results_detailed.csv`):
```csv
story_id,prediction,label,consistent_score,inconsistent_score,insufficient_score,contradictory_score,evidence_sufficient,evidence_temporal_spread,evidence_temporal_markers,chunks_retrieved,novel_chunks_total,reason
story_1,1,consistent,0.85,0.1,0.05,0.0,true,0.72,0.40,15,87,"Backstory is consistent (score: 0.85)"
```

This enables post-hoc analysis of classifier confidence, evidence quality, and failure modes.
