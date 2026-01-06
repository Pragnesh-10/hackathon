# Track A: Novel-Backstory Consistency Classifier

Hypothesis-based causal consistency system using Pathway framework. Evaluates whether a backstory is logically and causally coherent with a full novel through evidence retrieval, temporal reasoning, and multi-hypothesis scoring. Handles 100k+ word novels efficiently with token-based chunking and semantic search.

## Architecture

```
main.py → pipeline.py → [ingestion → chunking → indexing → retrieval → classification] → results.csv
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

## Data Structure

```
data/
├── story_1/
│   ├── novel.txt
│   └── backstory.txt
├── story_2/
│   ├── novel.txt
│   └── backstory.txt
└── ...
```

## Run

```bash
python main.py
```

Output: `output/results.csv`

## Pipeline Stages

1. **Ingestion**: Load novels and backstories into Pathway tables
2. **Chunking**: Split novels into 1000-token chunks with 200-token overlap
3. **Indexing**: Embed chunks using sentence-transformers
4. **Retrieval**: Query with backstory, retrieve top 15 relevant chunks per story
5. **Classification**: LLM analyzes evidence + backstory, outputs binary prediction
6. **Output**: Write `story_id,prediction` to CSV

## Configuration

Edit `config.py` to modify:
- Chunk size/overlap
- Top-k retrieval count
- LLM model
- Classification prompt

## Components

- `src/ingestion.py`: Pathway-based data loading
- `src/chunker.py`: Token-based text chunking
- `src/indexer.py`: Embedding and indexing
- `src/retriever.py`: Similarity-based retrieval
- `src/classifier.py`: LLM-based binary classification
- `src/pipeline.py`: End-to-end orchestration
