# Track A: Novel-Backstory Consistency Classifier

Claim-based causal consistency classifier that evaluates whether character backstories are logically and causally coherent with source novels. Uses evidence retrieval, hypothesis scoring, and strict validation to detect contradictions, timeline violations, and insufficient evidence.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Add your OpenAI API key to .env file
echo "OPENAI_API_KEY=your-key-here" > .env

# Run classifier
python3 classify.py
```

## Output

- `output/hardened_results.csv` - Binary predictions with confidence scores
- `output/evidence_log.json` - Detailed evidence trail for each claim

## How It Works

The classifier breaks backstories into atomic claims (childhood events, beliefs, affiliations, irreversible actions), retrieves claim-specific evidence from novels, and applies strict Python-controlled rules:

- **Hard contradictions** → prediction = 0
- **Timeline violations** → prediction = 0  
- **Missing evidence for key claims** → prediction = 0
- **Mixed or ambiguous evidence** → prediction = 0 (conservative bias)
- **All claims supported** → prediction = 1

All prediction logic is in Python (not LLM-controlled). Errors always fail to 0 (never default to 1).

## Configuration

Edit `config.py` to modify:
- Chunk size/overlap
- Top-k retrieval
- Model selection

## Architecture

```
classify.py
    ├── Data Loading (train.csv + novels)
    ├── For each backstory:
    │   ├── Decompose into atomic claims
    │   ├── Retrieve evidence per claim (top 2-3 chunks)
    │   ├── Classify evidence (supports/contradicts/insufficient)
    │   └── Apply Python rules → prediction
    └── Output results + evidence log
```

## Data Structure

```
data/Dataset/
├── train.csv              # Training examples
├── test.csv               # Test examples
└── Books/
    ├── In search of the castaways.txt
    └── The Count of Monte Cristo.txt
```

## Components

- `src/chunker.py` - Token-based text chunking
- `src/indexer.py` - Embedding utilities
- `src/retriever.py` - Similarity-based retrieval
- `src/classifier.py` - Original hypothesis-based classifier
- `src/pipeline.py` - Pathway-based orchestration (legacy)
