"""Configuration for the pipeline."""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Data paths
DATA_DIR = "data"
OUTPUT_DIR = "output"
RESULTS_FILE = os.path.join(OUTPUT_DIR, "results.csv")

# Chunking parameters
CHUNK_SIZE = 1000  # tokens
CHUNK_OVERLAP = 200  # tokens

# Retrieval parameters
TOP_K = 15  # Number of chunks to retrieve per query

# LLM parameters
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# Classification prompt template
CLASSIFICATION_PROMPT = """You are analyzing whether a backstory is logically and causally consistent with evidence from a novel.

**Novel Evidence (Retrieved Chunks):**
{evidence}

**Backstory to Verify:**
{backstory}

**Task:**
Determine if the backstory is logically and causally consistent with the novel evidence.
- Answer ONLY with "1" if consistent
- Answer ONLY with "0" if inconsistent

**Requirements:**
- Check causal relationships
- Verify logical consistency
- Look for contradictions
- Consider timeline coherence

**Output (1 or 0 only):**"""
