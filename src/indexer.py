"""Embedding and indexing using Pathway."""
import pathway as pw
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict


class ChunkIndexer:
    """Index chunks with embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize indexer.
        
        Args:
            model_name: Sentence transformer model name
        """
        self.model = SentenceTransformer(model_name)
    
    def embed_chunks(self, chunks: List[Dict[str, str]]) -> List[Dict]:
        """
        Embed chunks and prepare for indexing.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of chunks with embeddings
        """
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Attach embeddings to chunks
        indexed_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            indexed_chunks.append({
                **chunk,
                'embedding': embedding
            })
        
        return indexed_chunks
    
    def build_index(self, chunks: List[Dict]) -> pw.Table:
        """
        Build Pathway table with indexed chunks.
        
        Args:
            chunks: Chunks with embeddings
            
        Returns:
            Pathway table with indexed chunks
        """
        # Prepare rows for Pathway table
        rows = [
            (
                chunk['chunk_id'],
                chunk['story_id'],
                chunk['text'],
                chunk['chunk_num'],
                chunk['embedding'].tobytes()  # Serialize numpy array
            )
            for chunk in chunks
        ]
        
        # Create Pathway table
        index_table = pw.debug.table_from_rows(
            schema=pw.schema_from_types(
                chunk_id=str,
                story_id=str,
                text=str,
                chunk_num=int,
                embedding=bytes
            ),
            rows=rows
        )
        
        return index_table
