"""Retrieval of relevant chunks."""
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer


class ChunkRetriever:
    """Retrieve relevant chunks using similarity search."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize retriever.
        
        Args:
            model_name: Sentence transformer model name
        """
        self.model = SentenceTransformer(model_name)
    
    def retrieve(
        self,
        query: str,
        story_id: str,
        indexed_chunks: List[Dict],
        top_k: int = 15
    ) -> List[Dict]:
        """
        Retrieve top-k relevant chunks for a query.
        
        Args:
            query: Query text (backstory)
            story_id: Story identifier to filter chunks
            indexed_chunks: All indexed chunks
            top_k: Number of chunks to retrieve
            
        Returns:
            List of top-k relevant chunks
        """
        # Filter chunks for this story
        story_chunks = [
            chunk for chunk in indexed_chunks
            if chunk['story_id'] == story_id
        ]
        
        if not story_chunks:
            return []
        
        # Encode query
        query_embedding = self.model.encode([query])[0]
        
        # Compute similarities
        similarities = []
        for chunk in story_chunks:
            # Deserialize embedding
            chunk_embedding = np.frombuffer(chunk['embedding'], dtype=np.float32)
            
            # Cosine similarity
            similarity = np.dot(query_embedding, chunk_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
            )
            similarities.append(similarity)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return top-k chunks
        top_chunks = [story_chunks[idx] for idx in top_indices]
        
        return top_chunks
