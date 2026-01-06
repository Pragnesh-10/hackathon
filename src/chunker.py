"""Text chunking with overlap and temporal heuristics."""
import tiktoken
import re
from typing import List, Dict


class TextChunker:
    """Chunk text into overlapping segments with temporal heuristics."""
    
    # Simple temporal/causal markers for backstory consistency checking
    TEMPORAL_MARKERS = {
        'before', 'after', 'then', 'when', 'while', 'during', 'until',
        'since', 'first', 'next', 'later', 'previously', 'earlier',
        'eventually', 'finally', 'initially', 'subsequently'
    }
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize chunker with configurable parameters.
        
        Args:
            chunk_size: Target chunk size in tokens (default: 1000)
            overlap: Overlap between chunks in tokens (default: 200)
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap cannot be negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")
            
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoder = tiktoken.get_encoding("cl100k_base")
    
    def chunk_text(self, text: str, story_id: str) -> List[Dict[str, str]]:
        """
        Chunk text into overlapping segments with detailed metadata.
        
        Args:
            text: Input text to chunk
            story_id: Story identifier
            
        Returns:
            List of chunks with metadata including:
            - chunk_id: Unique identifier for chunk
            - chunk_index: Sequential index (0-based)
            - story_id: Story identifier
            - text: Chunk text content
            - start_char: Absolute starting character position in original text
            - end_char: Absolute ending character position in original text
            - token_count: Number of tokens in chunk
            - overlap_size: Number of overlapping tokens with previous chunk
            - temporal_position: Normalized position in document (0.0 to 1.0)
            - has_temporal_markers: Boolean indicating presence of temporal/causal markers
        """
        # Encode text to tokens
        tokens = self.encoder.encode(text)
        
        chunks = []
        token_idx = 0
        chunk_index = 0
        
        while token_idx < len(tokens):
            # Extract chunk tokens
            end_token_idx = min(token_idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[token_idx:end_token_idx]
            
            # Decode back to text
            chunk_text = self.encoder.decode(chunk_tokens)
            
            # Calculate ABSOLUTE character positions in original text
            # Decode up to start position to find char offset
            prefix_text = self.encoder.decode(tokens[:token_idx])
            start_char = len(prefix_text)
            end_char = start_char + len(chunk_text)
            
            # Calculate overlap size
            if chunk_index == 0:
                overlap_tokens = 0
            else:
                overlap_tokens = min(self.overlap, token_idx)
            
            # Calculate temporal position (0.0 = start, 1.0 = end)
            temporal_position = token_idx / len(tokens) if len(tokens) > 0 else 0.0
            
            # Simple temporal/causal heuristic: check for temporal markers
            chunk_lower = chunk_text.lower()
            has_temporal_markers = any(
                re.search(r'\b' + marker + r'\b', chunk_lower)
                for marker in self.TEMPORAL_MARKERS
            )
            
            # Create chunk metadata
            chunk_id = f"{story_id}_chunk_{chunk_index}"
            chunks.append({
                'chunk_id': chunk_id,
                'chunk_index': chunk_index,
                'story_id': story_id,
                'text': chunk_text,
                'start_char': start_char,
                'end_char': end_char,
                'token_count': len(chunk_tokens),
                'overlap_size': overlap_tokens,
                'temporal_position': temporal_position,
                'has_temporal_markers': has_temporal_markers
            })
            
            chunk_index += 1
            
            # Move start position with overlap
            token_idx += self.chunk_size - self.overlap
            
            # Break if we're at the end
            if end_token_idx >= len(tokens):
                break
        
        return chunks
