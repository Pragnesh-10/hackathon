"""End-to-end classification pipeline with Pathway-driven orchestration."""
import pandas as pd
import pathway as pw
from typing import List, Dict
from src.ingestion import ingest_long_context_stories
from src.chunker import TextChunker
from src.indexer import ChunkIndexer
from src.retriever import ChunkRetriever
from src.classifier import BackstoryClassifier


def calculate_evidence_diversity(evidence_chunks: List[Dict]) -> Dict[str, float]:
    """
    Calculate evidence diversity metrics for long-context analysis.
    
    Args:
        evidence_chunks: List of retrieved chunks
        
    Returns:
        Dict with diversity metrics
    """
    if not evidence_chunks:
        return {
            'temporal_spread': 0.0,
            'chunk_coverage': 0.0,
            'avg_temporal_position': 0.0,
            'temporal_marker_ratio': 0.0
        }
    
    # Temporal spread: range of temporal positions
    positions = [c.get('temporal_position', 0.0) for c in evidence_chunks]
    temporal_spread = max(positions) - min(positions) if positions else 0.0
    
    # Chunk coverage: unique chunks vs total possible
    chunk_indices = [c.get('chunk_index', 0) for c in evidence_chunks]
    chunk_coverage = len(set(chunk_indices)) / len(evidence_chunks) if evidence_chunks else 0.0
    
    # Average temporal position
    avg_temporal_position = sum(positions) / len(positions) if positions else 0.0
    
    # Temporal marker ratio: chunks with temporal markers
    temporal_chunks = sum(1 for c in evidence_chunks if c.get('has_temporal_markers', False))
    temporal_marker_ratio = temporal_chunks / len(evidence_chunks) if evidence_chunks else 0.0
    
    return {
        'temporal_spread': round(temporal_spread, 3),
        'chunk_coverage': round(chunk_coverage, 3),
        'avg_temporal_position': round(avg_temporal_position, 3),
        'temporal_marker_ratio': round(temporal_marker_ratio, 3)
    }


class ClassificationPipeline:
    """Full pipeline with Pathway-driven document iteration."""
    
    def __init__(
        self,
        data_dir: str,
        chunk_size: int,
        chunk_overlap: int,
        top_k: int,
        api_key: str,
        model_name: str,
        prompt_template: str
    ):
        """
        Initialize pipeline.
        
        Args:
            data_dir: Data directory path
            chunk_size: Chunk size in tokens
            chunk_overlap: Overlap in tokens
            top_k: Number of chunks to retrieve
            api_key: OpenAI API key
            model_name: LLM model name
            prompt_template: Classification prompt template (unused in new design)
        """
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        
        # Initialize components (classifier stays outside Pathway)
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.indexer = ChunkIndexer()
        self.retriever = ChunkRetriever()
        self.classifier = BackstoryClassifier(api_key, model_name, prompt_template)
    
    def run(self, output_path: str) -> pd.DataFrame:
        """
        Run full pipeline with Pathway-driven iteration.
        
        Args:
            output_path: Path to save results CSV
            
        Returns:
            DataFrame with predictions and metadata
        """
        print("=== Stage 1: Pathway Ingestion ===")
        tables = ingest_long_context_stories(self.data_dir)
        novels_table = tables['novels']
        backstories_table = tables['backstories']
        
        # Pathway controls document iteration via joins
        # Join novels and backstories on story_id
        joined_table = novels_table.join(
            backstories_table,
            novels_table.story_id == backstories_table.story_id,
            id=novels_table.id
        ).select(
            story_id=novels_table.story_id,
            novel_text=novels_table.text,
            backstory_text=backstories_table.text,
            novel_word_count=novels_table.word_count,
            backstory_word_count=backstories_table.word_count
        )
        
        # Extract data (Pathway iterates through joined documents)
        documents = list(joined_table)
        
        print(f"Pathway joined {len(documents)} story pairs")
        
        print("\n=== Stage 2: Chunking ===")
        all_chunks = []
        story_chunk_map = {}
        
        for doc in documents:
            story_id = doc.story_id
            novel_text = doc.novel_text
            chunks = self.chunker.chunk_text(novel_text, story_id)
            all_chunks.extend(chunks)
            story_chunk_map[story_id] = len(chunks)
            print(f"Story {story_id}: {len(chunks)} chunks")
        
        print(f"\nTotal chunks: {len(all_chunks)}")
        
        print("\n=== Stage 3: Indexing ===")
        indexed_chunks = self.indexer.embed_chunks(all_chunks)
        print(f"Indexed {len(indexed_chunks)} chunks")
        
        print("\n=== Stage 4: Classification (Downstream) ===")
        results = []
        
        # Pathway-driven iteration: process each document
        for doc in documents:
            story_id = doc.story_id
            backstory_text = doc.backstory_text
            
            print(f"\nProcessing {story_id}...")
            print(f"  Novel: {doc.novel_word_count} words, Backstory: {doc.backstory_word_count} words")
            
            # Retrieve relevant chunks
            print(f"  Retrieving top {self.top_k} chunks...")
            evidence_chunks = self.retriever.retrieve(
                backstory_text,
                story_id,
                indexed_chunks,
                self.top_k
            )
            
            # Calculate evidence diversity
            diversity = calculate_evidence_diversity(evidence_chunks)
            print(f"  Evidence diversity: temporal_spread={diversity['temporal_spread']}, "
                  f"temporal_markers={diversity['temporal_marker_ratio']}")
            
            # Classify (downstream, outside Pathway)
            print(f"  Classifying...")
            result = self.classifier.classify(backstory_text, evidence_chunks)
            
            # Structured output with full metadata
            results.append({
                'story_id': story_id,
                'prediction': result['prediction'],
                'label': result['label'],
                'consistent_score': result['scores'].get('consistent', 0.0),
                'inconsistent_score': result['scores'].get('inconsistent', 0.0),
                'insufficient_score': result['scores'].get('insufficient', 0.0),
                'contradictory_score': result['scores'].get('contradictory', 0.0),
                'evidence_sufficient': result['evidence_sufficient'],
                'evidence_temporal_spread': diversity['temporal_spread'],
                'evidence_temporal_markers': diversity['temporal_marker_ratio'],
                'chunks_retrieved': len(evidence_chunks),
                'novel_chunks_total': story_chunk_map.get(story_id, 0),
                'reason': result['reason']
            })
            
            print(f"  Label: {result['label']}")
            print(f"  Prediction: {result['prediction']}")
            print(f"  Reason: {result['reason']}")
        
        print("\n=== Stage 5: Structured Output ===")
        df = pd.DataFrame(results)
        
        # Save submission format (story_id, prediction only)
        submission_df = df[['story_id', 'prediction']]
        submission_df.to_csv(output_path, index=False)
        print(f"Submission results saved to {output_path}")
        
        # Save detailed results with all metadata
        detailed_path = output_path.replace('.csv', '_detailed.csv')
        df.to_csv(detailed_path, index=False)
        print(f"Detailed results saved to {detailed_path}")
        
        return df
