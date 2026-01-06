"""Long-context story ingestion using Pathway."""
import pathway as pw
from typing import Dict
import os
import glob


class LongContextNovelSchema(pw.Schema):
    """Richer schema for novel data with metadata."""
    story_id: str
    doc_id: str
    document_type: str
    text: str
    word_count: int
    char_count: int
    file_path: str


class LongContextBackstorySchema(pw.Schema):
    """Richer schema for backstory data with metadata."""
    story_id: str
    doc_id: str
    document_type: str
    text: str
    word_count: int
    char_count: int
    file_path: str


class InputFileSchema(pw.Schema):
    """Schema for raw file input."""
    data: bytes
    _metadata: pw.Json


def ingest_long_context_stories(data_dir: str) -> Dict[str, pw.Table]:
    """
    Ingest long-context novels and backstories using native Pathway IO.
    
    Uses Pathway transforms to enrich data with metadata fields.
    Handles 100k+ word novels efficiently.
    
    Expected structure:
    data/
        story_1/
            novel.txt
            backstory.txt
        story_2/
            novel.txt
            backstory.txt
    
    Args:
        data_dir: Root directory containing story subdirectories
        
    Returns:
        Dict with 'novels' and 'backstories' Pathway tables
    """
    # Read all novel files using native Pathway file connector
    novel_pattern = os.path.join(data_dir, "*/novel.txt")
    novels_raw = pw.io.fs.read(
        novel_pattern,
        format="binary",
        mode="static",
        with_metadata=True
    )
    
    # Read all backstory files
    backstory_pattern = os.path.join(data_dir, "*/backstory.txt")
    backstories_raw = pw.io.fs.read(
        backstory_pattern,
        format="binary",
        mode="static",
        with_metadata=True
    )
    
    # Transform novels using Pathway operations
    novels_table = novels_raw.select(
        # Decode bytes to text
        text=pw.apply(lambda x: x.decode('utf-8', errors='ignore'), novels_raw.data),
        # Extract story_id from file path
        file_path=pw.apply(lambda m: m["path"], novels_raw._metadata),
    ).select(
        pw.this.text,
        pw.this.file_path,
        story_id=pw.apply(
            lambda p: os.path.basename(os.path.dirname(p)),
            pw.this.file_path
        ),
        # Add document identification fields
        doc_id=pw.apply(
            lambda p: f"{os.path.basename(os.path.dirname(p))}_novel",
            pw.this.file_path
        ),
        document_type=pw.apply(lambda _: "novel", pw.this.file_path),
        # Add metadata fields
        word_count=pw.apply(lambda t: len(t.split()), pw.this.text),
        char_count=pw.apply(lambda t: len(t), pw.this.text),
    )
    
    # Transform backstories using Pathway operations
    backstories_table = backstories_raw.select(
        # Decode bytes to text
        text=pw.apply(lambda x: x.decode('utf-8', errors='ignore'), backstories_raw.data),
        # Extract story_id from file path
        file_path=pw.apply(lambda m: m["path"], backstories_raw._metadata),
    ).select(
        pw.this.text,
        pw.this.file_path,
        story_id=pw.apply(
            lambda p: os.path.basename(os.path.dirname(p)),
            pw.this.file_path
        ),
        # Add document identification fields
        doc_id=pw.apply(
            lambda p: f"{os.path.basename(os.path.dirname(p))}_backstory",
            pw.this.file_path
        ),
        document_type=pw.apply(lambda _: "backstory", pw.this.file_path),
        # Add metadata fields
        word_count=pw.apply(lambda t: len(t.split()), pw.this.text),
        char_count=pw.apply(lambda t: len(t), pw.this.text),
    )
    
    return {
        'novels': novels_table,
        'backstories': backstories_table
    }
