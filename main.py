"""Main entry point for the classification pipeline."""
import os
from src.pipeline import ClassificationPipeline
import config


def main():
    """Run the classification pipeline."""
    # Ensure output directory exists
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Validate API key
    if not config.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY not found. "
            "Set it via environment variable: export OPENAI_API_KEY='your-key'"
        )
    
    # Initialize pipeline
    pipeline = ClassificationPipeline(
        data_dir=config.DATA_DIR,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        top_k=config.TOP_K,
        api_key=config.OPENAI_API_KEY,
        model_name=config.MODEL_NAME,
        prompt_template=config.CLASSIFICATION_PROMPT
    )
    
    # Run pipeline
    print("Starting classification pipeline...\n")
    results_df = pipeline.run(config.RESULTS_FILE)
    
    print(f"\n=== Pipeline Complete ===")
    print(f"Total predictions: {len(results_df)}")
    print(f"Consistent (1): {sum(results_df['prediction'] == 1)}")
    print(f"Inconsistent (0): {sum(results_df['prediction'] == 0)}")
    print(f"Results: {config.RESULTS_FILE}")


if __name__ == "__main__":
    main()
