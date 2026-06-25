from kedro.pipeline import Pipeline
from bluesky_pipeline.pipelines.nlp_cleaning import pipeline as nlp_cleaning_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines explicitly."""
    full_pipeline = nlp_cleaning_pipeline.create_pipeline()
    scoring_pipeline = nlp_cleaning_pipeline.create_scoring_pipeline()
    return {
        "__default__": full_pipeline,
        "nlp_cleaning": full_pipeline,
        "nlp_full": full_pipeline,
        "nlp_scoring": scoring_pipeline,
    }
