from kedro.pipeline import Pipeline
from bluesky_pipeline.pipelines.nlp_cleaning import pipeline as nlp_cleaning_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines explicitly."""
    return {
        "__default__": nlp_cleaning_pipeline.create_pipeline(),
        "nlp_cleaning": nlp_cleaning_pipeline.create_pipeline(),
    }