from kedro.pipeline import Pipeline, node
from . import nodes

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=nodes.load_raw_posts,
                inputs=dict(
                    mongo_uri="params:MONGO_URI",
                    db_name="params:DB_NAME",
                    raw_collections="params:RAW_COLLECTIONS",
                ),
                outputs="raw_posts",
                name="load_raw_posts_node",
            ),
            node(
                func=nodes.clean_text_node,
                inputs="raw_posts",
                outputs="clean_posts_cleaned_text",
                name="clean_text_node",
            ),
            node(
                func=nodes.tokenize_and_lemmatize,
                inputs="clean_posts_cleaned_text",
                outputs="clean_posts",
                name="tokenize_and_lemmatize_node",
            ),
            node(
                func=nodes.debug_clean_posts,
                inputs="clean_posts",
                outputs="clean_posts_debug",
                name="debug_clean_posts_node"
            ),
            node(
                func=nodes.store_clean_posts,
                inputs=dict(
                    posts="clean_posts",
                    mongo_uri="params:MONGO_URI",
                    db_name="params:DB_NAME",
                    clean_collection="params:CLEAN_COLLECTION",
                ),
                outputs=None,
                name="store_clean_posts_node",
            ),
        ]
    )