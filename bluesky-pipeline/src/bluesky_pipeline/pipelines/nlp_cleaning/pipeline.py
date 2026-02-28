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
                outputs="clean_posts",
                name="clean_text_node",
            ),
            node(
                func=nodes.tokenize_and_lemmatize,
                inputs="clean_posts",
                outputs="tokenized_posts",
                name="tokenize_and_lemmatize_node",
            ),
            node(
                func=nodes.remove_duplicates,
                inputs="tokenized_posts",
                outputs="unique_posts",
                name="remove_duplicates_node",
            ),
            node(
                func=nodes.vectorize_posts,
                inputs="unique_posts",
                outputs=["X_vectors", "labels", "vectorizer"],
                name="vectorize_posts_node",
            ),
            node(
                func=nodes.add_sentiment,
                inputs="unique_posts",
                outputs="posts_with_sentiment",
                name="add_sentiment_node",
            ),
            node(
                func=nodes.compute_sentiment_summary,
                inputs="posts_with_sentiment",
                outputs=None,
                name="compute_sentiment_summary_node",
            ),
            node(
                func=nodes.store_vectors_to_postgres,
                inputs=["posts_with_sentiment", "X_vectors", "params:POSTGRES_URI"],
                outputs=None,
                name="store_vectors_postgres_node",
            ),
            node(
                func=nodes.train_kmeans,  # clustering + interpretability (top TF-IDF terms per cluster)
                inputs=["X_vectors", "unique_posts", "vectorizer"],
                outputs="kmeans_model",
                name="train_kmeans_node",
            ),
            node(
                func=nodes.train_classifier,
                inputs=["X_vectors", "labels"],
                outputs=["classifier_model", "label_encoder"],
                name="train_classifier_node",
            ), 
            node(
                func=nodes.save_models,
                inputs=["vectorizer", "kmeans_model", "classifier_model", "label_encoder"],
                outputs=None,
                name="save_models_node",
            ),
        ]
    )