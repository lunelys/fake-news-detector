from kedro.framework.session import KedroSession

project_name = "bluesky_pipeline"  # your Kedro project name

with KedroSession.create(project_name) as session:
    catalog = session.catalog
    clean_posts_data = catalog.load("clean_posts")  # this is the MemoryDataset
    print(f"Number of cleaned posts in memory: {len(clean_posts_data)}")
    if clean_posts_data:
        print("Sample post:", clean_posts_data[0])