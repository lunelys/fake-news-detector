"""
This module contains example tests for a Kedro project.
Tests should be placed in ``src/tests``, in modules that mirror your
project's structure, and in files named test_*.py.
"""
from pathlib import Path

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

# The tests below are here for the demonstration purpose
# and should be replaced with the ones testing the project
# functionality

class TestKedroRun:
    def test_kedro_project_bootstraps(self):
        project_path = Path(__file__).resolve().parents[1]
        metadata = bootstrap_project(project_path)
        import bluesky_pipeline

        with KedroSession.create(project_path=project_path) as session:
            assert metadata.package_name == "bluesky_pipeline"
            assert bluesky_pipeline.__name__ == "bluesky_pipeline"
            assert session.load_context().project_path == project_path
