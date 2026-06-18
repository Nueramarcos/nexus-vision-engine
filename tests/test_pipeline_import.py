import pytest

def test_pipeline_import():
    try:
        from nexus import pipeline
    except ImportError as e:
        pytest.fail(f"Failed to import nexus.pipeline: {e}")
