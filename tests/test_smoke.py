import pytest

def test_pipeline_import():
    from nexus.pipeline import Pipeline
    assert isinstance(Pipeline, type)
