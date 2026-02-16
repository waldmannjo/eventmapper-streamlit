"""Tests for MAPPER_CONFIG in app.py."""
import ast
import os
import pytest


def _extract_mapper_config():
    """Extract MAPPER_CONFIG from app.py using AST parsing.

    This avoids importing app.py, which would trigger Streamlit UI calls.
    """
    app_path = os.path.join(os.path.dirname(__file__), "..", "app.py")
    with open(app_path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MAPPER_CONFIG":
                    return ast.literal_eval(node.value)
    raise AssertionError("MAPPER_CONFIG not found in app.py")


@pytest.fixture
def mapper_config():
    return _extract_mapper_config()


def test_mapper_config_exists(mapper_config):
    """Test that MAPPER_CONFIG exists in app.py."""
    assert isinstance(mapper_config, dict)


def test_mapper_config_has_required_keys(mapper_config):
    """Test MAPPER_CONFIG has all required settings."""
    required_keys = [
        "use_multilingual_ce",
        "use_bm25",
        "use_keyword_boost",
        "embedding_dimensions",
        "knn_threshold",
    ]
    for key in required_keys:
        assert key in mapper_config, f"Missing config key: {key}"


def test_mapper_config_types(mapper_config):
    """Test MAPPER_CONFIG values have correct types."""
    assert isinstance(mapper_config["use_multilingual_ce"], bool)
    assert isinstance(mapper_config["use_bm25"], bool)
    assert isinstance(mapper_config["use_keyword_boost"], bool)
    assert isinstance(mapper_config["embedding_dimensions"], int)
    assert isinstance(mapper_config["knn_threshold"], float)
