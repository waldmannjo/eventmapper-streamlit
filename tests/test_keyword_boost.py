import pytest
from backend.mapper import extract_keywords_from_code, get_keyword_boost
from codes import CODES

def test_extract_keywords_from_code():
    """Test keyword extraction from code descriptions."""
    arr_code = next(c for c in CODES if c[0] == "ARR")
    keywords = extract_keywords_from_code(arr_code)
    assert isinstance(keywords, list)
    assert len(keywords) > 0
    assert any("arrival" in kw.lower() for kw in keywords)

def test_get_keyword_boost():
    """Test keyword boost calculation."""
    arr_code = next(c for c in CODES if c[0] == "ARR")
    keywords = extract_keywords_from_code(arr_code)
    input_text = "package arrived at depot"
    boost = get_keyword_boost(input_text, keywords)
    assert isinstance(boost, float)
    assert boost > 0
    assert boost <= 0.5

def test_keyword_boost_no_matches():
    """Test keyword boost with no matches returns 0."""
    keywords = ["arrival", "depot", "scan"]
    input_text = "customs clearance in progress"
    boost = get_keyword_boost(input_text, keywords)
    assert boost == 0.0

def test_keyword_boost_capping():
    """Test keyword boost is capped at 0.5."""
    keywords = ["a", "b", "c", "d", "e", "f"]
    input_text = "a b c d e f"
    boost = get_keyword_boost(input_text, keywords)
    assert boost == 0.5

def test_keyword_boost_affects_ranking():
    """Test that keyword boost improves ranking of matching codes."""
    import numpy as np
    base_scores = np.array([0.5, 0.6, 0.7])
    keywords_list = [
        ["arrival", "depot"],
        ["customs", "clearance"],
        ["damage", "broken"]
    ]
    input_text = "package arrival at depot"
    boosted_scores = []
    for i, base_score in enumerate(base_scores):
        boost = get_keyword_boost(input_text, keywords_list[i])
        boosted_scores.append(base_score + boost)
    boosted_scores = np.array(boosted_scores)
    assert np.argmax(boosted_scores) == 0
