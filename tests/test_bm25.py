import pytest
from backend.mapper import build_bm25_index, get_bm25_scores
from codes import CODES

def test_build_bm25_index():
    """Test BM25 index builds from CODES."""
    bm25_index = build_bm25_index()
    assert bm25_index is not None
    assert len(bm25_index.doc_freqs) > 0

def test_get_bm25_scores():
    """Test BM25 scores are computed correctly."""
    bm25_index = build_bm25_index()
    query = "package arrived depot"
    scores = get_bm25_scores(bm25_index, query)
    assert len(scores) == len(CODES)
    assert all(isinstance(s, (float, int)) for s in scores)
    assert all(s >= 0 for s in scores)

def test_bm25_keyword_match():
    """Test BM25 gives higher scores to keyword matches."""
    bm25_index = build_bm25_index()
    scores_arrival = get_bm25_scores(bm25_index, "arrival at depot")
    scores_customs = get_bm25_scores(bm25_index, "customs clearance")

    arr_idx = next(i for i, c in enumerate(CODES) if c[0] == "ARR")
    cus_idx = next(i for i, c in enumerate(CODES) if c[0] == "CUS")

    assert scores_arrival[arr_idx] > scores_arrival[cus_idx]
    assert scores_customs[cus_idx] > scores_customs[arr_idx]
