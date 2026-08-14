from src.preprocess import normalize

def test_normalize_sums_to_one():
    result = normalize([1, 1, 2])
    assert abs(sum(result) - 1.0) < 1e-6