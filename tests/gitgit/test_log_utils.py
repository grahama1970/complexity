"""
Tests for log_utils.py functionality.
"""
import pytest
from complexity.gitgit.log_utils import truncate_large_value, log_safe_results

def test_truncate_large_value_string():
    """Test truncation of different types of strings."""
    # Test regular string truncation
    long_string = "a" * 200
    truncated = truncate_large_value(long_string, max_str_len=100)
    assert len(truncated) < 200  # Should be truncated
    assert "..." in truncated  # Should contain ellipsis
    
    # Test short string (no truncation)
    short_string = "Short string"
    assert truncate_large_value(short_string, max_str_len=100) == short_string
    
    # Test base64 image truncation
    base64_header = "data:image/png;base64,"
    base64_data = "A" * 200
    base64_string = base64_header + base64_data
    
    truncated = truncate_large_value(base64_string, max_str_len=100)
    assert base64_header in truncated  # Header should be preserved
    assert len(truncated) < len(base64_string)  # Should be truncated
    assert "..." in truncated  # Should contain ellipsis

def test_truncate_large_value_lists():
    """Test truncation of lists."""
    # Test regular list
    short_list = [1, 2, 3, 4, 5]
    assert truncate_large_value(short_list, max_list_elements_shown=10) == short_list
    
    # Test long list
    long_list = list(range(50))
    truncated = truncate_large_value(long_list, max_list_elements_shown=10)
    assert isinstance(truncated, str)  # Should be summarized as string
    assert "<50 int elements>" in truncated
    
    # Test empty list
    empty_list = []
    truncated = truncate_large_value(empty_list, max_list_elements_shown=10)
    assert truncated == empty_list  # Should not summarize empty lists

def test_truncate_large_value_dict():
    """Test truncation of dictionaries with nested values."""
    # Simple dict
    simple_dict = {"key1": "value1", "key2": "value2"}
    assert truncate_large_value(simple_dict) == simple_dict
    
    # Dict with long string
    long_str_dict = {"key1": "a" * 200}
    truncated = truncate_large_value(long_str_dict, max_str_len=100)
    assert "..." in truncated["key1"]
    
    # Dict with nested values
    nested_dict = {
        "key1": "value1",
        "nested": {
            "long_value": "b" * 200
        },
        "list": list(range(50))
    }
    
    truncated = truncate_large_value(nested_dict, max_str_len=100, max_list_elements_shown=10)
    assert "..." in truncated["nested"]["long_value"]
    assert isinstance(truncated["list"], str)  # Should be summarized as string

def test_log_safe_results():
    """Test log_safe_results function."""
    # Valid input
    test_docs = [
        {"id": 1, "text": "a" * 200, "embeddings": list(range(100))},
        {"id": 2, "text": "b" * 50, "embeddings": list(range(20))},
    ]
    
    safe_results = log_safe_results(test_docs)
    
    # Check first doc
    assert safe_results[0]["id"] == 1
    assert "..." in safe_results[0]["text"]  # Long text should be truncated
    assert isinstance(safe_results[0]["embeddings"], str)  # Long list should be summarized
    
    # Check second doc
    assert safe_results[1]["id"] == 2
    assert safe_results[1]["text"] == "b" * 50  # Short text should not be truncated
    # Check if embeddings are summarized (log_utils default might be different than our test)
    if isinstance(safe_results[1]["embeddings"], str):
        assert "<20 int elements>" in safe_results[1]["embeddings"]
    else:
        assert safe_results[1]["embeddings"] == list(range(20))
    
    # Test with invalid inputs
    with pytest.raises(TypeError):
        log_safe_results("not a list")
    
    with pytest.raises(TypeError):
        log_safe_results([{"valid": "dict"}, "not a dict"])
    
    # Empty list is valid
    assert log_safe_results([]) == []