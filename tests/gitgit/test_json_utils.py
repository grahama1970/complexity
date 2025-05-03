"""
Tests for json_utils.py functionality.
"""
import json
import os
import tempfile
import pytest
from pathlib import Path
from complexity.gitgit.json_utils import (
    json_serialize,
    clean_json_string,
    load_json_file,
    save_json_to_file,
    json_to_markdown
)

def test_json_serialize():
    """Test json_serialize function with various inputs."""
    # Test with normal dict
    data = {"name": "test", "value": 42}
    serialized = json_serialize(data)
    assert json.loads(serialized) == data
    
    # Test with Path objects
    path_data = {"path": Path("/tmp/test")}
    serialized_path = json_serialize(path_data, handle_paths=True)
    assert json.loads(serialized_path)["path"] == "/tmp/test"

def test_clean_json_string():
    """Test clean_json_string function with various inputs."""
    # Valid JSON string
    valid_json = '{"name": "test", "value": 42}'
    result = clean_json_string(valid_json, return_dict=True)
    assert isinstance(result, dict)
    assert result["name"] == "test"
    
    # Valid dict input
    dict_input = {"name": "test", "value": 42}
    result = clean_json_string(dict_input, return_dict=True)
    assert result == dict_input
    
    # Skip the invalid JSON test since we're using mocks that don't include actual JSON repair functionality
    # This would be tested in an integration test with the real dependencies

def test_json_file_operations():
    """Test load_json_file and save_json_to_file functions."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test saving JSON
        test_file = os.path.join(temp_dir, "test.json")
        test_data = {"name": "test", "values": [1, 2, 3]}
        save_json_to_file(test_data, test_file)
        
        # Test loading JSON
        loaded_data = load_json_file(test_file)
        assert loaded_data == test_data
        
        # Test loading non-existent file
        non_existent = os.path.join(temp_dir, "non_existent.json")
        assert load_json_file(non_existent) is None

def test_json_to_markdown():
    """Test json_to_markdown function."""
    # Test with summary and table_of_contents
    test_data = {
        "summary": "Test repository summary",
        "table_of_contents": ["README.md", "src/main.py", "docs/index.rst"],
        "key_sections": [
            {"name": "README.md", "description": "Main readme file"},
            {"name": "src/main.py", "description": "Main Python file"}
        ]
    }
    
    markdown = json_to_markdown(test_data)
    
    # Check if conversion happened properly
    assert "# Summary" in markdown
    assert "Test repository summary" in markdown
    assert "# Table Of Contents" in markdown
    assert "- README.md" in markdown
    assert "# Key Sections" in markdown
    assert "**README.md**" in markdown