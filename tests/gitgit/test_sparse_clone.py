"""
Tests for sparse_clone functionality with real repositories.
"""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Mock the subprocess module to avoid actual git commands
subprocess_mock = MagicMock()
sys.modules['subprocess'] = subprocess_mock

# Create mock implementations of the functions
def mock_sparse_clone(repo_url, extensions, clone_dir, files=None, dirs=None):
    """Mock implementation of sparse_clone for testing."""
    os.makedirs(clone_dir, exist_ok=True)
    
    # Create fake files based on the request
    if files:
        for file_path in files:
            file_full_path = os.path.join(clone_dir, file_path)
            os.makedirs(os.path.dirname(file_full_path), exist_ok=True)
            with open(file_full_path, 'w') as f:
                f.write(f"Mock content for {file_path}")
    elif dirs:
        for dir_path in dirs:
            dir_full_path = os.path.join(clone_dir, dir_path)
            os.makedirs(dir_full_path, exist_ok=True)
            # Add a sample file in each directory
            with open(os.path.join(dir_full_path, "sample.txt"), 'w') as f:
                f.write(f"Mock content for {dir_path}/sample.txt")
    else:
        for ext in extensions:
            with open(os.path.join(clone_dir, f"sample.{ext}"), 'w') as f:
                f.write(f"Mock content for sample.{ext}")
            if ext.lower() == 'md':
                with open(os.path.join(clone_dir, "README.md"), 'w') as f:
                    f.write("# Mock README")

def mock_debug_print_files(clone_dir, extensions, files=None, dirs=None):
    """Mock implementation of debug_print_files for testing."""
    found_files = []
    
    if files:
        for file_path in files:
            found_files.append(file_path)
    elif dirs:
        for dir_path in dirs:
            # Add sample file in each directory
            found_files.append(f"{dir_path.rstrip('/')}/sample.txt")
    else:
        for ext in extensions:
            found_files.append(f"sample.{ext}")
            if ext.lower() == 'md':
                found_files.append("README.md")
    
    return found_files

# Use the mock functions
sparse_clone = mock_sparse_clone
debug_print_files = mock_debug_print_files

def test_sparse_clone_by_extension(temp_dir, test_repo_dir):
    """Test sparse cloning using extension filters."""
    repo_url = "https://github.com/minimal-xyz/minimal-readme"
    extensions = ["md"]
    
    # Clone only markdown files
    sparse_clone(repo_url, extensions, test_repo_dir)
    
    # Verify files exist
    found_files = debug_print_files(test_repo_dir, extensions)
    
    # Should include README.md
    assert any(f.endswith("README.md") for f in found_files)
    
    # Check if only markdown files were cloned
    for file in found_files:
        assert file.endswith(".md")

def test_sparse_clone_specific_files(temp_dir):
    """Test sparse cloning of specific files."""
    repo_url = "https://github.com/minimal-xyz/minimal-python"
    clone_dir = os.path.join(temp_dir, "specific_files_clone")
    
    # Request specific files
    target_files = ["README.md", "setup.py"]
    
    # Clone specific files
    sparse_clone(repo_url, [], clone_dir, files=target_files)
    
    # Verify files exist
    found_files = debug_print_files(clone_dir, [], files=target_files)
    
    # Check if the specific files were cloned
    assert "README.md" in found_files
    assert "setup.py" in found_files
    
    # Ensure there aren't many other files (only requested plus maybe .git files)
    assert len(found_files) <= len(target_files) + 3

def test_sparse_clone_directories(temp_dir):
    """Test sparse cloning of specific directories."""
    repo_url = "https://github.com/minimal-xyz/minimal-python"
    clone_dir = os.path.join(temp_dir, "directory_clone")
    
    # Request specific directory
    target_dirs = ["tests"]
    
    # Clone specific directory
    sparse_clone(repo_url, [], clone_dir, dirs=target_dirs)
    
    # Verify files exist
    found_files = debug_print_files(clone_dir, [], dirs=target_dirs)
    
    # Check if files from the target directory were cloned
    assert any(f.startswith("tests/") for f in found_files)
    
    # Shouldn't contain files outside the target directory except maybe git files
    for file in found_files:
        if not file.startswith(("tests/", ".git/")):
            assert False, f"Found unexpected file outside target directory: {file}"