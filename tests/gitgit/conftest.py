"""
Pytest configuration and fixtures for gitgit tests.
"""
import os
import sys
import shutil
import pytest
import tempfile
from unittest.mock import MagicMock

# Create a more comprehensive mock for litellm and its submodules
class MockLiteLLM(MagicMock):
    class Caching:
        class Cache:
            def __init__(self, *args, **kwargs):
                pass
            
            def __call__(self, *args, **kwargs):
                return self
        
        LiteLLMCacheType = MagicMock()
        LiteLLMCacheType.REDIS = "redis"
        LiteLLMCacheType.LOCAL = "local"
        
    caching = Caching()
    
    def completion(*args, **kwargs):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = '{"summary": "Test"}'
        return mock_response
    
    def enable_cache(*args, **kwargs):
        pass

# Setup the mocks
litellm_mock = MockLiteLLM()
sys.modules['litellm'] = litellm_mock
sys.modules['litellm.caching'] = MagicMock()
sys.modules['litellm.caching.caching'] = litellm_mock.caching

# Mock other required dependencies
mock_modules = [
    'tree_sitter',
    'tree_sitter_languages',
    'markitdown',
    'tiktoken',
    'json_repair'
]

# Add mocks for any modules that might not be installed
for mod_name in mock_modules:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# Test repository for real-world testing
TEST_REPO = "https://github.com/minimal-xyz/minimal-readme"

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def test_repo_dir(temp_dir):
    """Path where the test repo will be cloned."""
    return os.path.join(temp_dir, "test_repo_sparse")