# Database Operations and Search API Test Plan

## Overview

This test plan outlines the approach for comprehensively testing the database operations and search functionality in the Complexity project. These tests will verify that all components work correctly with real database interactions (no mocking).

## Test Components

### 1. Basic Database Operations (`db_operations.py`)

- **Create Document Test**
  - Create documents with various structures
  - Verify document creation with and without explicit keys
  - Verify timestamp generation

- **Read Document Test**
  - Retrieve documents by key
  - Handle retrieving non-existent documents

- **Update Document Test**
  - Update existing documents with new fields
  - Modify existing fields in documents
  - Verify updated timestamps

- **Delete Document Test**
  - Delete documents by key
  - Verify successful deletion
  - Test deleting non-existent documents

- **Query Document Test**
  - Test filtering documents by various criteria
  - Test sorting documents
  - Test pagination with limit and offset

### 2. Embedding Operations (`embedded_db_operations.py`)

- **Document Creation with Embedding**
  - Create documents with content that should trigger embedding generation
  - Verify embedding field is added to document
  - Check embedding dimensions and structure

- **Document Update with Embedding**
  - Update document content and verify embedding regeneration
  - Update non-content fields and verify embedding remains unchanged
  - Test edge cases (empty content, non-string content)

### 3. Graph Operations

- **Original API (`db_operations.py`)**
  - Test `create_relationship` between documents
  - Test `delete_relationship_by_key`
  - Verify relationship metadata

- **Enhanced API (`enhanced_relationships.py`)**
  - Test `create_edge_from_cli` with CLI-compatible parameters
  - Test `delete_edge_from_cli` with CLI-compatible parameters
  - Verify these operations correctly bridge the parameter mismatch

### 4. Search API

- **BM25 Search**
  - Test keyword-based search functionality
  - Verify scoring and ranking
  - Test with different thresholds

- **Semantic Search**
  - Generate embeddings for queries
  - Test similarity-based search
  - Verify vector similarity ranking

- **Hybrid Search**
  - Test combined BM25 and semantic search
  - Verify reciprocal rank fusion scoring
  - Test with different weights and thresholds

- **Graph Traversal**
  - Create document relationships
  - Test traversal with different depths
  - Test directional traversal (inbound, outbound, any)

## Test Approach

1. **Setup Phase**
   - Connect to ArangoDB
   - Create test collections
   - Prepare test data

2. **Execution Phase**
   - Run each test independently
   - Capture detailed logs for each operation
   - Verify results against expected outcomes

3. **Cleanup Phase**
   - Remove all test data and collections
   - Restore initial database state

## Testing Environment

- Use dedicated test collections (`test_docs`, `test_relationships`)
- Execute against a real ArangoDB instance
- Configure with environment variables for flexibility

## Success Criteria

- All CRUD operations complete successfully
- Embedding generation works for document creation and updates
- Graph operations correctly handle relationships
- Search operations return expected results
- All tests pass without any mock implementations

## Implementation

The test plan is implemented in `test_db_and_search.py`, which provides:
- Comprehensive tests for all components
- Detailed logging for troubleshooting
- Command-line arguments for test configuration
- Automated cleanup of test data