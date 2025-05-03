# Complexity CLI Usage Guide

This guide provides instructions for using the Complexity command-line interface (CLI) to search, manage, and visualize your data.

## Getting Started

### Prerequisites

Before using the CLI, ensure you have:

1. ArangoDB running (default: http://localhost:8529)
2. Python environment with required dependencies
3. Environment variables set:
   ```bash
   export ARANGO_HOST="http://localhost:8529"
   export ARANGO_USER="root"
   export ARANGO_PASSWORD="your_password"
   export ARANGO_DB_NAME="complexity"
   ```

### Database Initialization

Before running commands, initialize the database with collections and sample data:

```bash
# Basic initialization
python -m src.complexity.cli init

# Force recreation of collections
python -m src.complexity.cli init --force

# Create collections only (no sample data)
python -m src.complexity.cli init --no-create-sample-data
```

## Search Commands

The CLI provides several search mechanisms with rich table output:

### Hybrid Search

Combines BM25 and semantic search for optimal results:

```bash
python -m src.complexity.cli search hybrid "database search performance optimization" --top-n 5
```

Options:
- `--top-n`, `-n`: Number of results to return (default: 5)
- `--initial-k`, `-k`: Initial candidates to retrieve (default: 20)
- `--bm25-th`: BM25 threshold (default: 0.01)
- `--sim-th`: Similarity threshold (default: 0.70)
- `--tags`, `-t`: Filter by tags (comma-separated)

### BM25 Search

Text-based search using BM25 algorithm:

```bash
python -m src.complexity.cli search bm25 "python database performance" --threshold 0.05 --top-n 5
```

Options:
- `--threshold`, `-th`: Minimum score threshold (default: 0.1)
- `--top-n`, `-n`: Number of results to return (default: 5)
- `--offset`, `-o`: Pagination offset (default: 0)
- `--tags`, `-t`: Filter by tags (comma-separated)

### Tag Search

Search documents by their tag metadata:

```bash
# Find documents with ANY of the tags
python -m src.complexity.cli search tag "python,database" --top-n 5

# Find documents with ALL of the tags
python -m src.complexity.cli search tag "python,database" --require-all --top-n 3
```

Options:
- `--require-all`, `-a`: Require all tags to match (default: false)
- `--top-n`, `-n`: Number of results to return (default: 5)
- `--offset`, `-o`: Pagination offset (default: 0)

### Keyword Search

Fuzzy keyword matching:

```bash
# Basic keyword search
python -m src.complexity.cli search keyword "database performance" --threshold 95.0 --top-n 5

# Search in specific fields
python -m src.complexity.cli search keyword "efficient search" --fields problem,solution --top-n 3
```

Options:
- `--threshold`, `-th`: Similarity threshold (0-100, default: 97.0)
- `--top-n`, `-n`: Number of results to return (default: 5)
- `--fields`, `-f`: Fields to search in (comma-separated)
- `--tags`, `-t`: Filter by tags (comma-separated)

## Database Commands

### Create Document

```bash
# Create from file
python -m src.complexity.cli db create --collection test_docs --data-file /path/to/file.json

# Create from string
python -m src.complexity.cli db create --collection test_docs --data '{"field1": "value1", "field2": 123}'
```

### Read Document

```bash
python -m src.complexity.cli db read document_key --collection test_docs
```

### Update Document

```bash
# Update from file
python -m src.complexity.cli db update document_key --collection test_docs --data-file /path/to/update.json

# Update from string
python -m src.complexity.cli db update document_key --collection test_docs --data '{"field1": "new value"}'
```

### Delete Document

```bash
# With confirmation
python -m src.complexity.cli db delete document_key --collection test_docs

# Without confirmation
python -m src.complexity.cli db delete document_key --collection test_docs --yes
```

## Graph Commands

### Add Edge

```bash
python -m src.complexity.cli graph add-edge from_key to_key --collection test_docs --edge-collection relationships --type RELATED_TO --rationale "Documents about the same topic"
```

### Traverse Graph

```bash
# Basic traversal
python -m src.complexity.cli graph traverse start_key --collection test_docs --graph-name relationships

# Advanced options
python -m src.complexity.cli graph traverse start_key --collection test_docs --graph-name relationships --min-depth 1 --max-depth 3 --direction OUTBOUND
```

### Delete Edge

```bash
# With confirmation
python -m src.complexity.cli graph delete-edge edge_key --edge-collection relationships

# Without confirmation
python -m src.complexity.cli graph delete-edge edge_key --edge-collection relationships --yes
```

## Rich Table Output

All search and graph commands display results in formatted rich tables, showing:

- Result number
- Score/relevance
- Document key
- Content preview
- Tags (when available)

For detailed information about a specific result, the first result is typically expanded with all fields and metadata.

## JSON Output

Add the `--json-output` or `-j` flag to any command to get results in JSON format:

```bash
python -m src.complexity.cli search hybrid "query" --json-output
```

## Testing and Development

For development and testing, use the provided test script:

```bash
# Run all CLI tests
python fixed_test_cli_commands.py
```

## Troubleshooting

If you encounter issues:

1. Ensure ArangoDB is running and accessible
2. Check environment variables are set correctly
3. Run the initialization command to create required collections
4. For import errors, ensure your PYTHONPATH includes the src directory
5. Check logs for detailed error messages

## More Information

For full details on each command and its options, use the built-in help:

```bash
# Get general help
python -m src.complexity.cli --help

# Get help for a specific command group
python -m src.complexity.cli search --help

# Get help for a specific command
python -m src.complexity.cli search hybrid --help
```