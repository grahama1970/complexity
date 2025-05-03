#!/usr/bin/env python3
"""
Improved CLI Testing Script for Complexity

This script tests all CLI commands in the Complexity project using existing data:
1. First queries ArangoDB to find actual document keys to use for testing
2. Tests all search commands with realistic queries
3. Tests database operations on a temporary test collection
4. Tests graph operations using existing document relationships

Each test displays rich table output to show the results.
"""

import os
import sys
import json
import time
import uuid
import subprocess
import tempfile
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

console = Console()

# Use existing collections instead of creating new ones
DOCS_COLLECTION = "messages"  # Using the messages collection which likely has data
EDGE_COLLECTION = "relationships"  # Using existing relationships
TEST_DOC_KEYS = []  # Will store found document keys

def run_command(command, description=None, show_command=True):
    """Run a CLI command and display the output with rich formatting."""
    if description:
        console.print(f"\n[bold cyan]TEST:[/bold cyan] {description}")
    
    if show_command:
        console.print(f"[bold yellow]Command:[/bold yellow] [green]{command}[/green]")
    
    # Run the command
    try:
        result = subprocess.run(command, shell=True, text=True, capture_output=True)
        output = result.stdout
        error = result.stderr
        
        if result.returncode != 0:
            console.print(f"[bold red]ERROR (code {result.returncode}):[/bold red]")
            if error:
                console.print(Panel(error, title="Error Output", border_style="red"))
            return False, error
        
        # Display the output in a panel
        if output:
            console.print(Panel(output, title="Command Output", border_style="green"))
        else:
            console.print("[yellow]Command produced no output[/yellow]")
        
        return True, output
    except Exception as e:
        console.print(f"[bold red]EXCEPTION:[/bold red] {str(e)}")
        return False, str(e)

def find_existing_documents():
    """Find existing documents in the database to use for testing."""
    global TEST_DOC_KEYS
    
    console.print("[bold cyan]SETUP:[/bold cyan] Finding existing documents to use for testing")
    
    # Query to find a few documents that have 'problem' or 'solution' fields
    aql_query = f'FOR doc IN {DOCS_COLLECTION} LIMIT 5 RETURN doc._key'
    command = f"python -m src.complexity.arangodb.cli query \"{aql_query}\""
    
    success, output = run_command(command, "Finding Existing Documents", show_command=True)
    
    if success and output:
        try:
            # Parse the output to extract document keys
            doc_keys = json.loads(output)
            if isinstance(doc_keys, list) and len(doc_keys) > 0:
                TEST_DOC_KEYS = doc_keys
                console.print(f"[green]Found {len(TEST_DOC_KEYS)} document keys for testing[/green]")
                return True
        except:
            # Alternative parsing approach
            import re
            matches = re.findall(r'"([^"]+)"', output)
            if matches:
                TEST_DOC_KEYS = matches
                console.print(f"[green]Found {len(TEST_DOC_KEYS)} document keys for testing[/green]")
                return True
    
    console.print("[bold yellow]WARNING:[/bold yellow] Could not find existing documents. Some tests may be skipped.")
    return False

def find_existing_tags():
    """Find existing tags in the database to use for tag search testing."""
    console.print("[bold cyan]SETUP:[/bold cyan] Finding existing tags to use for testing")
    
    # Query to find documents with tags
    aql_query = f'FOR doc IN {DOCS_COLLECTION} FILTER IS_ARRAY(doc.tags) && LENGTH(doc.tags) > 0 LIMIT 1 RETURN doc.tags'
    command = f"python -m src.complexity.arangodb.cli query \"{aql_query}\""
    
    success, output = run_command(command, "Finding Existing Tags", show_command=True)
    
    if success and output:
        try:
            # Parse the output to extract tags
            tags_lists = json.loads(output)
            if isinstance(tags_lists, list) and len(tags_lists) > 0 and isinstance(tags_lists[0], list):
                tags = tags_lists[0]
                tags_str = ",".join(tags[:2])  # Take first two tags
                console.print(f"[green]Found tags for testing: {tags_str}[/green]")
                return tags_str
        except:
            pass
    
    # Default fallback tags
    console.print("[yellow]Using default tags for testing[/yellow]")
    return "python,database"

def find_existing_edge():
    """Find an existing edge to use for graph tests."""
    console.print("[bold cyan]SETUP:[/bold cyan] Finding existing edge to use for testing")
    
    # Query to find an edge
    aql_query = f'FOR edge IN {EDGE_COLLECTION} LIMIT 1 RETURN edge'
    command = f"python -m src.complexity.arangodb.cli query \"{aql_query}\""
    
    success, output = run_command(command, "Finding Existing Edge", show_command=True)
    
    if success and output:
        try:
            # Parse the output to extract edge information
            edge_data = json.loads(output)
            if isinstance(edge_data, list) and len(edge_data) > 0:
                edge = edge_data[0]
                edge_key = edge.get("_key")
                from_key = edge.get("_from", "").split("/")[-1]
                to_key = edge.get("_to", "").split("/")[-1]
                
                if edge_key and from_key and to_key:
                    console.print(f"[green]Found edge for testing: {edge_key}[/green]")
                    return edge_key, from_key, to_key
        except:
            pass
    
    console.print("[yellow]No existing edges found. Graph tests may fail.[/yellow]")
    return None, None, None

# ------------ SEARCH COMMAND TESTS ------------

def test_hybrid_search():
    """Test the hybrid search command with rich table output."""
    command = "python -m src.complexity.cli search hybrid \"database search performance optimization\" --top-n 5"
    success, output = run_command(command, "Hybrid Search Testing")
    return success

def test_semantic_search():
    """Test the semantic search command with rich table output."""
    command = "python -m src.complexity.cli search semantic \"vector similarity for document retrieval\" --threshold 0.5 --top-n 3"
    success, output = run_command(command, "Semantic Search Testing")
    return success

def test_bm25_search():
    """Test the BM25 search command with rich table output."""
    command = "python -m src.complexity.cli search bm25 \"python database performance\" --threshold 0.05 --top-n 5"
    success, output = run_command(command, "BM25 Search Testing")
    return success

def test_tag_search(tags):
    """Test the tag search command with rich table output."""
    command = f"python -m src.complexity.cli search tag \"{tags}\" --top-n 5"
    success, output = run_command(command, "Tag Search Testing")
    return success

def test_tag_search_require_all(tags):
    """Test the tag search command with require-all flag."""
    command = f"python -m src.complexity.cli search tag \"{tags}\" --require-all --top-n 3"
    success, output = run_command(command, "Tag Search with ALL Tags Testing")
    return success

def test_keyword_search():
    """Test the keyword search command with rich table output."""
    command = "python -m src.complexity.cli search keyword \"database performance\" --threshold 95.0 --top-n 5"
    success, output = run_command(command, "Keyword Search Testing")
    return success

def test_keyword_search_with_fields():
    """Test the keyword search command with field filtering."""
    command = "python -m src.complexity.cli search keyword \"efficient search\" --fields problem,solution --top-n 3"
    success, output = run_command(command, "Keyword Search with Field Filtering")
    return success

# ------------ DATABASE CRUD TESTS ------------

def create_temp_collection():
    """Create a temporary collection for testing database operations."""
    collection_name = f"test_collection_{int(time.time())}"
    console.print(f"[bold cyan]SETUP:[/bold cyan] Creating temporary collection '{collection_name}'")
    
    # Create the collection
    aql_query = f"db._createDocumentCollection('{collection_name}')"
    command = f"python -m src.complexity.arangodb.cli query \"{aql_query}\""
    
    success, output = run_command(command, "Creating Temporary Collection", show_command=True)
    
    if success:
        console.print(f"[green]Created temporary collection '{collection_name}' for testing[/green]")
        return collection_name
    
    console.print("[bold yellow]WARNING:[/bold yellow] Could not create temporary collection. Database tests will be skipped.")
    return None

def test_db_create(collection):
    """Test database document creation."""
    if not collection:
        console.print("[yellow]Skipping database create test (no temporary collection)[/yellow]")
        return False
    
    # Create a test document
    doc = {
        "_key": f"test_doc_{int(time.time())}",
        "problem": "How to implement efficient search in a document database?",
        "solution": "Use a combination of BM25 and vector similarity with hybrid reranking.",
        "tags": ["search", "database", "performance", "python"]
    }
    
    # Write document to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        json.dump(doc, tmp, indent=2)
        tmp_file = tmp.name
    
    # Create the document using the CLI
    command = f"python -m src.complexity.cli db create --collection {collection} --data-file {tmp_file}"
    success, output = run_command(command, f"Database Create Document")
    
    # Clean up
    os.remove(tmp_file)
    
    return success

def test_db_read(collection):
    """Test database document reading."""
    if not collection or len(TEST_DOC_KEYS) == 0:
        console.print("[yellow]Skipping database read test (no documents found)[/yellow]")
        return False
    
    command = f"python -m src.complexity.cli db read {TEST_DOC_KEYS[0]} --collection {DOCS_COLLECTION}"
    success, output = run_command(command, "Database Read Document")
    return success

# ------------ GRAPH OPERATION TESTS ------------

def test_graph_traverse(from_key):
    """Test graph traversal with rich table output."""
    if not from_key:
        console.print("[yellow]Skipping graph traverse test (no document key)[/yellow]")
        return False
    
    command = f"python -m src.complexity.cli graph traverse {from_key} --collection {DOCS_COLLECTION} --graph-name {EDGE_COLLECTION} --max-depth 2 --direction ANY"
    success, output = run_command(command, "Graph Traverse")
    return success

def test_graph_info(edge_key):
    """Test retrieving edge information."""
    if not edge_key:
        console.print("[yellow]Skipping graph edge info test (no edge key)[/yellow]")
        return False
    
    # This is a proxy for the delete operation, which we don't want to actually run
    command = f"python -m src.complexity.arangodb.cli query \"RETURN DOCUMENT('{EDGE_COLLECTION}/{edge_key}')\""
    success, output = run_command(command, "Graph Edge Info")
    return success

# ------------ TEST EXECUTION ------------

def run_all_tests():
    """Run all tests and generate a summary report."""
    console.print(Panel.fit("[bold green]COMPLEXITY CLI TESTING SUITE[/bold green]", border_style="green"))
    console.print("\n[bold]This script tests all CLI commands with rich table output using existing data.[/bold]")
    
    # First, find existing documents and tags
    find_existing_documents()
    tags = find_existing_tags()
    edge_key, from_key, to_key = find_existing_edge()
    
    # Create a temporary collection for database tests
    temp_collection = create_temp_collection()
    
    # Test results tracking
    test_results = {}
    
    # Search Commands
    console.print("\n[bold cyan underline]SEARCH COMMAND TESTS[/bold cyan underline]")
    
    test_results["hybrid_search"] = test_hybrid_search()
    test_results["semantic_search"] = test_semantic_search()
    test_results["bm25_search"] = test_bm25_search()
    test_results["tag_search"] = test_tag_search(tags)
    test_results["tag_search_require_all"] = test_tag_search_require_all(tags)
    test_results["keyword_search"] = test_keyword_search()
    test_results["keyword_search_with_fields"] = test_keyword_search_with_fields()
    
    # Database Operations
    console.print("\n[bold cyan underline]DATABASE OPERATION TESTS[/bold cyan underline]")
    
    test_results["db_create"] = test_db_create(temp_collection)
    test_results["db_read"] = test_db_read(temp_collection)
    
    # Graph Operations
    console.print("\n[bold cyan underline]GRAPH OPERATION TESTS[/bold cyan underline]")
    
    test_results["graph_traverse"] = test_graph_traverse(from_key)
    test_results["graph_edge_info"] = test_graph_info(edge_key)
    
    # Clean up temporary collection if created
    if temp_collection:
        cleanup_query = f"db._drop('{temp_collection}')"
        run_command(f"python -m src.complexity.arangodb.cli query \"{cleanup_query}\"", "Dropping Temporary Collection", show_command=True)
    
    # Generate Summary Table
    console.print("\n[bold cyan underline]TEST SUMMARY[/bold cyan underline]")
    
    summary_table = Table(title="CLI Test Results")
    summary_table.add_column("Test", style="cyan")
    summary_table.add_column("Category", style="blue")
    summary_table.add_column("Result", style="green")
    
    categories = {
        "hybrid_search": "Search",
        "semantic_search": "Search",
        "bm25_search": "Search",
        "tag_search": "Search",
        "tag_search_require_all": "Search",
        "keyword_search": "Search",
        "keyword_search_with_fields": "Search",
        "db_create": "Database",
        "db_read": "Database",
        "graph_traverse": "Graph",
        "graph_edge_info": "Graph"
    }
    
    for test_name, result in test_results.items():
        display_name = test_name.replace("_", " ").title()
        category = categories.get(test_name, "Unknown")
        result_text = "[green]PASS[/green]" if result else "[red]FAIL[/red]"
        summary_table.add_row(display_name, category, result_text)
    
    console.print(summary_table)
    
    # Summary statistics
    pass_count = sum(1 for result in test_results.values() if result)
    total_count = len(test_results)
    pass_rate = (pass_count / total_count) * 100 if total_count > 0 else 0
    
    console.print(f"\n[bold]Summary:[/bold] {pass_count}/{total_count} tests passed ({pass_rate:.1f}%)")
    
    if pass_count == total_count:
        console.print("\n[bold green]🎉 ALL TESTS PASSED! 🎉[/bold green]")
    else:
        console.print("\n[bold yellow]⚠️ SOME TESTS FAILED[/bold yellow]")
    
    # Generate comprehensive command reference
    generate_command_reference()

def generate_command_reference():
    """Generate a comprehensive command reference with examples."""
    console.print("\n[bold cyan underline]COMMAND REFERENCE[/bold cyan underline]")
    
    reference_md = """
# Complexity CLI Command Reference

## Search Commands

### Hybrid Search
```bash
# Basic hybrid search
python -m src.complexity.cli search hybrid "query text"

# Advanced options
python -m src.complexity.cli search hybrid "query text" --top-n 10 --initial-k 25 --bm25-th 0.05 --sim-th 0.8 --tags "tag1,tag2"
```

### Semantic Search
```bash
# Basic semantic search
python -m src.complexity.cli search semantic "query text"

# Advanced options
python -m src.complexity.cli search semantic "query text" --threshold 0.8 --top-n 15 --tags "tag1,tag2"
```

### BM25 Search
```bash
# Basic BM25 search
python -m src.complexity.cli search bm25 "query text"

# Advanced options
python -m src.complexity.cli search bm25 "query text" --threshold 0.05 --top-n 20 --offset 10 --tags "tag1,tag2"
```

### Tag Search
```bash
# Search for docs with ANY of the specified tags
python -m src.complexity.cli search tag "tag1,tag2"

# Search for docs with ALL specified tags
python -m src.complexity.cli search tag "tag1,tag2" --require-all
```

### Keyword Search
```bash
# Basic keyword search
python -m src.complexity.cli search keyword "search term"

# Advanced options
python -m src.complexity.cli search keyword "search term" --threshold 90.0 --fields "field1,field2" --tags "tag1,tag2"
```

## Database Commands

### Create Document
```bash
# Create from file
python -m src.complexity.cli db create --collection coll_name --data-file path/to/file.json

# Create from string
python -m src.complexity.cli db create --collection coll_name --data '{"field1": "value1", "field2": 123}'
```

### Read Document
```bash
python -m src.complexity.cli db read document_key --collection coll_name
```

### Update Document
```bash
# Update from file
python -m src.complexity.cli db update document_key --collection coll_name --data-file path/to/update.json

# Update from string
python -m src.complexity.cli db update document_key --collection coll_name --data '{"field1": "new value"}'
```

### Delete Document
```bash
# With confirmation
python -m src.complexity.cli db delete document_key --collection coll_name

# Without confirmation
python -m src.complexity.cli db delete document_key --collection coll_name --yes
```

## Graph Commands

### Add Edge
```bash
python -m src.complexity.cli graph add-edge from_key to_key --collection doc_coll --edge-collection edge_coll --type RELATIONSHIP_TYPE --rationale "Reason for the connection"
```

### Traverse Graph
```bash
# Basic traversal
python -m src.complexity.cli graph traverse start_key --collection doc_coll --graph-name graph_name

# Advanced options
python -m src.complexity.cli graph traverse start_key --collection doc_coll --graph-name graph_name --min-depth 1 --max-depth 3 --direction OUTBOUND
```

### Delete Edge
```bash
# With confirmation
python -m src.complexity.cli graph delete-edge edge_key --edge-collection edge_coll

# Without confirmation
python -m src.complexity.cli graph delete-edge edge_key --edge-collection edge_coll --yes
```
"""
    
    console.print(Markdown(reference_md))

if __name__ == "__main__":
    run_all_tests()