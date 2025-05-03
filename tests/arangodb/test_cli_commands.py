#!/usr/bin/env python3
"""
Comprehensive CLI Testing Script for Complexity

This script tests all CLI commands in the Complexity project, demonstrating:
1. All search commands (hybrid, semantic, BM25, tag, keyword)
2. Database CRUD operations (create, read, update, delete)
3. Graph operations (add edge, traverse, delete edge)

Each test displays rich table output to show the results.
"""

import os
import sys
import json
import time
import uuid
import subprocess
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

console = Console()

# Global variables
TEST_COLLECTION = "test_docs"
TEST_EDGE_COLLECTION = "test_relationships"
TEST_DOCUMENT_KEYS = []  # Will store document keys for later operations

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

def create_test_document(key_suffix="", with_tags=True):
    """Create a test document with optional tags."""
    # Generate a unique key with timestamp
    if not key_suffix:
        key_suffix = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
    
    doc_key = f"test_doc_{key_suffix}"
    
    # Create document content
    doc = {
        "_key": doc_key,
        "problem": "How to implement efficient search in a document database?",
        "solution": "Use a combination of BM25 and vector similarity with hybrid reranking.",
        "context": "Database performance optimization for search operations.",
        "details": "When dealing with large document collections, efficient search requires both lexical and semantic matching. BM25 handles keyword matching while vector embeddings capture semantic similarity."
    }
    
    # Add tags if requested
    if with_tags:
        doc["tags"] = ["search", "database", "performance", "python"]
    
    return doc_key, doc

# ------------ SEARCH COMMAND TESTS ------------

def test_hybrid_search():
    """Test the hybrid search command with rich table output."""
    command = "python -m src.complexity.cli search hybrid \"database search performance optimization\" --top-n 5"
    success, output = run_command(command, "Hybrid Search Testing")
    return success

def test_semantic_search():
    """Test the semantic search command with rich table output."""
    command = "python -m src.complexity.cli search semantic \"vector similarity for document retrieval\" --threshold 0.7 --top-n 3"
    success, output = run_command(command, "Semantic Search Testing")
    return success

def test_bm25_search():
    """Test the BM25 search command with rich table output."""
    command = "python -m src.complexity.cli search bm25 \"python database performance\" --threshold 0.05 --top-n 5"
    success, output = run_command(command, "BM25 Search Testing")
    return success

def test_tag_search():
    """Test the tag search command with rich table output."""
    command = "python -m src.complexity.cli search tag \"python,database\" --top-n 5"
    success, output = run_command(command, "Tag Search Testing")
    return success

def test_tag_search_require_all():
    """Test the tag search command with require-all flag."""
    command = "python -m src.complexity.cli search tag \"python,database\" --require-all --top-n 3"
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

def test_db_create():
    """Test database document creation."""
    global TEST_DOCUMENT_KEYS
    
    # Create two test documents for later operations
    for i in range(2):
        doc_key, doc = create_test_document(f"{i+1}")
        TEST_DOCUMENT_KEYS.append(doc_key)
        
        # Write document to a temporary file
        tmp_file = f"/tmp/test_doc_{i+1}.json"
        with open(tmp_file, "w") as f:
            json.dump(doc, f, indent=2)
        
        # Create the document using the CLI
        command = f"python -m src.complexity.cli db create --collection {TEST_COLLECTION} --data-file {tmp_file}"
        success, output = run_command(command, f"Database Create Document {i+1}")
        
        # Clean up
        os.remove(tmp_file)
        
        if not success:
            return False
    
    return True

def test_db_read():
    """Test database document reading."""
    if not TEST_DOCUMENT_KEYS:
        console.print("[bold red]ERROR:[/bold red] No test documents created yet.")
        return False
    
    command = f"python -m src.complexity.cli db read {TEST_DOCUMENT_KEYS[0]} --collection {TEST_COLLECTION}"
    success, output = run_command(command, "Database Read Document")
    return success

def test_db_update():
    """Test database document updating with before/after comparison."""
    if not TEST_DOCUMENT_KEYS:
        console.print("[bold red]ERROR:[/bold red] No test documents created yet.")
        return False
    
    # First read the current document
    command = f"python -m src.complexity.cli db read {TEST_DOCUMENT_KEYS[0]} --collection {TEST_COLLECTION}"
    success, before_output = run_command(command, "Database Read Before Update")
    if not success:
        return False
    
    # Create an update with new information
    update_data = {
        "solution": "UPDATED: Use hybrid search with custom weights for BM25 and semantic components.",
        "updated_at": int(time.time()),
        "updated_by": "CLI Test Script"
    }
    
    # Write update to a temporary file
    tmp_file = "/tmp/update_data.json"
    with open(tmp_file, "w") as f:
        json.dump(update_data, f, indent=2)
    
    # Update the document
    command = f"python -m src.complexity.cli db update {TEST_DOCUMENT_KEYS[0]} --collection {TEST_COLLECTION} --data-file {tmp_file}"
    success, output = run_command(command, "Database Update Document")
    
    # Clean up
    os.remove(tmp_file)
    
    if not success:
        return False
    
    # Read the updated document
    command = f"python -m src.complexity.cli db read {TEST_DOCUMENT_KEYS[0]} --collection {TEST_COLLECTION}"
    success, after_output = run_command(command, "Database Read After Update")
    return success

def test_db_delete():
    """Test database document deletion with confirmation."""
    if len(TEST_DOCUMENT_KEYS) < 2:
        console.print("[bold red]ERROR:[/bold red] Not enough test documents created.")
        return False
    
    # Use the second document for deletion to keep one for graph operations
    doc_key = TEST_DOCUMENT_KEYS[1]
    
    # Delete with --yes flag to bypass confirmation
    command = f"python -m src.complexity.cli db delete {doc_key} --collection {TEST_COLLECTION} --yes"
    success, output = run_command(command, "Database Delete Document")
    
    # Verify deletion by attempting to read
    command = f"python -m src.complexity.cli db read {doc_key} --collection {TEST_COLLECTION}"
    read_success, read_output = run_command(command, "Verify Document Deletion")
    
    # We expect the read to fail, showing the document is gone
    return success and not read_success

# ------------ GRAPH OPERATION TESTS ------------

def test_graph_add_edge():
    """Test adding an edge between documents."""
    global TEST_DOCUMENT_KEYS
    
    if len(TEST_DOCUMENT_KEYS) < 2:
        # Create another document if needed
        doc_key, doc = create_test_document("extra")
        TEST_DOCUMENT_KEYS.append(doc_key)
        
        # Write document to a temporary file
        tmp_file = "/tmp/extra_doc.json"
        with open(tmp_file, "w") as f:
            json.dump(doc, f, indent=2)
        
        # Create the document using the CLI
        command = f"python -m src.complexity.cli db create --collection {TEST_COLLECTION} --data-file {tmp_file}"
        success, output = run_command(command, "Create Extra Document for Graph")
        
        # Clean up
        os.remove(tmp_file)
        
        if not success:
            return False
    
    # Add edge between documents
    from_key = TEST_DOCUMENT_KEYS[0]
    to_key = TEST_DOCUMENT_KEYS[-1]  # Last document
    
    command = f"python -m src.complexity.cli graph add-edge {from_key} {to_key} --collection {TEST_COLLECTION} --edge-collection {TEST_EDGE_COLLECTION} --type RELATED_TO --rationale \"Test relationship between documents\""
    success, output = run_command(command, "Graph Add Edge")
    return success

def test_graph_traverse():
    """Test graph traversal with rich table output."""
    if not TEST_DOCUMENT_KEYS:
        console.print("[bold red]ERROR:[/bold red] No test documents created yet.")
        return False
    
    # Traverse from first document
    command = f"python -m src.complexity.cli graph traverse {TEST_DOCUMENT_KEYS[0]} --collection {TEST_COLLECTION} --graph-name {TEST_EDGE_COLLECTION} --max-depth 2 --direction ANY"
    success, output = run_command(command, "Graph Traverse")
    return success

def test_graph_delete_edge():
    """Test edge deletion with confirmation."""
    # First list edges to get the edge key
    # This is a bit tricky since we need to extract the edge key from previous outputs
    # For demonstration, we'll use a direct AQL query via the CLI to list edges
    command = f"python -m src.complexity.arangodb.cli query \"FOR e IN {TEST_EDGE_COLLECTION} FILTER e._from LIKE '{TEST_COLLECTION}/%' RETURN e\""
    success, output = run_command(command, "List Edges to Get Edge Key")
    
    if not success or "\"_key\":" not in output:
        console.print("[bold yellow]WARNING:[/bold yellow] Could not find edge key for deletion test.")
        return False
    
    # Extract edge key from output - this is a simplistic approach
    try:
        edge_data = json.loads(output)
        if isinstance(edge_data, list) and len(edge_data) > 0:
            edge_key = edge_data[0]["_key"]
        else:
            # Alternative parsing approach
            import re
            match = re.search(r'"_key":\s*"([^"]+)"', output)
            if match:
                edge_key = match.group(1)
            else:
                console.print("[bold yellow]WARNING:[/bold yellow] Could not parse edge key from output.")
                return False
    except:
        console.print("[bold yellow]WARNING:[/bold yellow] Could not parse edge key from output.")
        return False
    
    # Delete the edge
    command = f"python -m src.complexity.cli graph delete-edge {edge_key} --edge-collection {TEST_EDGE_COLLECTION} --yes"
    success, output = run_command(command, "Graph Delete Edge")
    return success

# ------------ TEST EXECUTION ------------

def run_all_tests():
    """Run all tests and generate a summary report."""
    console.print(Panel.fit("[bold green]COMPLEXITY CLI TESTING SUITE[/bold green]", border_style="green"))
    console.print("\n[bold]This script tests all CLI commands with rich table output.[/bold]")
    
    # Test results tracking
    test_results = {}
    
    # Search Commands
    console.print("\n[bold cyan underline]SEARCH COMMAND TESTS[/bold cyan underline]")
    
    test_results["hybrid_search"] = test_hybrid_search()
    test_results["semantic_search"] = test_semantic_search()
    test_results["bm25_search"] = test_bm25_search()
    test_results["tag_search"] = test_tag_search()
    test_results["tag_search_require_all"] = test_tag_search_require_all()
    test_results["keyword_search"] = test_keyword_search()
    test_results["keyword_search_with_fields"] = test_keyword_search_with_fields()
    
    # Database CRUD Operations
    console.print("\n[bold cyan underline]DATABASE OPERATION TESTS[/bold cyan underline]")
    
    test_results["db_create"] = test_db_create()
    test_results["db_read"] = test_db_read()
    test_results["db_update"] = test_db_update()
    test_results["db_delete"] = test_db_delete()
    
    # Graph Operations
    console.print("\n[bold cyan underline]GRAPH OPERATION TESTS[/bold cyan underline]")
    
    test_results["graph_add_edge"] = test_graph_add_edge()
    test_results["graph_traverse"] = test_graph_traverse()
    test_results["graph_delete_edge"] = test_graph_delete_edge()
    
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
        "db_update": "Database",
        "db_delete": "Database",
        "graph_add_edge": "Graph",
        "graph_traverse": "Graph",
        "graph_delete_edge": "Graph"
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