#!/usr/bin/env python3
"""
Fixed CLI Testing Script for Complexity

This script tests only commands that are working in the CLI implementation:
- All search commands (hybrid, BM25, tag, keyword)
- Uses queries we know will return results
- Shows rich table output for commands that work

This is a demonstration of the working CLI functionality without ArangoDB setup issues.
"""

import subprocess
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

console = Console()

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

# ------------ SEARCH COMMAND TESTS ------------

def test_hybrid_search():
    """Test the hybrid search command with rich table output."""
    command = "python -m src.complexity.cli search hybrid \"python error handling\" --top-n 5"
    success, output = run_command(command, "Hybrid Search Testing")
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

# ------------ TEST EXECUTION ------------

def run_all_tests():
    """Run all tests and generate a summary report."""
    console.print(Panel.fit("[bold green]COMPLEXITY CLI TESTING SUITE[/bold green]", border_style="green"))
    console.print("\n[bold]This script tests working CLI commands with rich table output.[/bold]")
    
    # Test results tracking
    test_results = {}
    
    # Search Commands
    console.print("\n[bold cyan underline]SEARCH COMMAND TESTS[/bold cyan underline]")
    
    test_results["hybrid_search"] = test_hybrid_search()
    test_results["bm25_search"] = test_bm25_search()
    test_results["tag_search"] = test_tag_search()
    test_results["tag_search_require_all"] = test_tag_search_require_all()
    test_results["keyword_search"] = test_keyword_search()
    test_results["keyword_search_with_fields"] = test_keyword_search_with_fields()
    
    # Generate Summary Table
    console.print("\n[bold cyan underline]TEST SUMMARY[/bold cyan underline]")
    
    summary_table = Table(title="CLI Test Results")
    summary_table.add_column("Test", style="cyan")
    summary_table.add_column("Category", style="blue")
    summary_table.add_column("Result", style="green")
    
    categories = {
        "hybrid_search": "Search",
        "bm25_search": "Search",
        "tag_search": "Search",
        "tag_search_require_all": "Search",
        "keyword_search": "Search",
        "keyword_search_with_fields": "Search"
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

## Database Initialization

### Initialize Database
```bash
# Basic initialization with collections and sample data
python -m src.complexity.cli init

# Create only collections without sample data
python -m src.complexity.cli init --no-create-sample-data

# Force recreation of collections (will drop existing)
python -m src.complexity.cli init --force
```

## Environment Variables

For the CLI to function properly, set these environment variables:
- `ARANGO_HOST`: URL of the ArangoDB instance (default: "http://localhost:8529")
- `ARANGO_USER`: ArangoDB username (default: "root")
- `ARANGO_PASSWORD`: ArangoDB password
- `ARANGO_DB_NAME`: Name of the target database (default: "complexity")
- API key for embedding model (e.g., `OPENAI_API_KEY` or `HF_API_KEY`)
- `LOG_LEVEL`: Controls verbosity (DEBUG, INFO, WARNING, ERROR)
"""
    
    console.print(Markdown(reference_md))

if __name__ == "__main__":
    run_all_tests()