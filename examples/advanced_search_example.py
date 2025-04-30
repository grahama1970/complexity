#!/usr/bin/env python3
"""
Advanced Semantic Search Example

This script demonstrates the optimized approach for semantic search:
- Direct ArangoDB search for simple queries (milliseconds)
- Two-stage ArangoDB for filtered queries (milliseconds)
- PyTorch for nesting/relationship building queries (seconds)

Examples:
    # Simple search - direct ArangoDB (milliseconds)
    python hybrid_semantic_search_ex.py "primary color"
    
    # Filtered search - two-stage ArangoDB (milliseconds)
    python hybrid_semantic_search_ex.py "python function" --tags python,function
    
    # Nesting/relationship building - PyTorch (seconds)
    python hybrid_semantic_search_ex.py "machine learning" --force-pytorch
    
    # Debug mode - see detailed execution steps
    python hybrid_semantic_search_ex.py "python function" --tags python,function --debug
"""

import sys
import time
from typing import List, Optional
import rich
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
import typer
from loguru import logger

try:
    from complexity.arangodb.arango_setup_unknown import connect_arango, ensure_database
    from complexity.arangodb.search_api.hybrid_semantic import semantic_search
    from complexity.arangodb.config import (
        COLLECTION_NAME,
        EMBEDDING_FIELD,
        EMBEDDING_MODEL,
        EMBEDDING_DIMENSIONS
    )
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure the complexity package is installed or in your PYTHONPATH")
    sys.exit(1)

# Create console for rich output
console = Console()

# Create Typer app
app = typer.Typer(
    help="Run semantic search with advanced filtering capabilities",
    add_completion=False,
)

@app.command()
def search(
    query: str = typer.Argument(
        ..., 
        help="Search query text"
    ),
    filter: Optional[str] = typer.Option(
        None, 
        "--filter", "-f", 
        help="AQL filter expression (e.g. 'doc.difficulty <= 3')"
    ),
    category: Optional[str] = typer.Option(
        None, 
        "--category", "-c",
        help="Filter by document category (creates AQL filter)"
    ),
    tags: Optional[List[str]] = typer.Option(
        None, 
        "--tags", "-t",
        help="Filter by tags (comma-separated with no spaces, e.g. python,error-handling)"
    ),
    min_score: float = typer.Option(
        0.7, 
        "--min-score", "-s",
        help="Minimum similarity score threshold (0.0-1.0)"
    ),
    top_n: int = typer.Option(
        10, 
        "--top-n", "-n",
        help="Maximum number of results to return"
    ),
    force_pytorch: bool = typer.Option(
        False, 
        "--force-pytorch",
        help="Force using PyTorch (required for relationship building)"
    ),
    show_all_approaches: bool = typer.Option(
        False,
        "--compare",
        help="Compare all three approaches (direct, two-stage, PyTorch)"
    ),
    debug: bool = typer.Option(
        False, 
        "--debug", "-d",
        help="Enable debug logging"
    ),
):
    """
    Perform semantic search using the optimal approach based on query type.
    
    - Simple search: Direct ArangoDB (milliseconds)
    - Filtered search: Two-stage ArangoDB (milliseconds)
    - Relationship building: PyTorch (seconds, use --force-pytorch)
    """
    # Set up logging
    log_level = "DEBUG" if debug else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    
    # Display configuration
    logger.debug(f"Using collection: {COLLECTION_NAME}")
    logger.debug(f"Embedding model: {EMBEDDING_MODEL} ({EMBEDDING_DIMENSIONS} dimensions)")
    logger.debug(f"Embedding field: {EMBEDDING_FIELD}")
    
    # Build the combined filter expression
    combined_filter = []
    
    # Add explicit filter if provided
    if filter:
        combined_filter.append(f"({filter})")
    
    # Add category filter if provided (convenience for common filtering)
    if category:
        combined_filter.append(f'doc.category == "{category}"')
    
    # Get the final filter expression
    filter_expr = " AND ".join(combined_filter) if combined_filter else None
    
    # Determine search approach
    if force_pytorch:
        approach = "PyTorch (for relationship building)"
    elif filter_expr or tags:
        approach = "Two-stage ArangoDB (for filtering)"
    else:
        approach = "Direct ArangoDB (simple search)"
    
    # Connect to ArangoDB
    try:
        with Progress() as progress:
            task = progress.add_task("[cyan]Connecting to ArangoDB...", total=1)
            client = connect_arango()
            db = ensure_database(client)
            progress.update(task, advance=1)
        
        console.print(f"[green]Connected to database:[/green] {db.name}")
    except Exception as e:
        console.print(f"[bold red]Failed to connect to ArangoDB:[/bold red] {e}")
        return 1
    
    # Compare all approaches if requested
    if show_all_approaches:
        console.print(Panel(f"[bold]Comparing all semantic search approaches for query:[/bold] '{query}'"))
        
        results_comparison = []
        
        # 1. Test Direct ArangoDB
        console.print("\n[bold cyan]Testing Direct ArangoDB approach...[/bold cyan]")
        start_time = time.time()
        direct_results = semantic_search(
            db=db,
            query=query,
            collections=[COLLECTION_NAME],
            min_score=min_score,
            top_n=top_n,
            force_direct=True  # Force direct approach regardless of filter
        )
        direct_time = (time.time() - start_time) * 1000
        results_comparison.append({
            "approach": "Direct ArangoDB",
            "engine": direct_results.get("search_engine", "unknown"),
            "count": len(direct_results.get("results", [])),
            "time_ms": direct_time,
            "results": direct_results.get("results", [])
        })
        console.print(f"[green]Direct ArangoDB:[/green] Found {results_comparison[0]['count']} results in {direct_time:.2f} ms")
        
        # 2. Test Two-stage ArangoDB
        if filter_expr or tags:
            console.print("\n[bold cyan]Testing Two-stage ArangoDB approach...[/bold cyan]")
            start_time = time.time()
            twostage_results = semantic_search(
                db=db,
                query=query,
                collections=[COLLECTION_NAME],
                filter_expr=filter_expr,
                min_score=min_score,
                top_n=top_n,
                tag_list=tags,
                force_twostage=True  # Force two-stage approach
            )
            twostage_time = (time.time() - start_time) * 1000
            results_comparison.append({
                "approach": "Two-stage ArangoDB",
                "engine": twostage_results.get("search_engine", "unknown"),
                "count": len(twostage_results.get("results", [])),
                "time_ms": twostage_time,
                "results": twostage_results.get("results", [])
            })
            console.print(f"[green]Two-stage ArangoDB:[/green] Found {results_comparison[1]['count']} results in {twostage_time:.2f} ms")
        
        # 3. Test PyTorch
        console.print("\n[bold cyan]Testing PyTorch approach...[/bold cyan]")
        start_time = time.time()
        pytorch_results = semantic_search(
            db=db,
            query=query,
            collections=[COLLECTION_NAME],
            filter_expr=filter_expr,
            min_score=min_score,
            top_n=top_n,
            tag_list=tags,
            force_pytorch=True
        )
        pytorch_time = (time.time() - start_time) * 1000
        results_comparison.append({
            "approach": "PyTorch",
            "engine": pytorch_results.get("search_engine", "unknown"),
            "count": len(pytorch_results.get("results", [])),
            "time_ms": pytorch_time,
            "results": pytorch_results.get("results", [])
        })
        console.print(f"[green]PyTorch:[/green] Found {results_comparison[2]['count']} results in {pytorch_time:.2f} ms")
        
        # Display comparison table
        compare_table = Table(title="Semantic Search Approach Comparison")
        compare_table.add_column("Approach", style="cyan")
        compare_table.add_column("Results", style="green")
        compare_table.add_column("Time (ms)", style="yellow")
        
        for result in results_comparison:
            color = "green" if result["time_ms"] < 500 else "yellow" if result["time_ms"] < 5000 else "red"
            compare_table.add_row(
                result["approach"],
                str(result["count"]),
                f"[{color}]{result['time_ms']:.2f}[/{color}]"
            )
        
        console.print(compare_table)
        
        # Show detailed results from the fastest approach
        fastest = min(results_comparison, key=lambda x: x["time_ms"])
        console.print(f"\n[bold]Showing results from fastest approach:[/bold] {fastest['approach']} ({fastest['time_ms']:.2f} ms)")
        
        # Display results
        display_results(fastest["results"], query)
        return 0
    
    # Run semantic search with selected approach
    try:
        console.print(Panel(f"[bold]Semantic Search[/bold]: '{query}'", 
                       subtitle=f"Using {approach}"))
        
        # Show filter information
        if filter_expr or tags or force_pytorch:
            filters_table = Table(title="Search Parameters")
            filters_table.add_column("Parameter", style="cyan")
            filters_table.add_column("Value", style="green")
            
            if filter_expr:
                filters_table.add_row("AQL Expression", filter_expr)
            if tags:
                filters_table.add_row("Tags", ", ".join(tags))
            if force_pytorch:
                filters_table.add_row("Search Approach", "PyTorch (for relationship building)")
            elif filter_expr or tags:
                filters_table.add_row("Search Approach", "Two-stage ArangoDB (for filtering)")
            else:
                filters_table.add_row("Search Approach", "Direct ArangoDB (simple search)")
            
            filters_table.add_row("Similarity Threshold", str(min_score))
            
            console.print(filters_table)
        
        console.print(f"\n[bold cyan]Searching...[/bold cyan]")
        
        # Show appropriate message based on search approach
        if force_pytorch:
            console.print("[yellow]Using PyTorch for nesting/relationship building (slower but more powerful)[/yellow]")
        elif filter_expr or tags:
            console.print("[cyan]Using two-stage ArangoDB approach for filtering[/cyan]")
            console.print("[cyan]  - Stage 1: Get document IDs matching filter criteria[/cyan]")
            console.print("[cyan]  - Stage 2: Perform vector search only on filtered documents[/cyan]")
        else:
            console.print("[green]Using direct ArangoDB approach for simple search[/green]")
        
        # Perform search
        start_time = time.time()
        
        # Updated semantic_search call
        search_results = semantic_search(
            db=db,
            query=query,
            collections=[COLLECTION_NAME],
            filter_expr=filter_expr,
            min_score=min_score,
            top_n=top_n,
            tag_list=tags,
            force_pytorch=force_pytorch
        )
        
        search_time = (time.time() - start_time) * 1000  # ms
        
        # Print search summary
        engine = search_results.get("search_engine", "unknown")
        count = len(search_results.get("results", []))
        total = search_results.get("total", 0)
        reported_time = search_results.get("time", 0) * 1000  # ms
        
        # Create results header with performance highlight based on engine
        if "arangodb" in engine:
            speed_msg = f"[green]{reported_time:.2f} ms[/green]"
            engine_display = f"[cyan]{engine}[/cyan]"
        elif engine == "pytorch":
            speed_msg = f"[yellow]{reported_time:.2f} ms[/yellow]"
            engine_display = f"[yellow]{engine}[/yellow]"
        else:
            speed_msg = f"[blue]{reported_time:.2f} ms[/blue]"
            engine_display = f"[blue]{engine}[/blue]"
        
        console.print(
            f"\n[bold green]Found {count} of {total} total matches[/bold green] "
            f"using {engine_display} in {speed_msg}"
        )
        
        # Display results
        display_results(search_results.get("results", []), query)
        
        return 0
    
    except Exception as e:
        console.print(f"[bold red]Error running semantic search:[/bold red] {e}")
        if debug:
            logger.exception("Detailed error information:")
        return 1

def display_results(results, query):
    """Display search results in a formatted table"""
    if results:
        results_table = Table(title=f"Search Results for '{query}'")
        results_table.add_column("#", style="dim")
        results_table.add_column("Score", style="cyan")
        results_table.add_column("ID", style="green")
        results_table.add_column("Question", style="bold")
        results_table.add_column("Tags", style="magenta")
        
        for i, result in enumerate(results):
            doc = result.get("doc", {})
            score = result.get("similarity_score", 0)
            
            # Get document identifier
            doc_id = doc.get("_key", "")
            
            # Get question or title
            question = doc.get("question", doc.get("title", ""))
            if not question and "answer" in doc:
                # If no question/title, use first 50 chars of answer
                question = doc["answer"][:50] + "..." if len(doc["answer"]) > 50 else doc["answer"]
            
            # Get tags
            tags_str = ", ".join(doc.get("tags", [])) if isinstance(doc.get("tags"), list) else ""
            
            # Add row to results table
            results_table.add_row(
                str(i+1),
                f"{score:.4f}",
                doc_id,
                question,
                tags_str
            )
        
        console.print(results_table)
    else:
        console.print("\n[yellow]No results found matching your criteria.[/yellow]")
        console.print("Try lowering the similarity threshold with --min-score or adjusting your filters.")

if __name__ == "__main__":
    app()