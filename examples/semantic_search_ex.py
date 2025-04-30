#!/usr/bin/env python3
"""
Advanced Semantic Search Example

This script demonstrates the optimized approach for semantic search:
- Direct ArangoDB search for simple queries (milliseconds)
- Two-stage ArangoDB for filtered queries (milliseconds)
- PyTorch for nesting/relationship building queries (seconds)

Examples:
    # Simple search - direct ArangoDB (milliseconds)
    python hybrid_semantic_search_example.py "primary color"
    
    # Filtered search - two-stage ArangoDB (milliseconds)
    python hybrid_semantic_search_example.py "python function" --tags python,function
    
    # Nesting/relationship building - PyTorch (seconds)
    python hybrid_semantic_search_example.py "machine learning" --force-pytorch
    
    # Debug mode
    python hybrid_semantic_search_example.py "python function" --tags python,function --debug
    
    # Debug mode with breakpoints (for VSCode debugging)
    python hybrid_semantic_search_example.py --debug-mode
"""

import sys
import time
import os
from typing import List, Optional
import rich
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
import typer
from loguru import logger

# Check for debug mode flag
DEBUG_MODE = os.environ.get("DEBUG_MODE", "false").lower() == "true"

try:
    from complexity.arangodb.arango_setup_unknown import connect_arango, ensure_database
    from complexity.arangodb.search_api.semantic_search import semantic_search
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

# Separate function for running the search - makes debugging easier
def run_search(
    query: str,
    filter: Optional[str] = None,
    category: Optional[str] = None,
    tags: Optional[List[str]] = None,
    min_score: float = 0.7,
    top_n: int = 10,
    force_pytorch: bool = False,
    debug: bool = False,
    show_aql: bool = False
):
    """
    Core search function that can be called directly for debugging.
    
    Args:
        query: Search query text
        filter: Optional AQL filter expression
        category: Optional document category for filtering
        tags: Optional list of tags to filter by
        min_score: Minimum similarity threshold
        top_n: Maximum number of results
        force_pytorch: Whether to force using PyTorch
        debug: Enable debug logging
        show_aql: Print AQL queries for debugging
    
    Returns:
        Search results dictionary
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
    
    # Show search parameters
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
        return {"error": str(e)}
    
    # Run semantic search
    try:
        console.print(f"\n[bold cyan]Searching...[/bold cyan]")
        
        # Show appropriate message based on search approach
        if force_pytorch:
            console.print("[yellow]Using PyTorch for nesting/relationship building (slower but more powerful)[/yellow]")
        elif filter_expr or tags:
            console.print("[cyan]Using two-stage ArangoDB approach for filtering[/cyan]")
            if show_aql:
                console.print("\n[bold cyan]Debug: Two-Stage ArangoDB Process[/bold cyan]")
                console.print("[cyan]Stage 1: Get filtered document IDs with query:[/cyan]")
                console.print(f"FOR doc IN {COLLECTION_NAME}")
                if filter_expr:
                    console.print(f"  FILTER {filter_expr}")
                if tags:
                    tag_filter = " AND ".join([f'"{tag}" IN doc.tags' for tag in tags])
                    console.print(f"  FILTER {tag_filter}")
                console.print("  RETURN doc._id")
                
                console.print("\n[cyan]Stage 2: Perform vector search on filtered documents:[/cyan]")
                console.print(f"FOR doc IN {COLLECTION_NAME}")
                console.print("  FILTER doc._id IN @filtered_ids")
                console.print(f"  LET score = APPROX_NEAR_COSINE(doc.{EMBEDDING_FIELD}, @query_embedding)")
                console.print("  SORT score DESC")
                console.print(f"  LIMIT {top_n}")
                console.print("  RETURN { doc, similarity_score: score }")
        else:
            console.print("[green]Using direct ArangoDB approach for simple search[/green]")
            if show_aql:
                console.print("\n[bold green]Debug: Direct ArangoDB Process[/bold green]")
                console.print("[green]Single query with no filtering:[/green]")
                console.print(f"FOR doc IN {COLLECTION_NAME}")
                console.print(f"  LET score = APPROX_NEAR_COSINE(doc.{EMBEDDING_FIELD}, @query_embedding)")
                # Note: We don't filter on score in AQL to avoid potential issues
                console.print("  SORT score DESC")
                console.print(f"  LIMIT {top_n}")
                console.print("  RETURN { doc, similarity_score: score }")
        
        # Perform search
        start_time = time.time()
        
        # If we're in debug mode with breakpoints, add a breakpoint here
        if DEBUG_MODE:
            breakpoint()  # This will pause execution for debugging
        
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
        
        # Return search results for further processing or display
        return search_results
    
    except Exception as e:
        console.print(f"[bold red]Error running semantic search:[/bold red] {e}")
        if debug:
            logger.exception("Detailed error information:")
        return {"error": str(e)}

def display_results(search_results, query):
    """
    Display search results in a formatted table.
    
    Args:
        search_results: Results from semantic search
        query: Original search query
    """
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
    
    # Print results
    if count > 0:
        results_table = Table(title=f"Search Results for '{query}'")
        results_table.add_column("#", style="dim")
        results_table.add_column("Score", style="cyan")
        results_table.add_column("ID", style="green")
        results_table.add_column("Question", style="bold")
        results_table.add_column("Tags", style="magenta")
        
        for i, result in enumerate(search_results.get("results", [])):
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
        if search_results.get("min_score", 0) > 0.5:
            console.print(f"Suggestion: Try --min-score {search_results.get('min_score', 0.7) - 0.2}")

@app.command()
def search(
    query: str = typer.Argument(
        None,  # Allow None for debug mode
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
    debug: bool = typer.Option(
        False, 
        "--debug", "-d",
        help="Enable debug logging"
    ),
    show_aql: bool = typer.Option(
        False,
        "--show-aql",
        help="Show AQL queries that would be executed"
    ),
    debug_mode: bool = typer.Option(
        False,
        "--debug-mode",
        help="Run with debugger breakpoints (for VSCode debugging)"
    ),
):
    """
    Perform semantic search using the optimal approach based on query type.
    
    - Simple search: Direct ArangoDB (milliseconds)
    - Filtered search: Two-stage ArangoDB (milliseconds)
    - Relationship building: PyTorch (seconds, use --force-pytorch)
    """
    # Set DEBUG_MODE environment variable if debug_mode is True
    if debug_mode:
        os.environ["DEBUG_MODE"] = "true"
        
        # Use default debug parameters if query is not provided
        if query is None:
            query = "python function"
            tags = ["python", "function"]
            debug = True
            console.print("[bold cyan]Running in DEBUG MODE with default parameters[/bold cyan]")
            console.print(f"Query: '{query}', Tags: {tags}")
    
    # Ensure we have a query
    if query is None:
        console.print("[bold red]Error: Query is required unless --debug-mode is used[/bold red]")
        return 1
    
    # Run the search
    search_results = run_search(
        query=query,
        filter=filter,
        category=category,
        tags=tags,
        min_score=min_score,
        top_n=top_n,
        force_pytorch=force_pytorch,
        debug=debug,
        show_aql=show_aql
    )
    
    # Check for errors
    if "error" in search_results:
        return 1
    
    # Display results
    display_results(search_results, query)
    return 0

if __name__ == "__main__":
    # For CLI usage
    if len(sys.argv) > 1:
        app()
    # Debug mode - runs with preset values when no CLI args provided
    else:
        print("RUNNING IN DEBUG MODE WITH PRESET VALUES")
        # Pre-configured test parameters for easy debugging
        debug_query = "primary color"
        filter = "doc.label == 1"
        # debug_tags = ["python", "function"]
        #debug_tags = []
        
        os.environ["DEBUG_MODE"] = "true"  # Set for breakpoint debugging
        
        # Call our function directly with debugging parameters
        results = run_search(
            query=debug_query,
            filter=filter,
            # tags=debug_tags,
            debug=True,
            show_aql=True
        )
        
        # Display results
        if "error" not in results:
            display_results(results, debug_query)