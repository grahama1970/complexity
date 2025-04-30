from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.box import DOUBLE_EDGE
import rich
from typing import Dict, Any
from colorama import init, Fore, Style
import sys
import json
from tabulate import tabulate
import os

from complexity.arangodb.log_utils import truncate_large_value
from loguru import logger  # Added for logging

def print_search_results(
    search_results: Dict[str, Any], 
    max_width: int = 120,
    title_field: str = "Content",
    id_field: str = "_key",
    score_field: str = "similarity_score",
    score_name: str = "Score",
    table_title: str = "Search Results"
) -> None:
    """
    Print search results in a well-formatted table or JSON using Rich when available.
    Falls back to tabulate with colorama if Rich is not available.
    Uses log_utils for safe display of large content.
    Supports tag-based scoring with dynamic score field and name.
    
    Args:
        search_results: The search results to display
        max_width: Maximum width for text fields in characters
        title_field: Field to use for primary content column
        id_field: Field to use for document ID
        score_field: Default field containing the score value (overridden for tag searches)
        score_name: Default display name for the score column (overridden for tag searches)
        table_title: Title for the results table
    """
    init(autoreset=True)
    
    # Determine search type and adjust score field/name
    search_type = search_results.get("search_type", "vector").lower()
    search_engine = search_results.get("search_engine", "").lower()
    if search_type == "tag" or search_engine == "tag-search":
        score_field = "tag_match_score"
        score_name = "Tag Match Score"
    
    # Get the requested output format
    output_format = search_results.get("format", "table").lower()
    
    # Create log-safe version of results for display
    try:
        safe_results = truncate_large_value(search_results.get("results", []))
    except Exception as e:
        print(f"{Fore.YELLOW}Warning: Failed to create log-safe results: {e}{Style.RESET_ALL}", file=sys.stderr)
        safe_results = search_results.get("results", [])
    
    # Log document contents for debugging
    if safe_results:
        logger.info(f"Raw document contents: {json.dumps(safe_results, indent=2, default=str)}")
        logger.info(f"Search type: {search_type}, Search engine: {search_engine}, Score field: {score_field}, Score name: {score_name}")
    
    # For JSON output, just print the JSON
    if output_format == "json":
        json_results = {
            "results": safe_results,
            "total": search_results.get("total", 0),
            "query": search_results.get("query", ""),
            "time": search_results.get("time", 0),
            "search_engine": search_engine,
            "tags": search_results.get("tags", []),
            "require_all_tags": search_results.get("require_all_tags", False),
            "offset": search_results.get("offset", 0),
            "limit": search_results.get("limit", 0),
            "search_type": search_type
        }
        print(json.dumps(json_results, indent=2))
        return
    
    # Check if we have an error
    if "error" in search_results:
        print(f"{Fore.RED}Error: {search_results['error']}{Style.RESET_ALL}")
        return
    
    # If not JSON, print a nice table view
    if not safe_results:
        print(f"{Fore.YELLOW}No results found.{Style.RESET_ALL}")
        return
    
    # Print basic search metadata
    result_count = len(safe_results)
    total_count = search_results.get("total", 0)
    query = search_results.get("query", "")
    search_time = search_results.get("time", 0)
    
    print(f"{Fore.CYAN}{'═' * 80}{Style.RESET_ALL}")
    print(f"Found {Fore.GREEN}{result_count}{Style.RESET_ALL} results out of {Fore.CYAN}{total_count}{Style.RESET_ALL} total")
    if query:
        safe_query = truncate_large_value(query, max_str_len=60)
        print(f"Query: '{Fore.YELLOW}{safe_query}{Style.RESET_ALL}'")
    if search_engine:
        print(f"Engine: {Fore.MAGENTA}{search_engine}{Style.RESET_ALL}")
    print(f"Time: {Fore.CYAN}{search_time:.3f}s{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'─' * 80}{Style.RESET_ALL}")
    
    # Try to use Rich for better table formatting
    try:
        console = Console()
        
        # Create Rich Table
        table = Table(
            title=f"{table_title} ({len(safe_results)})",
            show_header=True,
            header_style="bold cyan",
            box=DOUBLE_EDGE,
            expand=True,
            min_width=60,
            padding=(0, 1)
        )
        
        # Add columns with appropriate sizing
        table.add_column("#", justify="center", style="cyan", width=3, no_wrap=True)
        table.add_column("Key", style="yellow", width=20, no_wrap=True)
        table.add_column(title_field, ratio=3)
        table.add_column(score_name, justify="right", width=8, no_wrap=True)
        
        # Use fields_to_return from search_results
        fields_to_display = search_results.get("fields_to_return", ["question", "label", "validated"])
        
        # Add rows to the table
        for i, result in enumerate(safe_results):
            doc = result.get("doc", {})
            score = result.get(score_field, 0.0) if score_field in result else 0.0
            
            # Get document info with fallbacks
            key = str(doc.get(id_field, "N/A"))
            
            # Try fields_to_return in order to display
            text = "N/A"
            for field in fields_to_display:
                if field in doc and doc[field] is not None:
                    text = truncate_large_value(doc[field], max_str_len=80)
                    if text is None:
                        text = "N/A"
                    break
            
            # Add the main row with adaptive score coloring
            score_style = "green" if score > 0.9 else "yellow" if score > 0.7 else "white"
            table.add_row(
                str(i + 1),
                key,
                text,
                Text(f"{score:.2f}", style=score_style)
            )
            
            # Add tags if present
            if "tags" in doc and isinstance(doc["tags"], list) and doc["tags"]:
                tags = doc["tags"]
                safe_tags = truncate_large_value(tags, max_list_elements_shown=5)
                
                if isinstance(safe_tags, list):
                    tag_str = ", ".join(str(tag) for tag in safe_tags)
                    if len(tags) > 5:
                        tag_str += f" +{len(tags)-5} more"
                else:
                    tag_str = str(safe_tags) or "N/A"
                
                table.add_row(
                    "",
                    "",
                    Text(f"Tags: {tag_str}", style="blue"),
                    ""
                )
        
        # Print the table
        console.print(table)
        
    except ImportError as e:
        # Fall back to tabulate with colorama
        try:
            terminal_width = os.get_terminal_size().columns
            max_width = min(max_width, terminal_width - 20)
        except:
            pass
        
        key_width = min(20, max(12, max_width // 10))
        score_width = 8
        text_width = max_width - key_width - score_width - 10
        
        table_data = []
        fields_to_display = search_results.get("fields_to_return", ["question", "label", "validated"])
        
        for i, result in enumerate(safe_results):
            doc = result.get("doc", {})
            score = result.get(score_field, 0.0) if score_field in result else 0.0
            
            key = str(doc.get(id_field, "N/A"))
            if len(key) > key_width:
                key = key[:key_width-3] + "..."
            
            text = "N/A"
            for field in fields_to_display:
                if field in doc and doc[field] is not None:
                    text = truncate_large_value(doc[field], max_str_len=text_width)
                    if text is None:
                        text = "N/A"
                    break
            
            table_data.append([
                str(i + 1),
                key,
                text,
                f"{score:.2f}"
            ])
            
            if "tags" in doc and isinstance(doc["tags"], list) and doc["tags"]:
                tags = doc["tags"]
                safe_tags = truncate_large_value(tags, max_list_elements_shown=5)
                
                if isinstance(safe_tags, list):
                    tag_str = ", ".join(str(tag) for tag in safe_tags)
                    if len(tags) > 5:
                        tag_str += f" +{len(tags)-5} more"
                else:
                    tag_str = str(safe_tags) or "N/A"
                
                if isinstance(tag_str, str) and len(tag_str) > text_width:
                    tag_str = tag_str[:text_width-3] + "..."
                
                table_data.append([
                    "",
                    "",
                    f"Tags: {tag_str}",
                    ""
                ])
        
        headers = ["#", "Key", title_field, score_name]
        
        print(f"{Fore.CYAN}{table_title} ({len(safe_results)}):{Style.RESET_ALL}")
        print(tabulate(
            table_data, 
            headers=headers, 
            tablefmt="grid",
            numalign="center",
            stralign="left"
        ))
    
    print(f"{Fore.CYAN}{'═' * 80}{Style.RESET_ALL}")
    
    # Print detailed info for first result if available
    if safe_results and "doc" in safe_results[0]:
        print_result_details(safe_results[0], score_field, score_name)


def print_result_details(result: Dict[str, Any], score_field: str = "similarity_score", score_name: str = "Score") -> None:
    """
    Print beautifully formatted details about a search result.
    Uses log_utils for safe display of large content.
    
    Args:
        result: Search result to display
        score_field: Field containing the score value
        score_name: Display name for the score column
    """
    init(autoreset=True)
    
    doc = result.get("doc", {})
    score = result.get(score_field, 0.0) if score_field in result else 0.0
    
    # Print document header with key
    key = doc.get("_key", "N/A")
    header = f"{Fore.GREEN}{'═' * 80}{Style.RESET_ALL}"
    print(f"\n{header}")
    print(f"{Fore.GREEN}  DOCUMENT: {Fore.YELLOW}{key}{Style.RESET_ALL}  ")
    print(f"{header}")
    
    # Get fields to display (excluding internal fields and tags)
    display_fields = [f for f in doc.keys() if f not in ["_key", "_id", "tags", "_rev"]]
    
    # Print all document fields with truncation for large values
    for field in display_fields:
        if field in doc and doc[field] is not None:
            field_title = field.title()
            safe_value = truncate_large_value(doc[field], max_str_len=100)
            print(f"{Fore.YELLOW}{field_title}:{Style.RESET_ALL} {safe_value}")
    
    # Print score with color coding based on value
    if score > 0.9:
        score_str = f"{Fore.GREEN}{score:.2f}{Style.RESET_ALL}"
    elif score > 0.7:
        score_str = f"{Fore.YELLOW}{score:.2f}{Style.RESET_ALL}"
    else:
        score_str = f"{Fore.WHITE}{score:.2f}{Style.RESET_ALL}"
    print(f"\n{Fore.CYAN}{score_name}:{Style.RESET_ALL} {score_str}")
    
    # Print tags in a special section if present
    if "tags" in doc and isinstance(doc["tags"], list) and doc["tags"]:
        tags = doc["tags"]
        print(f"\n{Fore.BLUE}Tags:{Style.RESET_ALL}")
        
        safe_tags = truncate_large_value(tags, max_list_elements_shown=10)
        
        if isinstance(safe_tags, str):
            print(f"  {safe_tags}")
        else:
            tag_colors = [Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.GREEN, Fore.YELLOW]
            for i, tag in enumerate(safe_tags):
                color = tag_colors[i % len(tag_colors)]
                print(f"  • {color}{tag}{Style.RESET_ALL}")
    
    # Print footer
    print(f"{header}\n")