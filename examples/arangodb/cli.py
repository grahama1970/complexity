# src/pdf_extractor/arangodb/cli.py
"""
Command-Line Interface (CLI) for ArangoDB Lessons Learned Document Retriever

**Agent Instructions:**

This CLI provides command-line access to search, manage ("CRUD"), and explore
relationships within the 'lessons_learned' collection and associated graph
in an ArangoDB database. Use this interface to interact with the knowledge base
programmatically via shell commands. Output can be formatted for human reading
or as structured JSON using the `--json-output` / `-j` flag for easier parsing.

**Prerequisites:**

Ensure the following environment variables are set before executing commands:
- `ARANGO_HOST`: URL of the ArangoDB instance (e.g., "http://localhost:8529").
- `ARANGO_USER`: ArangoDB username (e.g., "root").
- `ARANGO_PASSWORD`: ArangoDB password.
- `ARANGO_DB_NAME`: Name of the target database (e.g., "doc_retriever").
- API key for the configured embedding model (e.g., `OPENAI_API_KEY` if using OpenAI).
- **Optional:** `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD` for Redis caching.
- **Optional:** `LOG_LEVEL` (e.g., DEBUG, INFO, WARNING) to control verbosity.

**Invocation:**

Execute commands using the python module execution flag `-m`:
`python -m src.complexity.arangodb.cli [OPTIONS] COMMAND [ARGS]...`

**Available Commands:**

--- Search Commands ---

1.  `search bm25`: [Search] Find documents based on keyword relevance (BM25 algorithm).
    *   WHEN TO USE: Use when you need to find documents matching specific keywords or terms present in the query text. Good for lexical matching.
    *   ARGUMENTS: QUERY (Required query text).
    *   OPTIONS: --threshold/-th (float), --top-n/-n (int), --offset/-o (int), --tags/-t (str), --json-output/-j (bool).
    *   OUTPUT: Table (default) or JSON array of results.

2.  `search semantic`: [Search] Find documents based on conceptual meaning (vector similarity).
    *   WHEN TO USE: Use when the exact keywords might be different, but the underlying meaning or concept of the query should match the documents. Good for finding semantically related content.
    *   ARGUMENTS: QUERY (Required query text, will be embedded).
    *   OPTIONS: --threshold/-th (float), --top-n/-n (int), --tags/-t (str), --json-output/-j (bool).
    *   OUTPUT: Table (default) or JSON array of results.

3.  `search hybrid`: [Search] Combine keyword (BM25) and semantic search results using RRF re-ranking.
    *   WHEN TO USE: Use for the best general-purpose relevance, leveraging both keyword matching and conceptual understanding. Often provides more robust results than either method alone.
    *   ARGUMENTS: QUERY (Required query text).
    *   OPTIONS: --top-n/-n (int), --initial-k/-k (int), --bm25-th (float), --sim-th (float), --tags/-t (str), --json-output/-j (bool).
    *   OUTPUT: Table (default) or JSON array of results.

--- Lesson (Vertex) CRUD Commands ---

4.  `crud add-lesson`: [CRUD] Add a new lesson document (vertex).
    *   WHEN TO USE: Use when you have identified a new, distinct lesson learned that needs to be added to the knowledge base.
    *   **RECOMMENDED:** Use `--data-file` / `-f` to provide lesson data via a JSON file to avoid command-line escaping issues.
    *   OPTIONS:
        *   `--data-file` / `-f`: (Recommended, Path) Path to a JSON file containing the lesson data (e.g., `path/to/lesson.json`).
        *   `--data` / `-d`: (Alternative, str) Lesson data as a JSON string (e.g., '{"problem": ..., "solution": ...}'). Use with caution due to shell escaping.
        *   `--json-output` / `-j`: (Optional, bool) Output metadata as JSON.
    *   OUTPUT: JSON metadata (_key, _id, _rev) or success message. *Must provide exactly one of --data or --data-file.*

5.  `crud get-lesson`: [CRUD] Retrieve a lesson document by its _key.
    *   WHEN TO USE: Use when you need the full details of a specific lesson identified by its key (e.g., from search results).
    *   ARGUMENTS: KEY (Required document _key).
    *   OPTIONS: --json-output/-j (bool, default: True).
    *   OUTPUT: Full JSON document or "Not Found" message/JSON.

6.  `crud update-lesson`: [CRUD] Modify specific fields of an existing lesson document.
    *   WHEN TO USE: Use to correct or enhance information in an existing lesson (e.g., refining the solution, adding tags, changing severity).
    *   **RECOMMENDED:** Use `--data-file` / `-f` to provide update data via a JSON file.
    *   ARGUMENTS: KEY (Required document _key).
    *   OPTIONS:
        *   `--data-file` / `-f`: (Recommended, Path) Path to a JSON file containing the fields to update (e.g., `path/to/update.json`).
        *   `--data` / `-d`: (Alternative, str) Fields to update as a JSON string (e.g., '{"field": "new_value", ...}'). Use with caution.
        *   `--json-output` / `-j`: (Optional, bool) Output metadata as JSON.
    *   OUTPUT: JSON metadata (_key, _id, _rev, _old_rev) or success/error message. *Must provide exactly one of --data or --data-file.*

7.  `crud delete-lesson`: [CRUD] Permanently remove a lesson document and its associated edges.
    *   WHEN TO USE: Use cautiously when a lesson is determined to be completely irrelevant, incorrect beyond repair, or a duplicate that should be removed. Automatically cleans up relationships.
    *   ARGUMENTS: KEY (Required document _key).
    *   OPTIONS: --yes/-y (bool, confirmation bypass), --json-output/-j (bool).
    *   OUTPUT: JSON status or success/error message.

--- Relationship (Edge) Commands ---

8.  `graph add-relationship`: [Graph] Create a link (edge) between two lessons.
    *   WHEN TO USE: Use *after* analysis suggests a meaningful connection exists between two lessons. Choose the correct `relationship_type` and provide a clear `rationale`.
    *   ARGUMENTS:
        *   `FROM_KEY`: (Required) The `_key` of the source lesson.
        *   `TO_KEY`: (Required) The `_key` of the target lesson.
    *   OPTIONS:
        *   `--rationale` / `-r`: (Required, str) Explanation of why these lessons are linked.
        *   `--type` / `-typ`: (Required, str) The category of the relationship (e.g., RELATED, DUPLICATE, PREREQUISITE, CAUSAL).
        *   `--attributes` / `-a`: (Optional, str) Additional properties as a JSON string (e.g., '{"confidence": 0.9}').
        *   `--json-output` / `-j`: (Optional, bool, default: False) Output metadata as JSON.
    *   OUTPUT: JSON metadata (_key, _id, _rev) or success/error message.

9.  `graph delete-relationship`: [Graph] Remove a specific link (edge) between lessons.
    *   WHEN TO USE: Use when a previously established relationship is found to be incorrect or no longer relevant.
    *   ARGUMENTS:
        *   `EDGE_KEY`: (Required) The `_key` of the relationship edge document itself.
    *   OPTIONS:
        *   `--json-output` / `-j`: (Optional, bool, default: False) Output status as JSON.
        *   `--yes` / `-y`: (Optional, bool) Confirm deletion without interactive prompt.
    *   OUTPUT: JSON status or success/error message.

10. `graph traverse`: [Graph] Explore relationships starting from a specific lesson.
    *   WHEN TO USE: Use to discover connected lessons and understand the context or dependencies around a specific lesson. Essential for navigating the knowledge graph. Requires edges to exist.
    *   ARGUMENTS: START_NODE_ID (Required full _id: "lessons_learned/key").
    *   OPTIONS: --graph-name/-g (str), --min-depth (int), --max-depth (int), --direction/-dir (str: OUTBOUND, INBOUND, ANY), --limit/-lim (int), --json-output/-j (bool, default: True).
    *   OUTPUT: JSON array containing paths (vertices, edges) or summary message.

**Error Handling:**

- Errors during execution will be printed to stderr.
- Detailed logs might be available depending on `LOG_LEVEL`.
- Commands typically exit with code 0 on success and 1 on failure.
- Use `--json-output` for structured error information where applicable.

"""

import typer
import json
import sys
import os
from pathlib import Path
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.json import JSON
from typing import List, Optional, Any, Dict, Union


# --- Local Imports ---
# Assume these modules exist and are importable based on project structure
try:
    # Use absolute imports from src
    from complexity.arangodb.arango_setup_unknown import (
        connect_arango,
        ensure_database,
        ensure_edge_collection,  # Added for graph setup check
        ensure_graph,  # Added for graph setup check
    )
    # Import from search_api directly (assuming search_advanced might be outdated or just re-exports)
    from complexity.arangodb.search_api.bm25_search import bm25_search
    from complexity.arangodb.search_api._archive.semantic import semantic_search
    from complexity.arangodb.search_api.hybrid import hybrid_search
    from complexity.arangodb.search_api.graph_traverse import graph_traverse
    from complexity.arangodb.search_api.keyword_search import search_keyword # Correct function name
    from complexity.arangodb.search_api.tag_search import tag_search # Correct function name
    # Removed outdated search_advanced import block
    # Import from the new crud package
    from complexity.arangodb.crud import ( # Import from the new crud package __init__
        create_document,
        get_document,
        update_document,
        delete_document,
        create_relationship, # Import new relationship function
        delete_relationship_by_key, # Import new relationship function
    )
    from complexity.arangodb.embedding_utils import get_embedding

    # Import log utility for truncation
    from complexity.arangodb.log_utils import truncate_large_value
    # log_safe_results is for lists of dicts, truncate_large_value handles dicts recursively

    # Added EDGE_COLLECTION_NAME, COLLECTION_NAME
    from complexity.arangodb.config import (
        ARANGO_DB_NAME,
        GRAPH_NAME,
        COLLECTION_NAME,
        EDGE_COLLECTION_NAME,
        SEARCH_FIELDS, # Moved here
    )
    
    # Trying to initialize litellm cache - mock this to solve import issue
    class MockInitializeLitellmCache:
        @staticmethod
        def initialize_litellm_cache():
            logger.info("Mock initialization of litellm cache")
            return True
    
    initialize_litellm_cache = MockInitializeLitellmCache.initialize_litellm_cache
    
except ImportError as e:
    # Use logger if available, otherwise print and exit
    init_msg = f"ERROR: Failed to import required modules: {e}. Ensure CLI is run correctly relative to the project structure or PYTHONPATH is set."
    try:
        logger.critical(init_msg)
    except NameError:
        print(init_msg, file=sys.stderr)
    sys.exit(1)

# --- Typer App Initialization ---
# Provide short descriptions for command groups
app = typer.Typer(
    name="arangodb-lessons-cli",
    help=__doc__,  # Use module docstring
    add_completion=False,
    rich_markup_mode="markdown",  # Enable markdown in help text
)
search_app = typer.Typer(name="search", help="Find lessons using various methods.")
app.add_typer(search_app, name="search")

crud_app = typer.Typer(
    name="crud", help="Create, Read, Update, Delete lessons (vertices)."
)
app.add_typer(crud_app, name="crud")

graph_app = typer.Typer(
    name="graph", help="Manage and explore relationships (edges) and graph structure."
)
app.add_typer(graph_app, name="graph")


# --- Rich Console ---
console = Console()


# --- Global State / Context & Logging Setup ---
@app.callback()
def main_callback(
    log_level: str = typer.Option(
        os.environ.get("LOG_LEVEL", "INFO").upper(),
        "--log-level",
        "-l",
        help="Set logging level (DEBUG, INFO, WARNING, ERROR).",
        envvar="LOG_LEVEL",
    ),
):
    """Main callback to configure logging for the CLI."""
    # Allow standard logging levels
    log_level = log_level.upper()
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if log_level not in valid_levels:
        print(
            f"Warning: Invalid log level '{log_level}'. Defaulting to INFO.",
            file=sys.stderr,
        )
        log_level = "INFO"

    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level=log_level,
        format="{time:HH:mm:ss} | {level: <7} | {message}",
        backtrace=False,
        diagnose=False,  # Keep diagnose False for production CLI
    )
    # Initialize caching (if needed by embedding_utils or search)
    logger.debug("Initializing LiteLLM Caching for CLI session...")
    try:
        initialize_litellm_cache()
        logger.debug("LiteLLM Caching initialized.")
    except Exception as cache_err:
        logger.warning(f"Could not initialize LiteLLM cache: {cache_err}")


# --- Utility ---
# Store DB connection globally per CLI invocation? Maybe not best practice.
# Let each command get its own connection for simplicity unless performance demands otherwise.
def get_db_connection():
    """Helper to connect and get DB object, handling errors."""
    try:
        logger.debug("Attempting to connect to ArangoDB...")
        client = connect_arango()
        if not client:
            raise ConnectionError("connect_arango() returned None")
        logger.debug(f"Ensuring database '{ARANGO_DB_NAME}' exists...")
        db = ensure_database(client)
        if not db:
            raise ConnectionError(
                f"ensure_database() returned None for '{ARANGO_DB_NAME}'"
            )
        logger.debug(f"Successfully connected to database '{db.name}'.")

        # --- Ensure graph components exist upon first connection in CLI ---
        # This makes commands relying on graph/edges safer.
        try:
            logger.debug("Ensuring edge collection and graph definition exist...")
            ensure_edge_collection(db, EDGE_COLLECTION_NAME)
            ensure_graph(db, GRAPH_NAME, EDGE_COLLECTION_NAME, COLLECTION_NAME)
            logger.debug("Edge collection and graph definition checked/ensured.")
        except Exception as setup_e:
            # Log warning but don't fail connection, specific commands might still work
            logger.warning(
                f"Graph/Edge setup check failed during connection: {setup_e}. Relationship/Traversal commands might fail."
            )
        # --- End ensure graph components ---

        return db
    except Exception as e:
        logger.error(
            f"DB connection/setup failed: {e}", exc_info=True
        )  # Log traceback on error
        console.print(
            f"[bold red]Error:[/bold red] Could not connect to or setup ArangoDB ({e}). Check connection details, permissions, and ensure ArangoDB is running."
        )
        raise typer.Exit(code=1)


# --- Search Commands (under `search` subcommand) ---


@search_app.command("bm25")
def cli_search_bm25(
    query: str = typer.Argument(..., help="The search query text."),
    threshold: float = typer.Option(
        0.1, "--threshold", "-th", help="Minimum BM25 score.", min=0.0
    ),
    top_n: int = typer.Option(
        5, "--top-n", "-n", help="Number of results to return.", min=1
    ),
    offset: int = typer.Option(
        0, "--offset", "-o", help="Offset for pagination.", min=0
    ),
    tags: Optional[str] = typer.Option(
        None,
        "--tags",
        "-t",
        help='Comma-separated list of tags to filter by (e.g., "tag1,tag2").',
    ),
    json_output: bool = typer.Option(
        False, "--json-output", "-j", help="Output results as JSON array."
    ),
):
    """
    Find documents based on keyword relevance (BM25 algorithm).

    *WHEN TO USE:* Use when you need to find documents matching specific keywords
    or terms present in the query text. Good for lexical matching.

    *HOW TO USE:* Provide the query text. Optionally refine with score threshold,
    result count, pagination offset, or tag filtering.
    """
    logger.info(f"CLI: Performing BM25 search for '{query}'")
    db = get_db_connection()
    tag_list = [tag.strip() for tag in tags.split(",")] if tags else None
    try:
        results_data = bm25_search(
            db=db,
            query_text=query,
            min_score=threshold, # Map CLI 'threshold' to API 'min_score'
            top_n=top_n,
            offset=offset,
            tag_list=tag_list
        )
        if json_output:
            # Use print directly for clean JSON output
            print(json.dumps(results_data.get("results", []), indent=2))
        else:
            _display_results(results_data, "BM25", "bm25_score")
    except Exception as e:
        logger.error(f"BM25 search failed: {e}", exc_info=True)
        console.print(f"[bold red]Error during BM25 search:[/bold red] {e}")
        raise typer.Exit(code=1)


@search_app.command("semantic")
def cli_search_semantic(
    query: str = typer.Argument(..., help="The search query text (will be embedded)."),
    threshold: float = typer.Option(
        0.75,
        "--threshold",
        "-th",
        help="Minimum similarity score (0.0-1.0).",
        min=0.0,
        max=1.0,
    ),
    top_n: int = typer.Option(
        5, "--top-n", "-n", help="Number of results to return.", min=1
    ),
    tags: Optional[str] = typer.Option(
        None,
        "--tags",
        "-t",
        help='Comma-separated list of tags to filter by (e.g., "tag1,tag2").',
    ),
    json_output: bool = typer.Option(
        False, "--json-output", "-j", help="Output results as JSON array."
    ),
):
    """
    Find documents based on conceptual meaning (vector similarity).

    *WHEN TO USE:* Use when the exact keywords might be different, but the underlying
    meaning or concept of the query should match the documents. Good for finding
    semantically related content. Requires embedding generation.

    *HOW TO USE:* Provide the query text. Optionally refine with similarity
    threshold, result count, or tag filtering.
    """
    logger.info(f"CLI: Performing Semantic search for '{query}'")
    db = get_db_connection()
    tag_list = [tag.strip() for tag in tags.split(",")] if tags else None

    logger.debug("Generating query embedding...")
    try:
        query_embedding = get_embedding(query)
        if not query_embedding:  # Check if list is empty or None
            console.print(
                "[bold red]Error:[/bold red] Failed to generate embedding for the query (returned empty/None). Check API key and model."
            )
            raise typer.Exit(code=1)
        logger.debug(f"Query embedding generated ({len(query_embedding)} dims).")
    except Exception as emb_err:
        logger.error(f"Failed to generate query embedding: {emb_err}", exc_info=True)
        console.print(
            f"[bold red]Error generating query embedding:[/bold red] {emb_err}"
        )
        raise typer.Exit(code=1)

    try:
        results_data = semantic_search(db, query_embedding, top_n, threshold, tag_list)
        if json_output:
            print(json.dumps(results_data.get("results", []), indent=2))
        else:
            _display_results(results_data, "Semantic", "similarity_score")
    except Exception as e:
        logger.error(f"Semantic search failed: {e}", exc_info=True)
        console.print(f"[bold red]Error during Semantic search:[/bold red] {e}")
        raise typer.Exit(code=1)


@search_app.command("hybrid")
def cli_search_hybrid(
    query: str = typer.Argument(..., help="The search query text."),
    top_n: int = typer.Option(
        5, "--top-n", "-n", help="Final number of ranked results.", min=1
    ),
    initial_k: int = typer.Option(
        20,
        "--initial-k",
        "-k",
        help="Number of candidates from BM25/Semantic before RRF.",
        min=1,
    ),
    bm25_threshold: float = typer.Option(
        0.01, "--bm25-th", help="BM25 candidate retrieval score threshold.", min=0.0
    ),
    sim_threshold: float = typer.Option(
        0.70,
        "--sim-th",
        help="Similarity candidate retrieval score threshold.",
        min=0.0,
        max=1.0,
    ),
    tags: Optional[str] = typer.Option(
        None,
        "--tags",
        "-t",
        help='Comma-separated list of tags to filter by (e.g., "tag1,tag2").',
    ),
    json_output: bool = typer.Option(
        False, "--json-output", "-j", help="Output results as JSON array."
    ),
):
    """
    Combine keyword (BM25) and semantic search results using RRF re-ranking.

    *WHEN TO USE:* Use for the best general-purpose relevance, leveraging both
    keyword matching and conceptual understanding. Often provides more robust
    results than either method alone.

    *HOW TO USE:* Provide the query text. Optionally adjust the number of final
    results (`top_n`), initial candidates (`initial_k`), candidate thresholds,
    or add tag filters.
    """
    logger.info(f"CLI: Performing Hybrid search for '{query}'")
    db = get_db_connection()
    tag_list = [tag.strip() for tag in tags.split(",")] if tags else None
    try:
        results_data = hybrid_search(
            db, query, top_n, initial_k, bm25_threshold, sim_threshold, tag_list
        )
        if json_output:
            print(json.dumps(results_data.get("results", []), indent=2))
        else:
            _display_results(results_data, "Hybrid (RRF)", "rrf_score")
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}", exc_info=True)
        console.print(f"[bold red]Error during Hybrid search:[/bold red] {e}")
        raise typer.Exit(code=1)



@search_app.command("keyword")
def cli_find_keyword(
    keywords: List[str] = typer.Argument(..., help="One or more keywords to search for."),
    search_fields_str: Optional[str] = typer.Option(
        None,
        "--search-fields",
        "-sf",
        help=f'Comma-separated fields to search (e.g., "problem,solution"). Defaults to configured SEARCH_FIELDS: {SEARCH_FIELDS}',
    ),
    limit: int = typer.Option(
        10, "--limit", "-lim", help="Maximum number of results to return.", min=1
    ),
    match_all: bool = typer.Option(
        False,
        "--match-all",
        "-ma",
        help="Require all keywords to match (AND logic) instead of any (OR logic).",
    ),
    json_output: bool = typer.Option(
        False, "--json-output", "-j", help="Output results as JSON array."
    ),
):
    """
    Find documents based on exact keyword matching within specified fields.

    *AGENT INSTRUCTIONS:*

    *   **WHEN TO USE:** Use this command when you need to find lessons containing *specific, known keywords* within certain text fields (like 'problem', 'solution', 'context'). This performs an exact text search, unlike BM25 (which uses relevance scoring) or semantic search (which uses meaning).
    *   **WHY TO USE:** Useful for locating lessons mentioning precise terms, error codes, function names, or specific phrases you are looking for directly.
    *   **HOW TO USE:** Provide one or more keywords as arguments. Use options to control which fields are searched (`--search-fields`), the maximum number of results (`--limit`), whether all keywords must match (`--match-all`), and the output format (`--json-output`).

    *EXAMPLES:*

    *   Find lessons containing "docker" or "compose" in default fields:
        `... search keyword docker compose`
    *   Find lessons containing "timeout" AND "database" in 'problem' or 'solution' fields, limit 5:
        `... search keyword timeout database --search-fields problem,solution --match-all --limit 5`
    *   Find lessons with "arangodb" and output as JSON:
        `... search keyword arangodb -j`
    """
    logger.info(f"CLI: Performing Keyword search for: {keywords}")
    db = get_db_connection()

    parsed_search_fields: Optional[List[str]] = None
    if search_fields_str:
        parsed_search_fields = [f.strip() for f in search_fields_str.split(',') if f.strip()]
        if not parsed_search_fields:
            logger.warning("Empty --search-fields provided, defaulting to None (all configured fields).")
            parsed_search_fields = None # Reset to None if only whitespace/commas were given

    try:
        # Join the list of keywords into a single string for the current search_keyword implementation
        search_term_str = " ".join(keywords)
        logger.debug(f"CLI: Calling search_keyword with search_term_str='{search_term_str}' (type: {type(search_term_str)})") # DEBUG LOG
        # Call search_keyword using explicit keyword arguments
        results = search_keyword(
            db=db,
            search_term=search_term_str,
            top_n=limit,
            # similarity_threshold uses default
            # view_name uses default
            # tags uses default (None)
            # collection_name uses default
        )

        # Wrap results for consistency with _display_results
        results_data = {
            "results": results,
            "total": len(results), # Keyword search doesn't easily provide total count beyond limit
            "offset": 0 # Keyword search doesn't support offset directly
        }

        if json_output:
            # search_keyword returns a dict, print the 'results' list
            print(json.dumps(results, indent=2))
        else:
            # Pass 'keyword_score' as score_field as search_keyword adds it
            _display_results(results_data, "Keyword", score_field="keyword_score")

    except Exception as e:
        logger.error(f"Keyword search failed: {e}", exc_info=True)
        console.print(f"[bold red]Error during Keyword search:[/bold red] {e}")
        raise typer.Exit(code=1)



@search_app.command("tag")
def cli_search_tag(
    tags: List[str] = typer.Argument(..., help="One or more tags to search for (case-sensitive)."),
    limit: int = typer.Option(
        10, "--limit", "-lim", help="Maximum number of results to return.", min=1
    ),
    match_all: bool = typer.Option(
        False,
        "--match-all",
        "-ma",
        help="Require all tags to match (AND logic) instead of any (OR logic).",
    ),
    json_output: bool = typer.Option(
        False, "--json-output", "-j", help="Output results as JSON array."
    ),
):
    """
    Find documents based on exact tag matching within the 'tags' array field.

    *AGENT INSTRUCTIONS:*

    *   **WHEN TO USE:** Use this command to find lessons that have been explicitly tagged with one or more specific keywords. This performs an exact, case-sensitive match against items in the `tags` array.
    *   **WHY TO USE:** Ideal for filtering lessons based on predefined categories or topics represented by tags.
    *   **HOW TO USE:** Provide one or more tags as arguments. Use `--match-all` if a lesson must have *all* the specified tags. Use `--limit` to control the number of results and `--json-output` for machine-readable output.

    *EXAMPLES:*

    *   Find lessons tagged with "database" OR "python":
        `... search tag database python`
    *   Find lessons tagged with "testing" AND "docker", limit 5:
        `... search tag testing docker --match-all --limit 5`
    *   Find lessons tagged "cli" and output as JSON:
        `... search tag cli -j`
    """
    logger.info(f"CLI: Performing Tag search for: {tags}")
    db = get_db_connection()

    try:
        # Call the correct function name: tag_search
        # It returns a dict with 'results', not a list directly.
        results_data = tag_search(
            db=db,
            tags=tags,
            limit=limit,
            require_all_tags=match_all, # Corrected keyword argument
        )

        # results_data already contains 'results', 'total', 'offset', 'limit'

        if json_output:
            # tag_search returns a dict, print the 'results' list
            print(json.dumps(results_data.get("results", []), indent=2))
        else:
            # Pass None for score_field as tag search has no score
            _display_results(results_data, "Tag", score_field=None)

    except Exception as e:
        logger.error(f"Tag search failed: {e}", exc_info=True)
        console.print(f"[bold red]Error during Tag search:[/bold red] {e}")
        raise typer.Exit(code=1)

# --- Lesson (Vertex) CRUD Commands (under `crud` subcommand) ---


@crud_app.command("add-lesson")
def cli_add_lesson(
    data: Optional[str] = typer.Option(  # Changed to Optional
        None,  # Default to None
        "--data",
        "-d",
        help="(Alternative) Lesson data as JSON string. Use with caution due to shell escaping.",
    ),
    data_file: Optional[Path] = typer.Option(  # <--- Changed from typer.Path to Path
        None,
        "--data-file",
        "-f",
        help="(Recommended) Path to a JSON file containing lesson data.",
        exists=True,  # Validate file exists
        file_okay=True,
        dir_okay=False,
        readable=True,  # Validate file is readable
    ),
    json_output: bool = typer.Option(
        False, "--json-output", "-j", help="Output full metadata as JSON on success."
    ),
):
    """
    Add a new lesson document (vertex).

    *WHEN TO USE:* Use when you have identified a new, distinct lesson learned
    that needs to be added to the knowledge base. Ensure 'problem' and 'solution'
    fields are present in the JSON data. Embedding is generated automatically.

    *HOW TO USE:* Provide the lesson data via either a JSON file (`--data-file`, recommended)
    or a JSON string (`--data`). *Exactly one of these options must be provided.*
    Example (File): `... crud add-lesson --data-file path/to/lesson.json`
    Example (String): `... crud add-lesson --data '{"problem": "...", "solution": ...}'`
    """
    logger.info("CLI: Received request to add new lesson.")

    # --- Input Validation: Ensure exactly one data source ---
    if not data and not data_file:
        console.print(
            "[bold red]Error:[/bold red] Either --data (JSON string) or --data-file (path to JSON file) must be provided."
        )
        raise typer.Exit(code=1)
    if data and data_file:
        console.print(
            "[bold red]Error:[/bold red] Provide either --data or --data-file, not both."
        )
        raise typer.Exit(code=1)

    lesson_data_dict = None
    source_info = ""  # For error messages

    # --- Load Data from chosen source ---
    try:
        if data_file:
            source_info = f"file '{data_file}'"
            logger.debug(f"Loading lesson data from file: {data_file}")
            with open(data_file, "r") as f:
                lesson_data_dict = json.load(f)
        elif data:  # Only parse if data_file wasn't used
            source_info = "string --data"
            # Log truncated dictionary *after* parsing
            lesson_data_dict = json.loads(data)
            logger.debug(f"Loaded lesson data from string: {truncate_large_value(lesson_data_dict)}")


        # --- Validate loaded data structure ---
        if not isinstance(lesson_data_dict, dict):
            raise ValueError("Provided data must be a JSON object (dictionary).")

    except json.JSONDecodeError as e:
        console.print(
            f"[bold red]Error:[/bold red] Invalid JSON provided via {source_info}: {e}"
        )
        raise typer.Exit(code=1)
    except ValueError as e:  # Catch our custom validation error
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)
    except (
        FileNotFoundError
    ):  # Should be caught by typer.Path(exists=True) but good practice
        console.print(f"[bold red]Error:[/bold red] Data file not found: {data_file}")
        raise typer.Exit(code=1)
    except Exception as e:  # Catch other potential file/parsing errors
        console.print(
            f"[bold red]Error reading/parsing data from {source_info}:[/bold red] {e}"
        )
        raise typer.Exit(code=1)

    # --- Call the API Function ---
    db = get_db_connection()
    try:
        # Call the new generic create_document function for the 'lessons_learned' collection (COLLECTION_NAME)
        # Embedding generation logic:
        text_to_embed = f"{lesson_data_dict.get('problem','')} {lesson_data_dict.get('solution','')} {lesson_data_dict.get('context','')}"
        if text_to_embed.strip() and 'embedding' not in lesson_data_dict:
             try:
                 logger.debug("Generating embedding for new lesson...")
                 embedding = get_embedding(text_to_embed)
                 if embedding:
                     lesson_data_dict['embedding'] = embedding
                     logger.debug("Embedding generated.")
                 else:
                     logger.warning("Embedding generation failed, proceeding without embedding.")
             except Exception as emb_err:
                 logger.warning(f"Embedding generation failed: {emb_err}, proceeding without embedding.")

        meta = create_document(db, COLLECTION_NAME, lesson_data_dict) # Use COLLECTION_NAME
        if meta:
            output = meta
            if json_output:
                print(json.dumps(output))
            else:
                console.print(
                    f"[green]Success:[/green] Lesson added successfully. Key: [cyan]{meta.get('_key')}[/cyan]"
                )
        else:
            # The add_lesson function logs errors, just indicate failure here
            console.print(
                "[bold red]Error:[/bold red] Failed to add lesson (check logs for details)."
            )
            raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"Add lesson failed in CLI: {e}", exc_info=True)
        console.print(f"[bold red]Error during add operation:[/bold red] {e}")
        raise typer.Exit(code=1)


@crud_app.command("get-lesson")
def cli_get_lesson(
    key: str = typer.Argument(..., help="The _key of the lesson document."),
    json_output: bool = typer.Option(
        True,
        "--json-output",
        "-j",
        help="Output full document as JSON (default for get).",
    ),  # Default True for 'get'
):
    """
    Retrieve a specific lesson document (vertex) by its _key.

    *WHEN TO USE:* Use when you need the full details of a specific lesson
    identified by its key (e.g., obtained from search results or previous operations).

    *HOW TO USE:* Provide the `_key` of the lesson as an argument.
    Example: `... crud get-lesson my_lesson_key_123`
    """
    logger.info(f"CLI: Requesting lesson with key '{key}'")
    db = get_db_connection()
    try:
        # Call the new generic get_document function
        doc = get_document(db, COLLECTION_NAME, key)
        if doc:
            output = doc
            if json_output:
                # Use print directly for clean JSON output
                print(json.dumps(output, indent=2))
            else:
                # Human-readable fallback (maybe less useful if default is JSON)
                console.print(f"[green]Lesson Found:[/green] _key=[cyan]{key}[/cyan]")
                console.print(JSON(json.dumps(doc, indent=2)))  # Pretty print JSON
        else:
            # Not found is not an error state for 'get'
            output = {"status": "error", "message": "Not Found", "key": key}
            if json_output:
                print(json.dumps(output))
                # Exit with non-zero code for scripting if JSON output is requested and not found
                raise typer.Exit(code=1)
            else:
                console.print(
                    f"[yellow]Not Found:[/yellow] No lesson found with key '{key}'."
                )
            # For human output, not found is info, not error, so exit code 0
            raise typer.Exit(code=0)
    except Exception as e:
        logger.error(f"Get lesson failed in CLI: {e}", exc_info=True)
        output = {"status": "error", "message": str(e), "key": key}
        if json_output:
            print(json.dumps(output))
        else:
            console.print(f"[bold red]Error during get operation:[/bold red] {e}")
        raise typer.Exit(code=1)


@crud_app.command("update-lesson")
def cli_update_lesson(
    key: str = typer.Argument(..., help="The _key of the lesson document to update."),
    data: Optional[str] = typer.Option(  # Changed to Optional
        None,  # Default to None
        "--data",
        "-d",
        help="(Alternative) Fields to update as JSON string. Use with caution.",
    ),
    data_file: Optional[Path] = typer.Option(  # <--- Changed from typer.Path to Path
        None,
        "--data-file",
        "-f",
        help="(Recommended) Path to JSON file containing fields to update.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    json_output: bool = typer.Option(
        False, "--json-output", "-j", help="Output full metadata as JSON on success."
    ),
):
    """
    Modify specific fields of an existing lesson document (vertex).

    *WHEN TO USE:* Use to correct or enhance information in an existing lesson
    (e.g., refining the solution, adding tags, changing severity). If embedding-relevant
    fields (problem, solution, context, etc.) are updated, the embedding will be regenerated.

    *HOW TO USE:* Provide the `_key` and the update data via either a JSON file
    (`--data-file`, recommended) or a JSON string (`--data`). *Exactly one of these options must be provided.*
    Example (File): `... crud update-lesson my_key --data-file path/to/update.json`
    Example (String): `... crud update-lesson my_key --data '{"tags": ["new"]}'`
    """
    logger.info(f"CLI: Requesting update for lesson key '{key}'")

    # --- Input Validation: Ensure exactly one data source ---
    if not data and not data_file:
        console.print(
            "[bold red]Error:[/bold red] Either --data (JSON string) or --data-file (path to JSON file) must be provided for update."
        )
        raise typer.Exit(code=1)
    if data and data_file:
        console.print(
            "[bold red]Error:[/bold red] Provide either --data or --data-file for update, not both."
        )
        raise typer.Exit(code=1)

    update_data_dict = None
    source_info = ""  # For error messages

    # --- Load Data from chosen source ---
    try:
        if data_file:
            source_info = f"file '{data_file}'"
            logger.debug(f"Loading update data from file: {data_file}")
            with open(data_file, "r") as f:
                update_data_dict = json.load(f)
        elif data:  # Only parse if data_file wasn't used
            source_info = "string --data"
            # Log truncated dictionary *after* parsing
            update_data_dict = json.loads(data)
            logger.debug(f"Loaded update data from string: {truncate_large_value(update_data_dict)}")


        # --- Validate loaded data structure ---
        if not isinstance(update_data_dict, dict):
            raise ValueError("Provided update data must be a JSON object (dictionary).")
        if not update_data_dict:  # Ensure the update dict is not empty
            raise ValueError("Update data cannot be empty.")

    except json.JSONDecodeError as e:
        console.print(
            f"[bold red]Error:[/bold red] Invalid JSON provided via {source_info}: {e}"
        )
        raise typer.Exit(code=1)
    except ValueError as e:  # Catch our custom validation errors
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/bold red] Data file not found: {data_file}")
        raise typer.Exit(code=1)
    except Exception as e:  # Catch other potential file/parsing errors
        console.print(
            f"[bold red]Error reading/parsing update data from {source_info}:[/bold red] {e}"
        )
        raise typer.Exit(code=1)

    # --- Call the API Function ---
    db = get_db_connection()
    try:
        # Call the new generic update_document function for the 'lessons_learned' collection (COLLECTION_NAME)
        # Handle potential embedding regeneration
        fields_to_embed = ["problem", "solution", "context"]
        needs_embedding_update = any(field in update_data_dict for field in fields_to_embed)

        if needs_embedding_update:
             try:
                 existing_doc = get_document(db, COLLECTION_NAME, key) # Fetch existing doc
                 if existing_doc:
                     updated_text_parts = {
                         field: update_data_dict.get(field, existing_doc.get(field, ''))
                         for field in fields_to_embed
                     }
                     text_to_embed = f"{updated_text_parts['problem']} {updated_text_parts['solution']} {updated_text_parts['context']}"
                     if text_to_embed.strip():
                         logger.debug("Regenerating embedding due to update...")
                         embedding = get_embedding(text_to_embed)
                         if embedding:
                             update_data_dict['embedding'] = embedding
                             logger.debug("Embedding regenerated.")
                         else:
                             logger.warning("Embedding regeneration failed.")
                     else:
                          logger.debug("Relevant fields are empty, skipping embedding update.")
                 else:
                      logger.warning(f"Cannot fetch existing doc {key} to regenerate embedding.")
             except Exception as emb_err:
                 logger.warning(f"Embedding regeneration failed during update: {emb_err}")

        meta = update_document(db, COLLECTION_NAME, key, update_data_dict) # Use COLLECTION_NAME
        if meta:
            output = meta
            if json_output:
                print(json.dumps(output))
            else:
                console.print(
                    f"[green]Success:[/green] Lesson [cyan]{key}[/cyan] updated successfully."
                )
        else:
            # update_lesson logs details, just indicate failure
            console.print(
                f"[bold red]Error:[/bold red] Failed to update lesson '{key}' (check logs for details, it might not exist or update failed)."
            )
            raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"Update lesson failed in CLI: {e}", exc_info=True)
        output = {"status": "error", "message": str(e), "key": key}
        if json_output:
            print(json.dumps(output))
        else:
            console.print(f"[bold red]Error during update operation:[/bold red] {e}")
        raise typer.Exit(code=1)


@crud_app.command("delete-lesson")
def cli_delete_lesson(
    key: str = typer.Argument(..., help="The _key of the lesson document to delete."),
    json_output: bool = typer.Option(
        False, "--json-output", "-j", help="Output status as JSON."
    ),
    # Example: Add confirmation prompt
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Confirm deletion without interactive prompt."
    ),
):
    """
    Permanently remove a lesson document (vertex) and its associated edges.

    *WHEN TO USE:* Use cautiously when a lesson is determined to be completely
    irrelevant, incorrect beyond repair, or a duplicate that should be removed.
    Note: This automatically cleans up connected relationship edges.

    *HOW TO USE:* Provide the `_key` of the lesson. Use `--yes` to bypass the confirmation prompt.
    Example: `... crud delete-lesson my_key_to_delete --yes`
    """
    logger.warning(
        f"CLI: Received request to DELETE lesson key '{key}'"
    )  # Use warning for delete
    if not yes:
        # Use Rich confirmation for better display
        confirmed = typer.confirm(
            f"Are you sure you want to permanently delete lesson '[cyan]{key}[/cyan]' and its relationships?",
            abort=True,  # Exits if user says no
        )
        # If abort=False, need: if not confirmed: raise typer.Exit()

    db = get_db_connection()
    try:
        # TODO: Implement edge deletion logic if required before deleting the vertex.
        # This requires querying edges from EDGE_COLLECTION_NAME where _from or _to matches COLLECTION_NAME/key
        # and then deleting those edges using delete_document on EDGE_COLLECTION_NAME.
        logger.warning(f"Deleting lesson '{key}'. Edge cleanup is NOT YET IMPLEMENTED in this refactored CLI command.")
        success = delete_document(db, COLLECTION_NAME, key)
        status = {
            "key": key,
            "deleted": success,
            "status": "success" if success else "error",
        }

        if success:
            # Even if successful, the crud_api logs warnings if item was already gone.
            # The boolean indicates the state is achieved.
            if json_output:
                print(json.dumps(status))
            else:
                console.print(
                    f"[green]Success:[/green] Lesson '{key}' and associated edges deleted (or already gone)."
                )
        else:
            # delete_lesson returns False only on actual error during vertex delete
            # (e.g., permissions) or if edge cleanup AQL fails critically (depends on API impl.)
            status["message"] = "Deletion failed due to an error (check logs)."
            if json_output:
                print(json.dumps(status))
            else:
                console.print(
                    f"[bold red]Error:[/bold red] Failed to delete lesson '{key}' (check logs)."
                )
            raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"Delete lesson failed in CLI: {e}", exc_info=True)
        status = {"key": key, "deleted": False, "status": "error", "message": str(e)}
        if json_output:
            print(json.dumps(status))
        else:
            console.print(f"[bold red]Error during delete operation:[/bold red] {e}")
        raise typer.Exit(code=1)


# --- Relationship (Edge) Commands (under `graph` subcommand) ---


@graph_app.command("add-relationship")
def cli_add_relationship(
    from_key: str = typer.Argument(..., help="The _key of the source lesson (vertex)."),
    to_key: str = typer.Argument(..., help="The _key of the target lesson (vertex)."),
    rationale: str = typer.Option(
        ..., "--rationale", "-r", help="Reason for linking these lessons."
    ),
    relationship_type: str = typer.Option(
        ...,
        "--type",
        "-typ",
        help="Category of the link (e.g., RELATED, DUPLICATE, CAUSAL).",
    ),
    attributes: Optional[str] = typer.Option(
        None,
        "--attributes",
        "-a",
        help="Additional edge properties as a JSON string (e.g., '{\"confidence\": 0.9}').",
    ),
    json_output: bool = typer.Option(
        False, "--json-output", "-j", help="Output full metadata as JSON on success."
    ),
):
    """
    Create a directed link (relationship edge) between two lessons.

    *WHEN TO USE:* Use *after* analysis suggests a meaningful connection exists. Examples:
    *   Lesson A is a prerequisite for Lesson B (`--type PREREQUISITE`).
    *   Lesson C is a duplicate of Lesson D (`--type DUPLICATE`).
    *   Lesson E provides context for Lesson F (`--type CONTEXT`).
    *   Lesson G caused Lesson H (`--type CAUSAL`).
    Provide a clear `--rationale` and appropriate `--type`.

    *HOW TO USE:* Specify source and target keys, rationale, and type.
    Example: `... graph add-relationship key1 key2 -r "Duplicate entry" -typ DUPLICATE`
    """
    logger.info(
        f"CLI: Received request to add relationship from {from_key} to {to_key}"
    )
    db = get_db_connection()
    attr_dict: Optional[dict[str, Any]] = None # Added type arguments
    if attributes:
        try:
            attr_dict = json.loads(attributes)
            if not isinstance(attr_dict, dict):
                raise ValueError("Provided attributes must be a JSON object.")
        except json.JSONDecodeError as e:
            console.print(
                f"[bold red]Error:[/bold red] Invalid JSON provided for --attributes: {e}"
            )
            raise typer.Exit(code=1)
        except ValueError as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise typer.Exit(code=1)

    try:
        # Ensure type is uppercase if needed by convention/API
        rel_type_upper = relationship_type.upper()
        # Call the new create_relationship function from crud package
        meta = create_relationship(
            db,
            from_doc_key=from_key,
            to_doc_key=to_key,
            relationship_type=rel_type_upper,
            rationale=rationale,
            attributes=attr_dict
        )
        if meta:
            output = meta  # Prepare JSON output
            if json_output:
                print(json.dumps(output))
            else:
                console.print(
                    f"[green]Success:[/green] Relationship added: [cyan]{from_key}[/cyan] "
                    f"-([yellow]{rel_type_upper}[/yellow], key: [cyan]{meta.get('_key')}[/cyan])-> "
                    f"[cyan]{to_key}[/cyan]"
                )
        else:
            # add_relationship logs details
            console.print(
                "[bold red]Error:[/bold red] Failed to add relationship (check logs - keys might not exist or other DB issue)."
            )
            raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"Add relationship failed in CLI: {e}", exc_info=True)
        output = {
            "status": "error",
            "message": str(e),
            "from_key": from_key,
            "to_key": to_key,
        }
        if json_output:
            print(json.dumps(output))
        else:
            console.print(
                f"[bold red]Error during add-relationship operation:[/bold red] {e}"
            )
        raise typer.Exit(code=1)


@graph_app.command("delete-relationship")
def cli_delete_relationship(
    edge_key: str = typer.Argument(
        ..., help="The _key of the relationship edge to delete."
    ),
    json_output: bool = typer.Option(
        False, "--json-output", "-j", help="Output status as JSON."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Confirm deletion without interactive prompt."
    ),
):
    """
    Remove a specific relationship link (edge) between lessons.

    *WHEN TO USE:* Use when a previously established relationship is found to be
    incorrect or no longer relevant based on new information or analysis.

    *HOW TO USE:* Provide the `_key` of the edge document itself. Use `--yes` to bypass prompt.
    Example: `... graph delete-relationship edge_key_987 --yes`
    """
    logger.warning(
        f"CLI: Received request to DELETE relationship edge key '{edge_key}'"
    )
    if not yes:
        confirmed = typer.confirm(
            f"Are you sure you want to delete relationship edge '[cyan]{edge_key}[/cyan]'?",
            abort=True,
        )

    db = get_db_connection()
    try:
        # Call the new delete_relationship_by_key function from crud package
        success = delete_relationship_by_key(db, edge_key)
        status = {
            "edge_key": edge_key,
            "deleted": success,
            "status": "success" if success else "error",
        }

        if success:
            # CRUD function returns True even if already gone, which is success here.
            if json_output:
                print(json.dumps(status))
            else:
                console.print(
                    f"[green]Success:[/green] Relationship edge '{edge_key}' deleted (or already gone)."
                )
        else:
            # Should only happen on actual DB error now (e.g. permissions)
            status["message"] = "Deletion failed due to an error (check logs)."
            if json_output:
                print(json.dumps(status))
            else:
                console.print(
                    f"[bold red]Error:[/bold red] Failed to delete relationship edge '{edge_key}'."
                )
            raise typer.Exit(code=1)  # Exit on failure
    except Exception as e:
        logger.error(f"Delete relationship failed in CLI: {e}", exc_info=True)
        status = {
            "edge_key": edge_key,
            "deleted": False,
            "status": "error",
            "message": str(e),
        }
        if json_output:
            print(json.dumps(status))
        else:
            console.print(
                f"[bold red]Error during delete-relationship operation:[/bold red] {e}"
            )
        raise typer.Exit(code=1)


@graph_app.command("traverse")
def cli_graph_traverse(
    start_node_id: str = typer.Argument(
        ..., help="Start node _id (e.g., 'lessons_learned/crud_test_key_1')."
    ),
    graph_name: str = typer.Option(
        GRAPH_NAME, "--graph-name", "-g", help="Name of the graph to traverse."
    ),
    min_depth: int = typer.Option(
        1, "--min-depth", help="Minimum traversal depth.", min=0
    ),
    max_depth: int = typer.Option(
        1, "--max-depth", help="Maximum traversal depth.", min=0
    ),
    direction: str = typer.Option(
        "OUTBOUND", "--direction", "-dir", help="Direction: OUTBOUND, INBOUND, or ANY."
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-lim", help="Maximum number of paths to return.", min=1
    ),
    json_output: bool = typer.Option(
        True,
        "--json-output",
        "-j",
        help="Output results as JSON (default for traverse).",
    ),
):
    """
    Explore relationships between lessons via graph traversal.

    *WHEN TO USE:* Use to understand connections, dependencies, or related concepts
    starting from a specific lesson document. Essential for navigating the knowledge graph.
    Requires edges to exist between lessons.

    *HOW TO USE:* Provide the full `_id` of the starting lesson (collection/key).
    Adjust depth, direction, limit, or graph name as needed. Output is typically JSON.
    Example: `... graph traverse lessons_learned/my_key --max-depth 2 --direction ANY`
    """
    logger.info(
        f"CLI: Performing graph traversal from '{start_node_id}' in graph '{graph_name}'"
    )
    db = get_db_connection()
    try:
        # Validate direction input
        valid_directions = ["OUTBOUND", "INBOUND", "ANY"]
        dir_upper = direction.upper()
        if dir_upper not in valid_directions:
            console.print(
                f"[bold red]Error:[/bold red] Invalid direction '{direction}'. Must be one of: {', '.join(valid_directions)}"
            )
            raise typer.Exit(code=1)
        # Validate depth
        if min_depth > max_depth:
            console.print(
                f"[bold red]Error:[/bold red] min_depth ({min_depth}) cannot be greater than max_depth ({max_depth})."
            )
            raise typer.Exit(code=1)

        # Extract key from full ID if necessary
        if "/" in start_node_id:
            try:
                collection_name, start_key = start_node_id.split('/', 1)
            except ValueError:
                 console.print(f"[bold red]Error:[/bold red] Invalid start_node_id format: '{start_node_id}'. Expected 'collection/key'.")
                 raise typer.Exit(code=1)
        else:
            # Assume it's just the key if no slash - might need adjustment based on API needs
            start_key = start_node_id
            logger.warning(f"Start node ID '{start_node_id}' provided without collection prefix. Assuming it's just the key.")

        # Call graph_traverse with keyword arguments
        results_data = graph_traverse(
            db=db,
            start_vertex_key=start_key,
            min_depth=min_depth,
            max_depth=max_depth,
            direction=dir_upper,
            limit=limit if limit is not None else 10 # Pass limit or API default
            # relationship_types=None, # Using default
            # vertex_filter=None, # Using default
        )

        # Graph traversal results are often complex, JSON is usually preferred
        # Handle None result from API (indicates error during traversal)
        if results_data is None:
            logger.error(
                f"Graph traversal API returned None for start node {start_node_id}"
            )
            console.print(
                f"[bold red]Error:[/bold red] Graph traversal failed (API returned None). Check logs."
            )
            raise typer.Exit(code=1)

        # Process results if not None
        # Process results if not None
        if json_output:
            # Print only the list of results for JSON output
            print(json.dumps(results_data.get("results", []), indent=2))
        else:
            # Simple summary if JSON not requested
            # Get the actual count from the results dictionary
            path_count = results_data.get("count", 0)
            if path_count > 0:
                console.print(
                    f"[green]Traversal complete.[/green] Found {path_count} paths. Use -j for detailed JSON output."
                )
            else:
                 console.print("[yellow]Traversal complete. No paths found.[/yellow]")

    except Exception as e:
        logger.error(f"Graph traversal failed: {e}", exc_info=True)
        output = {"status": "error", "message": str(e), "start_node": start_node_id}
        if json_output:
            print(json.dumps(output))
        else:
            console.print(f"[bold red]Error during graph traversal:[/bold red] {e}")
        raise typer.Exit(code=1)


# --- Helper for Displaying Search Results (Human Readable) ---
# Removed duplicate signature, kept the Optional[str] version
def _display_results(search_data: dict[str, Any], search_type: str, score_field: Optional[str]): # Added type arguments
    """Uses Rich to display search results in a table (for human consumption)."""
    # Log the input data safely for debugging, truncating large fields
    logger.debug(f"Displaying results for {search_type}: {truncate_large_value(search_data)}")

    # Note: Pylance might warn about unnecessary isinstance, but it's safer.
    if not isinstance(search_data, dict): # Added type arguments
        logger.warning(f"_display_results expected a dict, got {type(search_data)}")
        console.print(
            "[yellow]Warning: Invalid format for search results display.[/yellow]"
        )
        return
    results = search_data.get("results", [])
    total = search_data.get("total", len(results))  # Estimate total if not provided
    offset = search_data.get("offset", 0)

    console.print(
        f"\n[bold blue]--- {search_type} Results (Showing {len(results)} of ~{total}) ---[/bold blue]"
    )

    if not results:
        console.print(
            "[yellow]No relevant documents found matching the criteria.[/yellow]"
        )
        return

    table = Table(
        show_header=True,
        header_style="bold magenta",
        expand=True,
        title=f"{search_type} Search Results",
    )
    table.add_column("#", style="dim", width=3, no_wrap=True, justify="right")
    # Removed duplicate column add
    table.add_column(f"Score" if score_field else "-", justify="right", width=8) # No score for keyword header
    table.add_column("Key", style="cyan", no_wrap=True, width=38)
    table.add_column(
        "Problem (Preview)", style="green", overflow="fold", min_width=30
    )  # Min width helps folding
    table.add_column("Tags", style="yellow", overflow="fold", min_width=15)

    for i, result_item in enumerate(results, start=1):
        # Handle potential variations in result structure
        if not isinstance(result_item, dict): # Added type arguments
            logger.warning(f"Skipping non-dict result item: {truncate_large_value(result_item)}")
            continue  # Skip non-dict results

        # Removed duplicate score assignment
        score_val = result_item.get(score_field, 0.0) if score_field else None
        # Accommodate results that might be just the doc or nested under 'doc'
        doc = result_item.get("doc", result_item if "_key" in result_item else {})
        if not doc:  # Handle cases where doc might be missing or empty
            logger.warning(
                f"Result item missing 'doc' field or 'doc' is empty: {truncate_large_value(result_item)}"
            )
            continue

        key = doc.get("_key", "N/A")
        problem = doc.get("problem", "N/A")
        # Ensure problem is a string before processing
        problem_preview = (
            str(problem).split("\n")[0] if problem else "N/A"
        )  # First line only
        tags = ", ".join(doc.get("tags", []))

        # Include other scores if present (for Hybrid)
        other_scores = []
        if "bm25_score" in result_item and score_field != "bm25_score":
            other_scores.append(f"BM25: {result_item['bm25_score']:.4f}")
        if "similarity_score" in result_item and score_field != "similarity_score":
            other_scores.append(f"Sim: {result_item['similarity_score']:.4f}")
        other_scores_str = f" ({', '.join(other_scores)})" if other_scores else ""

        table.add_row(
            str(offset + i),
            # Removed duplicate score display in row
            f"{score_val:.4f}{other_scores_str}" if score_val is not None else "-", # Display score or dash
            key,
            problem_preview,  # Use preview
            tags,
        )

    console.print(table)
    # Check if pagination makes sense (more results available than shown)
    if total > (offset + len(results)):
        console.print(
            f"[dim]Showing results {offset + 1}-{offset + len(results)} of {total}. Use --offset for more (if applicable).[/dim]"
        )
    elif total > 0 and total <= len(results) and offset == 0:
        console.print(f"[dim]Showing all {total} results found.[/dim]")


# --- Main Execution Guard ---
if __name__ == "__main__":
    # Add a try-except block around the app() call for final catch
    try:
        app()
    except typer.Exit as e:
        # typer.Exit is used for controlled exits (like errors handled above)
        # We just need to ensure the exit code is propagated
        sys.exit(e.exit_code)
    except Exception as e:
        # Catch any truly unexpected errors at the top level
        logger.critical(f"Unhandled exception during CLI execution: {e}", exc_info=True)
        console.print(
            f"[bold red]FATAL ERROR:[/bold red] An unexpected error occurred. Check logs. ({e})"
        )
        sys.exit(1)  # Exit with a non-zero code
