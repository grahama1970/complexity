#!/usr/bin/env python3
"""
Command-Line Interface (CLI) for Complexity with Embedding Support

This is a modified version of cli.py that uses enhanced database operations
with automatic embedding generation.
"""

import typer
import json
import sys
import uuid
import time
import os
from pathlib import Path
from typing import List, Optional, Any, Dict, Union
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.json import JSON

# Import from original CLI
from complexity.cli import (
    app, search_app, db_app, graph_app, console,
    main_callback, get_db_connection, display_results,
    search_hybrid, search_semantic, search_bm25, search_tag, search_keyword,
    db_read, db_delete, graph_traverse
)

# Don't import graph_add_edge and graph_delete_edge since we will override these

# Define our own initialize_database function with uuid support
@app.command("init")
def initialize_database(
    create_collections: bool = typer.Option(
        True, "--create-collections", "-c", help="Create required collections"
    ),
    create_sample_data: bool = typer.Option(
        True, "--create-sample-data", "-s", help="Create sample data for testing"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force recreation of collections even if they exist"
    )
):
    """
    Initialize ArangoDB with collections and sample data for testing the CLI.
    
    *WHEN TO USE:* Use before running tests to ensure your database has the required
    structure and sample data for testing all CLI commands.
    
    *HOW TO USE:* Run with default options to create all necessary collections and
    sample data, or customize with flags.
    
    *EXAMPLES:*
    
    - Basic initialization with collections and sample data:
      ```
      init
      ```
      
    - Create only collections without sample data:
      ```
      init --no-create-sample-data
      ```
      
    - Force recreation of collections (will drop existing):
      ```
      init --force
      ```
    """
    logger.info("Initializing database for CLI testing")
    
    # Get database connection
    db = get_db_connection()
    
    # Define required collections
    required_collections = [
        "test_docs",          # For database CRUD operations
        "messages",           # For search operations
        "relationships",      # For graph operations
        "test_relationships"  # For graph tests
    ]
    
    # Create collections if requested
    if create_collections:
        for collection_name in required_collections:
            collection_exists = db.has_collection(collection_name)
            
            if collection_exists and force:
                logger.info(f"Dropping existing collection '{collection_name}'")
                db.delete_collection(collection_name)
                collection_exists = False
            
            if not collection_exists:
                is_edge = "relationship" in collection_name.lower()
                logger.info(f"Creating {'edge ' if is_edge else ''}collection '{collection_name}'")
                db.create_collection(collection_name, edge=is_edge)
                console.print(f"[green]Created {'edge ' if is_edge else ''}collection:[/green] [cyan]{collection_name}[/cyan]")
            else:
                console.print(f"[yellow]Collection already exists:[/yellow] [cyan]{collection_name}[/cyan]")
    
    # Create sample data if requested
    if create_sample_data:
        console.print("[bold cyan]Creating sample data...[/bold cyan]")
        
        # Create sample documents for search testing
        for i in range(1, 6):
            doc_key = f"python_error_{i}_{uuid.uuid4().hex[:8]}"
            
            # Check if document exists
            if db.collection("messages").has(doc_key) and not force:
                console.print(f"[yellow]Sample document already exists:[/yellow] [cyan]{doc_key}[/cyan]")
                continue
            
            # Create sample document
            doc = {
                "_key": doc_key,
                "question": f"What is Python error #{i}?",
                "answer": f"This is a sample document about Python error #{i}.",
                "tags": ["python", "error", f"error-{i}"],
                "created_at": time.time()
            }
            
            # Import embedding-aware document creation
            from complexity.arangodb.embedded_db_operations import create_document_with_embedding
            
            # Insert document with embedding
            result = create_document_with_embedding(db, "messages", doc)
            if result:
                console.print(f"[green]Created sample document with embedding:[/green] [cyan]{doc_key}[/cyan]")
            else:
                console.print(f"[red]Failed to create sample document:[/red] [cyan]{doc_key}[/cyan]")
        
        # Create sample edge
        if len(list(db.collection("messages").all())) >= 2:
            # Get first two document IDs
            doc_keys = [doc["_key"] for doc in db.collection("messages").all(limit=2)]
            
            # Create edge
            edge = {
                "_from": f"messages/{doc_keys[0]}",
                "_to": f"messages/{doc_keys[1]}",
                "type": "RELATED_TO",
                "weight": 0.8
            }
            
            # Insert edge
            db.collection("relationships").insert(edge)
            console.print(f"[green]Created sample edge:[/green] [cyan]{doc_keys[0]}[/cyan] -> [cyan]{doc_keys[1]}[/cyan]")
    
    console.print("[bold green]✓ Database initialization completed![/bold green]")

# Override database commands to use embedding-enhanced operations
@db_app.command("create", help="Create a new document with auto-generated embedding.")
def db_create(
    collection: str = typer.Option(
        ..., "--collection", "-c", help="Name of the collection to add document to."
    ),
    data: Optional[str] = typer.Option(
        None, "--data", "-d", help="Document data as JSON string."
    ),
    data_file: Optional[Path] = typer.Option(
        None, "--data-file", "-f", help="Path to JSON file containing document data.",
        exists=True, file_okay=True, dir_okay=False, readable=True,
    ),
    json_output: bool = typer.Option(
        False, "--json-output", "-j", help="Output metadata as JSON on success."
    ),
):
    """
    Create a new document in a collection with automatic embedding generation.

    *WHEN TO USE:* Use when you need to add a new document to a collection.

    *HOW TO USE:* Provide the collection name and document data either as a JSON string
    or by referencing a JSON file.
    """
    logger.info(f"CLI: Creating document in collection '{collection}'")
    
    # Input validation: Ensure exactly one data source
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
    
    # Load document data
    document_data = None
    source_info = ""
    
    try:
        if data_file:
            source_info = f"file '{data_file}'"
            logger.debug(f"Loading document data from file: {data_file}")
            with open(data_file, "r") as f:
                document_data = json.load(f)
        elif data:
            source_info = "string --data"
            document_data = json.loads(data)
            logger.debug("Loaded document data from string")
        
        # Validate document data
        if not isinstance(document_data, dict):
            raise ValueError("Provided data must be a JSON object (dictionary).")
    
    except json.JSONDecodeError as e:
        console.print(
            f"[bold red]Error:[/bold red] Invalid JSON provided via {source_info}: {e}"
        )
        raise typer.Exit(code=1)
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(
            f"[bold red]Error reading/parsing data from {source_info}:[/bold red] {e}"
        )
        raise typer.Exit(code=1)
    
    # Get database connection
    db = get_db_connection()
    
    try:
        # Import create_document_with_embedding function
        from complexity.arangodb.embedded_db_operations import create_document_with_embedding
        
        # Create document with embedding
        meta = create_document_with_embedding(db, collection, document_data)
        
        if meta:
            if json_output:
                print(json.dumps(meta, indent=2))
            else:
                console.print(
                    f"[green]Success:[/green] Document added to collection '{collection}'. Key: [cyan]{meta.get('_key')}[/cyan]"
                )
                
                # Check if embedding was generated
                if "embedding" in meta:
                    embedding_len = len(meta["embedding"])
                    console.print(
                        f"[green]Embedding:[/green] Generated embedding with [cyan]{embedding_len}[/cyan] dimensions."
                    )
                else:
                    console.print(
                        f"[yellow]Note:[/yellow] No embedding was generated for this document."
                    )
        else:
            console.print(
                "[bold red]Error:[/bold red] Failed to add document (check logs for details)."
            )
            raise typer.Exit(code=1)
    
    except Exception as e:
        logger.error(f"Document creation failed: {e}", exc_info=True)
        console.print(f"[bold red]Error during create operation:[/bold red] {e}")
        raise typer.Exit(code=1)


@db_app.command("update", help="Update a document with auto-generated embedding.")
def db_update(
    key: str = typer.Argument(..., help="The _key of the document to update."),
    collection: str = typer.Option(
        ..., "--collection", "-c", help="Name of the collection containing the document."
    ),
    data: Optional[str] = typer.Option(
        None, "--data", "-d", help="Update data as JSON string."
    ),
    data_file: Optional[Path] = typer.Option(
        None, "--data-file", "-f", help="Path to JSON file containing update data.",
        exists=True, file_okay=True, dir_okay=False, readable=True,
    ),
    json_output: bool = typer.Option(
        False, "--json-output", "-j", help="Output metadata as JSON on success."
    ),
):
    """
    Update an existing document with automatic embedding regeneration.

    *WHEN TO USE:* Use to modify fields in an existing document.

    *HOW TO USE:* Provide the document key, collection name, and update data
    either as a JSON string or by referencing a JSON file.
    """
    logger.info(f"CLI: Updating document '{key}' in collection '{collection}'")
    
    # Input validation: Ensure exactly one data source
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
    
    # Load update data
    update_data = None
    source_info = ""
    
    try:
        if data_file:
            source_info = f"file '{data_file}'"
            logger.debug(f"Loading update data from file: {data_file}")
            with open(data_file, "r") as f:
                update_data = json.load(f)
        elif data:
            source_info = "string --data"
            update_data = json.loads(data)
            logger.debug("Loaded update data from string")
        
        # Validate update data
        if not isinstance(update_data, dict):
            raise ValueError("Provided update data must be a JSON object (dictionary).")
        if not update_data:
            raise ValueError("Update data cannot be empty.")
    
    except json.JSONDecodeError as e:
        console.print(
            f"[bold red]Error:[/bold red] Invalid JSON provided via {source_info}: {e}"
        )
        raise typer.Exit(code=1)
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(
            f"[bold red]Error reading/parsing update data from {source_info}:[/bold red] {e}"
        )
        raise typer.Exit(code=1)
    
    # Get database connection
    db = get_db_connection()
    
    try:
        # Import update_document_with_embedding function
        from complexity.arangodb.embedded_db_operations import update_document_with_embedding
        
        # Update document with embedding
        meta = update_document_with_embedding(db, collection, key, update_data)
        
        if meta:
            if json_output:
                print(json.dumps(meta, indent=2))
            else:
                console.print(
                    f"[green]Success:[/green] Document [cyan]{key}[/cyan] in collection '{collection}' updated successfully."
                )
                
                # Check if embedding was updated
                if "embedding" in meta and "content" in update_data:
                    embedding_len = len(meta["embedding"])
                    console.print(
                        f"[green]Embedding:[/green] Updated embedding with [cyan]{embedding_len}[/cyan] dimensions."
                    )
        else:
            console.print(
                f"[bold red]Error:[/bold red] Failed to update document '{key}' in collection '{collection}' (check logs for details)."
            )
            raise typer.Exit(code=1)
    
    except Exception as e:
        logger.error(f"Document update failed: {e}", exc_info=True)
        console.print(f"[bold red]Error during update operation:[/bold red] {e}")
        raise typer.Exit(code=1)


# --- Add the fixed graph operations ---

@graph_app.command("add-edge")
def graph_add_edge(
    from_key: str = typer.Argument(..., help="The _key of the source document."),
    to_key: str = typer.Argument(..., help="The _key of the target document."),
    collection: str = typer.Option(
        ..., "--collection", "-c", help="Name of the document collection."
    ),
    edge_collection: str = typer.Option(
        ..., "--edge-collection", "-e", help="Name of the edge collection."
    ),
    edge_type: str = typer.Option(
        ..., "--type", "-t", help="Type of the relationship."
    ),
    rationale: str = typer.Option(
        ..., "--rationale", "-r", help="Reason for linking these documents."
    ),
    attributes: Optional[str] = typer.Option(
        None, "--attributes", "-a", help="Additional edge properties as JSON string."
    ),
    json_output: bool = typer.Option(
        False, "--json-output", "-j", help="Output metadata as JSON on success."
    ),
):
    """
    Create a relationship between two documents.

    *WHEN TO USE:* Use to establish a connection between documents in a graph.

    *HOW TO USE:* Provide the source and target document keys, collection names,
    relationship type, and rationale for the connection.
    """
    logger.info(f"CLI: Creating edge from '{from_key}' to '{to_key}' in collection '{edge_collection}'")
    
    # Parse additional attributes if provided
    attr_dict = None
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
    
    # Get database connection
    db = get_db_connection()
    
    try:
        # Import enhanced relationship function
        from complexity.arangodb.enhanced_relationships import create_edge_from_cli
        
        # Create relationship using the enhanced function
        rel_type_upper = edge_type.upper()
        meta = create_edge_from_cli(
            db=db,
            from_key=from_key,
            to_key=to_key,
            collection=collection,
            edge_collection=edge_collection,
            edge_type=rel_type_upper,
            rationale=rationale,
            attributes=attr_dict
        )
        
        if meta:
            if json_output:
                print(json.dumps(meta, indent=2))
            else:
                console.print(
                    f"[green]Success:[/green] Relationship added: [cyan]{from_key}[/cyan] "
                    f"-([yellow]{rel_type_upper}[/yellow], key: [cyan]{meta.get('_key')}[/cyan])-> "
                    f"[cyan]{to_key}[/cyan]"
                )
        else:
            console.print(
                "[bold red]Error:[/bold red] Failed to add relationship (check logs - keys might not exist or other DB issue)."
            )
            raise typer.Exit(code=1)
            
    except Exception as e:
        logger.error(f"Edge creation failed: {e}", exc_info=True)
        console.print(f"[bold red]Error during edge creation:[/bold red] {e}")
        raise typer.Exit(code=1)

@graph_app.command("delete-edge")
def graph_delete_edge(
    edge_key: str = typer.Argument(..., help="The _key of the edge to delete."),
    edge_collection: str = typer.Option(
        ..., "--edge-collection", "-e", help="Name of the edge collection."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Confirm deletion without interactive prompt."
    ),
    json_output: bool = typer.Option(
        False, "--json-output", "-j", help="Output status as JSON."
    ),
):
    """
    Remove a relationship between documents.

    *WHEN TO USE:* Use to remove a connection between documents in a graph.

    *HOW TO USE:* Provide the edge document key and edge collection name. Use --yes
    to bypass the confirmation prompt.
    """
    logger.info(f"CLI: Deleting edge '{edge_key}' from collection '{edge_collection}'")
    
    # Confirmation prompt if --yes not provided
    if not yes:
        confirmed = typer.confirm(
            f"Are you sure you want to delete edge '[cyan]{edge_key}[/cyan]' from collection '{edge_collection}'?",
            abort=True,
        )
    
    # Get database connection
    db = get_db_connection()
    
    try:
        # Import enhanced relationship function
        from complexity.arangodb.enhanced_relationships import delete_edge_from_cli
        
        # Delete edge using the enhanced function
        success = delete_edge_from_cli(db, edge_key, edge_collection)
        
        status = {
            "edge_key": edge_key,
            "edge_collection": edge_collection,
            "deleted": success,
            "status": "success" if success else "error",
        }
        
        if success:
            if json_output:
                print(json.dumps(status, indent=2))
            else:
                console.print(
                    f"[green]Success:[/green] Edge '{edge_key}' from collection '{edge_collection}' deleted (or already gone)."
                )
        else:
            status["message"] = "Deletion failed due to an error (check logs)."
            if json_output:
                print(json.dumps(status, indent=2))
            else:
                console.print(
                    f"[bold red]Error:[/bold red] Failed to delete edge '{edge_key}' from collection '{edge_collection}'."
                )
            raise typer.Exit(code=1)
            
    except Exception as e:
        logger.error(f"Edge deletion failed: {e}", exc_info=True)
        status = {"edge_key": edge_key, "edge_collection": edge_collection, "deleted": False, "status": "error", "message": str(e)}
        if json_output:
            print(json.dumps(status, indent=2))
        else:
            console.print(f"[bold red]Error during edge deletion:[/bold red] {e}")
        raise typer.Exit(code=1)

# --- Main Execution ---
if __name__ == "__main__":
    try:
        app()
    except typer.Exit as e:
        sys.exit(e.exit_code)
    except Exception as e:
        logger.critical(f"Unhandled exception during CLI execution: {e}", exc_info=True)
        console.print(
            f"[bold red]FATAL ERROR:[/bold red] An unexpected error occurred. Check logs. ({e})"
        )
        sys.exit(1)