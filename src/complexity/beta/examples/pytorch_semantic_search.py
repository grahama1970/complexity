#!/usr/bin/env python3
"""
Example of using the PyTorch-based semantic search functionality with filters.

This demonstrates how to use the enhanced semantic search capabilities to
find similar documents with complex filtering that ArangoDB's built-in
APPROX_NEAR_COSINE function may not support.

Usage:
    python pytorch_semantic_search_example.py [--filter "doc.topic == 'physics'"]
"""

import sys
import argparse
from loguru import logger
from complexity.beta.utils.arango_setup import connect_arango, ensure_database
from complexity.beta.utils.relationship_builder import RelationshipBuilder

def main():
    # Set up logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run PyTorch-based semantic search with optional filters")
    parser.add_argument("--filter", type=str, help="AQL filter condition (e.g. 'doc.topic == \"physics\"')")
    parser.add_argument("--threshold", type=float, default=0.8, help="Similarity threshold (0.0-1.0)")
    args = parser.parse_args()
    
    # Connect to ArangoDB
    logger.info("Connecting to ArangoDB...")
    client = connect_arango()
    db = ensure_database(client)
    logger.info(f"Connected to database: {db.name}")
    
    # Initialize relationship builder
    builder = RelationshipBuilder(db)
    
    # Run semantic search with filters
    logger.info("Starting semantic relationship search...")
    if args.filter:
        logger.info(f"Applying filter: {args.filter}")
        
    count = builder.build_semantic_relationships(
        threshold=args.threshold,
        filter_conditions=args.filter
    )
    
    logger.info(f"Created {count} new relationships")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())