#!/usr/bin/env python3
"""
Demo script to showcase the graph-enhanced classification system.
This demonstrates how the system uses both semantic similarity and
graph relationships to classify questions and provide explanations.
"""

import sys
import os
from loguru import logger
from complexity.beta.utils.arango_setup import (
    connect_arango, 
    ensure_database, 
    ensure_collection, 
    ensure_edge_collections,
    ensure_graph
)
from complexity.beta.utils.relationship_builder import RelationshipBuilder
from complexity.beta.utils.enhanced_classifier import enhanced_classify_complexity
from tabulate import tabulate

# Sample questions for demonstration
SAMPLE_QUESTIONS = [
    "What is the capital of France?",
    "Explain the process of photosynthesis in detail",
    "How does quantum entanglement work?",
    "What are the primary colors?",
    "Describe the structure of DNA and how it replicates"
]

def main():
    """Demonstrate graph-enhanced classification system."""
    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} | {level:<8} | {message}", level="INFO")
    
    logger.info("Setting up ArangoDB and ensuring collections exist")
    client = connect_arango()
    db = ensure_database(client)
    ensure_collection(db)
    ensure_edge_collections(db)
    ensure_graph(db)
    
    # Initialize relationship builder
    relationship_builder = RelationshipBuilder(db)
    
    # Generate semantic and prerequisite relationships
    logger.info("Building relationships between questions")
    relationship_count = relationship_builder.generate_all_relationships()
    logger.info(f"Generated {relationship_count} relationships")
    
    # Classify each sample question
    results = []
    for question in SAMPLE_QUESTIONS:
        logger.info(f"Classifying: '{question}'")
        
        # Get enhanced classification with explanation
        result = enhanced_classify_complexity(db, question, with_explanation=True)
        
        # Add to results table
        row = [
            question,
            result["classification"],
            f"{result['confidence']:.2f}",
            result.get("explanation", {}).get("explanation", "N/A")
        ]
        results.append(row)
        
        # Show detailed neighbors
        logger.info("Top similar questions:")
        for i, neighbor in enumerate(result["neighbors"]):
            logger.info(f"  {i+1}. {neighbor['question']} ({neighbor['label']}, score: {neighbor['score']:.3f})")
        
        # Show explanation if available
        if "explanation" in result:
            explanation = result["explanation"]
            logger.info(f"Explanation: {explanation.get('explanation')}")
            
            # Show prerequisites if present
            if "prerequisites" in explanation:
                prereq_table = []
                for i, p in enumerate(explanation["prerequisites"]):
                    rationale = ""
                    if "rationales" in explanation and i < len(explanation["rationales"]):
                        rationale = explanation["rationales"][i]
                    prereq_table.append([p["question"], f"Distance: {p['distance']}", rationale])
                
                if prereq_table:
                    print("\nPrerequisites:")
                    print(tabulate(prereq_table, headers=["Question", "Distance", "Rationale"], tablefmt="simple"))
            
            # Show related topics if present
            if "related_topics" in explanation:
                related_table = []
                for i, r in enumerate(explanation["related_topics"]):
                    rationale = ""
                    if "rationales" in explanation and i < len(explanation["rationales"]):
                        rationale = explanation["rationales"][i]
                    related_table.append([r["question"], f"Distance: {r['distance']}", rationale])
                
                if related_table:
                    print("\nRelated Topics:")
                    print(tabulate(related_table, headers=["Question", "Distance", "Rationale"], tablefmt="simple"))
        
        print("\n" + "-" * 80 + "\n")
    
    # Print summary table
    print("\nClassification Results with Graph-Enhanced Explanations:\n")
    print(tabulate(results, headers=["Question", "Classification", "Confidence", "Explanation"], tablefmt="grid"))
    
    logger.info("Demo completed successfully")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Demo failed: {e}")
        sys.exit(1)