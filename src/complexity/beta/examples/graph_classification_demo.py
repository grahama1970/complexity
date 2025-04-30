#!/usr/bin/env python3
"""
Module Description:
Demo script to showcase the graph-enhanced classification system.
This script connects to ArangoDB, ensures necessary collections and graphs exist,
builds relationships between questions using RelationshipBuilder, and then classifies
a set of sample questions using the enhanced_classify_complexity function,
displaying the results and explanations.

Links:
- python-arango Driver: https://python-arango.readthedocs.io/en/latest/
- Loguru: https://loguru.readthedocs.io/en/stable/
- Tabulate: https://github.com/astanin/python-tabulate

Sample Input/Output:

- Running the script: `uv run python -m src.complexity.beta.examples.graph_classification_demo` # Adjusted path
- Expected Output:
  - Log messages indicating setup steps.
  - Log messages for each classification, including neighbors and explanation.
  - A summary table printed to stdout with classification results.
  - Final message "✅ VALIDATION COMPLETE - Demo script executed successfully." and exit code 0.
  - On error, "❌ VALIDATION FAILED - Demo script encountered an error." and exit code 1.
"""

import sys
import os
from loguru import logger
# Adjust import paths based on the new location (one level up from utils)
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
from typing import List, Dict, Any # Import necessary types

# Sample questions for demonstration
SAMPLE_QUESTIONS: List[str] = [
    "What is the capital of France?",
    "Explain the process of photosynthesis in detail",
    "How does quantum entanglement work?",
    "What are the primary colors?",
    "Describe the structure of DNA and how it replicates"
]

# Global variable to store results for validation
results_table_data: List[List[Any]] = []

def process_results(results_table_data: List[List[Any]], success_count: int) -> bool:
    """Process and validate the classification results."""
    # Print summary table
    print("\nClassification Results with Graph-Enhanced Explanations:\n")
    print(tabulate(results_table_data, headers=["Question", "Classification", "Confidence", "Explanation"], tablefmt="grid"))
    
    logger.info("Processing results completed")
    return success_count > 0

def main() -> bool:
    """Demonstrate graph-enhanced classification system."""
    global results_table_data
    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} | {level:<8} | {message}", level="DEBUG")

    try:
        # Database setup with enhanced error handling
        logger.info("Setting up ArangoDB connection")
        client = connect_arango()
        if not client:
            raise RuntimeError("Failed to establish ArangoDB connection")
        
        logger.info("Ensuring database exists")
        db = ensure_database(client)
        if not db:
            raise RuntimeError("Failed to ensure database exists")
        
        logger.info("Setting up collections and graph")
        ensure_collection(db)
        ensure_edge_collections(db)
        ensure_graph(db)

        # Initialize relationship builder with enhanced error handling
        logger.info("Initializing RelationshipBuilder...")
        try:
            relationship_builder = RelationshipBuilder(db)
        except Exception as e:
            logger.error(f"Failed to initialize RelationshipBuilder: {e}")
            logger.debug(f"Full error context: {str(e)}", exc_info=True)
            raise RuntimeError("Failed to initialize relationship builder") from e

    # Generate semantic and prerequisite relationships
    logger.info("Building relationships between questions (this may take time and use LLM calls)...")
    relationship_count = relationship_builder.generate_all_relationships()
    logger.info(f"Generated/verified {relationship_count} relationships")

    # Clear any previous results
    results_table_data.clear()
        # Process each sample question
        success_count = 0
        for question in SAMPLE_QUESTIONS:
            logger.info(f"Processing question: '{question}'")
            result = enhanced_classify_complexity(db, question, with_explanation=True)
            if not result:
                logger.error(f"Classification failed for question: {question}")
                continue

        # Safely extract data for the results table
        classification = result.get("classification", "Error")
        confidence = result.get("confidence", 0.0)
        explanation_data = result.get("explanation", {})
        explanation_text = "N/A"
        if isinstance(explanation_data, dict):
            explanation_text = explanation_data.get("explanation", "N/A")

        # Add to results table
        row: List[Any] = [
            question,
            classification,
            f"{confidence:.2f}",
            explanation_text
        ]
        results_table_data.append(row)

        # Show detailed neighbors
        logger.info("Top similar questions:")
        neighbors: List[Dict[str, Any]] = result.get("neighbors", [])
        # No need to check isinstance since we know the type from type hints
        for i, neighbor in enumerate(neighbors):
            neighbor_question = neighbor.get('question', 'N/A')
            neighbor_label = neighbor.get('label', 'N/A')
            neighbor_score = neighbor.get('score', 0.0)
            logger.info(f"  {i+1}. {neighbor_question} ({neighbor_label}, score: {neighbor_score:.3f})")


        # Show explanation details if available
        if explanation_data:  # Just check if we have explanation data
            logger.info(f"Explanation: {explanation_text}")

            # Get rationales once for both prerequisites and related topics
            rationales = explanation_data.get("rationales", [])

            # Show prerequisites if present
            prerequisites = explanation_data.get("prerequisites", [])
            if prerequisites:  # Just check if we have prerequisites
                prereq_table_data: List[List[Any]] = []
                for i, p in enumerate(prerequisites):
                    rationale = rationales[i] if i < len(rationales) else ""
                    prereq_question = p.get("question", "N/A")
                    prereq_distance = p.get("distance", "N/A")
                    prereq_table_data.append([prereq_question, f"Distance: {prereq_distance}", rationale])

                if prereq_table_data:
                    print("\nPrerequisites:")
                    print(tabulate(prereq_table_data, headers=["Question", "Distance", "Rationale"], tablefmt="simple"))

            # Show related topics if present
            related_table_data: List[List[Any]] = []  # Initialize outside try block
            try:
                related_topics = explanation_data.get("related_topics", [])
                if related_topics:  # Just check if we have related topics
                    # Ensure we get fresh rationales in case they changed
                    related_rationales = explanation_data.get("rationales", [])
                    for i, r in enumerate(related_topics):
                        try:
                            rationale = related_rationales[i] if i < len(related_rationales) else ""
                            related_question = r.get("question", "N/A")
                            related_distance = r.get("distance", "N/A")
                            related_table_data.append([related_question, f"Distance: {related_distance}", rationale])
                        except Exception as e:
                            logger.warning(f"Error processing related topic {i}: {e}")
                            continue

                if related_table_data:  # Moved outside the if related_topics block
                    print("\nRelated Topics:")
                    print(tabulate(related_table_data, headers=["Question", "Distance", "Rationale"], tablefmt="simple"))
            except Exception as e:
                logger.error(f"Error displaying related topics: {e}")

                if related_table_data:
                    print("\nRelated Topics:")
                    print(tabulate(related_table_data, headers=["Question", "Distance", "Rationale"], tablefmt="simple"))
                    

        print("\n" + "-" * 80 + "\n")

    # Print summary table
    print("\nClassification Results with Graph-Enhanced Explanations:\n")
    print(tabulate(results_table_data, headers=["Question", "Classification", "Confidence", "Explanation"], tablefmt="grid"))

    logger.info("Demo completed successfully")

if __name__ == "__main__":
    validation_passed = True
    validation_failures: Dict[str, Any] = {}
    results_table_data = []

    # Define expected results for validation
    EXPECTED_RESULTS = {
        "sample_questions": len(SAMPLE_QUESTIONS),
        "min_confidence": 0.4,  # Minimum expected confidence for classifications
        "expected_labels": ["Simple", "Complex", "Complex", "Simple", "Complex"],  # Expected classifications
    }

    try:
        main()
        
        # Validate the results
        if not results_table_data:
            validation_passed = False
            validation_failures["results"] = "No classification results generated"
        else:
            # Verify number of questions processed
            if len(results_table_data) != EXPECTED_RESULTS["sample_questions"]:
                validation_passed = False
                validation_failures["question_count"] = {
                    "expected": EXPECTED_RESULTS["sample_questions"],
                    "actual": len(results_table_data)
                }

            # Verify classifications and confidence scores
            for i, row in enumerate(results_table_data):
                question, classification, confidence_str, _ = row
                confidence = float(confidence_str)
                
                # Check confidence threshold
                if confidence < EXPECTED_RESULTS["min_confidence"]:
                    validation_passed = False
                    validation_failures[f"confidence_{i}"] = {
                        "expected": f">= {EXPECTED_RESULTS['min_confidence']}",
                        "actual": confidence
                    }

                # Check classification matches expected
                if classification != EXPECTED_RESULTS["expected_labels"][i]:
                    validation_passed = False
                    validation_failures[f"classification_{i}"] = {
                        "expected": EXPECTED_RESULTS["expected_labels"][i],
                        "actual": classification
                    }

            success_count += 1
            # Add result to results table
            classification = result.get("classification", "Error")
            confidence = result.get("confidence", 0.0)
            explanation = result.get("explanation", {}).get("explanation", "N/A")
            results_table_data.append([question, classification, f"{confidence:.2f}", explanation])

        # Process and print results
        success = process_results(results_table_data, success_count)
        logger.info("Demo completed successfully" if success else "Demo completed with some failures")
        return success

    except Exception as e:
        logger.error(f"Demo failed with runtime error: {e}")
        logger.debug("Full error context:", exc_info=True)
        validation_failures["runtime_error"] = {
            "error": str(e),
            "type": type(e).__name__,
            "context": str(getattr(e, "__context__", "No context available"))
        }
        return False

    # --- Final Reporting ---
    if validation_passed:
        print("✅ VALIDATION COMPLETE - Demo script executed successfully with expected results.")
        logger.success("Standalone execution and validation successful.")
        sys.exit(0)
    else:
        print("❌ VALIDATION FAILED - Issues detected during demo execution or validation.")
        print("FAILURE DETAILS:")
        for field, details in validation_failures.items():
            if isinstance(details, dict):
                 print(f"  - {field}: Expected: {details.get('expected', 'N/A')}, Got: {details.get('actual', 'N/A')}")
            else:
                 print(f"  - {field}: {details}")
        logger.error("Standalone execution and validation failed.")
        sys.exit(1)