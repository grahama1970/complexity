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

- Running the script: `uv run python src/complexity/beta/utils/graph_classification_demo.py`
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

# Expected results for validation
EXPECTED_RESULTS = {
    "question_count": 5,
    "min_confidence": 0.5,
    "expected_classifications": {
        "What is the capital of France?": "Basic",
        "Explain the process of photosynthesis in detail": "Intermediate",
        "How does quantum entanglement work?": "Advanced",
        "What are the primary colors?": "Basic",
        "Describe the structure of DNA and how it replicates": "Intermediate"
    }
}

# Sample questions for demonstration
SAMPLE_QUESTIONS = list(EXPECTED_RESULTS["expected_classifications"].keys())

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
    validation_passed = True
    validation_failures = {}
    actual_results = {
        "classifications": {},
        "confidences": []
    }
    
    try:
        # Run main function and capture results
        logger.info("Starting demo execution")
        results = []
        
        client = connect_arango()
        db = ensure_database(client)
        ensure_collection(db)
        ensure_edge_collections(db)
        ensure_graph(db)
        
        relationship_builder = RelationshipBuilder(db)
        relationship_count = relationship_builder.generate_all_relationships()
        
        # Validate each question's classification
        for question in SAMPLE_QUESTIONS:
            result = enhanced_classify_complexity(db, question, with_explanation=True)
            actual_results["classifications"][question] = result["classification"]
            actual_results["confidences"].append(result["confidence"])
            
        # Validation checks
        # 1. Check number of questions processed
        if len(actual_results["classifications"]) != EXPECTED_RESULTS["question_count"]:
            validation_passed = False
            validation_failures["question_count"] = {
                "expected": EXPECTED_RESULTS["question_count"],
                "actual": len(actual_results["classifications"])
            }
            
        # 2. Check classifications match expected
        for question, expected_class in EXPECTED_RESULTS["expected_classifications"].items():
            actual_class = actual_results["classifications"].get(question)
            if actual_class != expected_class:
                validation_passed = False
                if "classification_mismatches" not in validation_failures:
                    validation_failures["classification_mismatches"] = []
                validation_failures["classification_mismatches"].append({
                    "question": question,
                    "expected": expected_class,
                    "actual": actual_class
                })
                
        # 3. Check confidence scores meet minimum threshold
        low_confidence = [conf for conf in actual_results["confidences"] if conf < EXPECTED_RESULTS["min_confidence"]]
        if low_confidence:
            validation_passed = False
            validation_failures["confidence_threshold"] = {
                "expected": f">= {EXPECTED_RESULTS['min_confidence']}",
                "actual": f"Found {len(low_confidence)} results below threshold"
            }
            
        logger.info("Demo main function completed.")
        
    except Exception as e:
        logger.exception(f"Demo failed with runtime error: {e}")
        validation_passed = False
        validation_failures["runtime_error"] = str(e)

    # --- Final Reporting ---
    if validation_passed:
        print("\n✅ VALIDATION COMPLETE - Demo script executed successfully.")
        print("All classifications match expected values and confidence scores meet threshold.")
        logger.success("Standalone execution and validation successful.")
        sys.exit(0)
    else:
        print("\n❌ VALIDATION FAILED - Demo script encountered validation errors.")
        print("\nFAILURE DETAILS:")
        for field, details in validation_failures.items():
            if field == "classification_mismatches":
                print("\nClassification Mismatches:")
                for mismatch in details:
                    print(f"  Question: {mismatch['question']}")
                    print(f"    Expected: {mismatch['expected']}")
                    print(f"    Got: {mismatch['actual']}\n")
            elif isinstance(details, dict):
                print(f"  - {field}:")
                print(f"    Expected: {details.get('expected', 'N/A')}")
                print(f"    Got: {details.get('actual', 'N/A')}")
            else:
                print(f"  - {field}: {details}")
        logger.error("Standalone execution and validation failed.")
        sys.exit(1)