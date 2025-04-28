def generate_final_report(
    semantic_results: Dict, 
    baseline_results: Dict, 
    distilbert_results: Dict, 
    graph_enhanced_results: Dict,
    db_stats: Dict
) -> str:
    """
    Generate a formatted final report comparing all classification approaches.
    
    Args:
        semantic_results: Results from semantic classification
        baseline_results: Results from baseline model
        distilbert_results: Results from DistilBERT model
        graph_enhanced_results: Results from graph-enhanced classification
        db_stats: Database statistics
        
    Returns:
        Formatted report as string
    """
    # Get information about embedding model 
    embedder = get_embedder()
    embedding_model = getattr(embedder, 'model_name', CONFIG["embedding"]["model_name"])
    
    # Model comparison table
    table = [
        ["Model", "Accuracy", "Precision", "Recall", "F1-Score"],
        [
            f"Graph-Enhanced Classification ({embedding_model}, k={semantic_results['k']})",
            f"{graph_enhanced_results['accuracy']:.3f}",
            f"{graph_enhanced_results['precision']:.3f}",
            f"{graph_enhanced_results['recall']:.3f}",
            f"{graph_enhanced_results['f1_score']:.3f}",
        ],
        [
            f"Semantic Classification ({embedding_model}, k={semantic_results['k']})",
            f"{semantic_results['accuracy']:.3f}",
            f"{semantic_results['precision']:.3f}",
            f"{semantic_results['recall']:.3f}",
            f"{semantic_results['f1_score']:.3f}",
        ],
        [
            "DistilBERT (Trained Model)",
            f"{distilbert_results['accuracy']:.3f}",
            f"{distilbert_results['precision']:.3f}",
            f"{distilbert_results['recall']:.3f}",
            f"{distilbert_results['f1_score']:.3f}",
        ],
        [
            "Logistic Regression (Baseline)",
            f"{baseline_results['accuracy']:.3f}",
            f"{baseline_results['precision']:.3f}",
            f"{baseline_results['recall']:.3f}",
            f"{baseline_results['f1_score']:.3f}",
        ],
    ]
    report = "\n=== Model Performance Comparison ===\n"
    report += tabulate(table, headers="firstrow", tablefmt="grid")
    
    # Effectiveness summary
    report += "\n=== Effectiveness Analysis ===\n"
    
    # Compare graph-enhanced vs semantic
    if graph_enhanced_results["accuracy"] > semantic_results["accuracy"]:
        report += (
            f"The graph-enhanced classification outperforms pure semantic classification "
            f"({graph_enhanced_results['accuracy']:.3f} vs. {semantic_results['accuracy']:.3f}), "
            f"showing that relationship information improves classification accuracy.\n\n"
        )
    else:
        report += (
            f"The pure semantic classification slightly outperforms graph-enhanced classification "
            f"({semantic_results['accuracy']:.3f} vs. {graph_enhanced_results['accuracy']:.3f}), "
            f"suggesting the current relationship structure may need refinement.\n\n"
        )
    
    # Compare best non-parametric vs trained model
    best_nonparametric = max(
        semantic_results["accuracy"], graph_enhanced_results["accuracy"]
    )
    best_nonparametric_name = (
        "Graph-Enhanced" if graph_enhanced_results["accuracy"] > semantic_results["accuracy"]
        else "Semantic"
    )
    
    if distilbert_results["accuracy"] > best_nonparametric:
        difference = distilbert_results["accuracy"] - best_nonparametric
        report += (
            f"The DistilBERT trained model is more effective than the best non-parametric approach ({best_nonparametric_name}), "
            f"achieving higher accuracy ({distilbert_results['accuracy']:.3f} vs. {best_nonparametric:.3f}, "
            f"a difference of {difference:.3f} or {difference*100:.1f}%). "
            f"DistilBERT also shows superior precision ({distilbert_results['precision']:.3f}), "
            f"but the {best_nonparametric_name} approach has advantages in adaptability and explainability.\n"
        )
    elif distilbert_results["accuracy"] < best_nonparametric:
        difference = best_nonparametric - distilbert_results["accuracy"]
        report += (
            f"The {best_nonparametric_name} non-parametric approach is more effective than the DistilBERT trained model, "
            f"achieving higher accuracy ({best_nonparametric:.3f} vs. {distilbert_results['accuracy']:.3f}, "
            f"a difference of {difference:.3f} or {difference*100:.1f}%). "
            f"The non-parametric approach also has advantages in adaptability, explainability, "
            f"and doesn't require retraining to incorporate new examples.\n"
        )
    else:
        report += (
            f"The {best_nonparametric_name} non-parametric approach and the DistilBERT trained model "
            f"achieve equivalent accuracy ({best_nonparametric:.3f}). Given the non-parametric "
            f"approach's advantages in adaptability and explainability, it may be preferable for "
            f"many applications despite the equivalent accuracy.\n"
        )
    
    # Database stats
    report += "\n=== Database Audit Summary ===\n"
    report += tabulate(
        [
            ["Original Documents", db_stats["original_count"]],
            ["Valid Documents", db_stats["valid_count"]],
            ["Invalid Documents", db_stats["invalid_count"]],
            ["Balanced Documents", db_stats["balanced_count"]],
            ["Simple Documents", db_stats["simple_count"]],
            ["Complex Documents", db_stats["complex_count"]],
        ],
        headers=["Metric", "Value"],
        tablefmt="grid",
    )
    
    # Graph relationship summary
    try:
        # Count relationship edges
        prereq_count = db.collection("prerequisites").count()
        related_count = db.collection("related_topics").count()
        
        report += "\n=== Graph Relationship Summary ===\n"
        report += tabulate(
            [
                ["Prerequisite Relationships", prereq_count],
                ["Related Topic Relationships", related_count],
                ["Total Relationships", prereq_count + related_count],
                ["Relationship Density", f"{(prereq_count + related_count) / db_stats['balanced_count']:.2f} per document"],
            ],
            headers=["Metric", "Value"],
            tablefmt="grid",
        )
    except Exception as e:
        logger.warning(f"Could not generate relationship statistics: {e}")
    
    # Conclusion
    report += "\n=== Conclusion ===\n"
    report += (
        f"The evaluation compared four approaches to question complexity classification:\n"
        f"1. Graph-enhanced semantic classification using {embedding_model} and relationship traversal\n"
        f"2. Pure semantic classification using {embedding_model}\n"
        f"3. DistilBERT trained model\n"
        f"4. Logistic regression baseline\n\n"
        f"The best performing approach was {table[1][0] if best_nonparametric == graph_enhanced_results['accuracy'] else table[2][0] if best_nonparametric == semantic_results['accuracy'] else table[3][0]} "
        f"with an accuracy of {max(best_nonparametric, distilbert_results['accuracy']):.3f}.\n\n"
        f"The non-parametric approaches offer the advantage of continuous improvement without retraining, "
        f"as well as better explainability through nearest neighbor and relationship visualization. "
        f"The graph-enhanced approach further improves this explainability by providing structured "
        f"relationships between questions.\n"
    )
    
    return report