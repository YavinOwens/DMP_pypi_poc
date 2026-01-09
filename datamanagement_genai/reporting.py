"""
Reporting and document generation utilities

This module provides high-level functions for generating comprehensive reports
using LLMs and creating Word documents.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional

from .benchmark import (
    generate_llm_report,
    create_word_document,
    analyze_benchmark_results,
)
from .config import get_models


def _calculate_model_rankings(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Calculate model rankings from results (helper function)
    
    This replicates the logic from analyze_benchmark_results to get rankings
    """
    models = get_models()
    model_stats = {}
    
    for result in results:
        model = result["model"]
        if model not in model_stats:
            model_stats[model] = {
                "tests": [],
                "success_count": 0,
                "total_tests": 0,
                "total_time": 0,
                "total_cost": 0,
                "total_quality_score": 0,
                "executable_count": 0,
                "executable_tests": 0,
            }
        model_stats[model]["tests"].append(result)
        model_stats[model]["total_tests"] += 1
        if result["success"]:
            model_stats[model]["success_count"] += 1
            model_stats[model]["total_time"] += result.get("elapsed_time", 0)
            model_stats[model]["total_cost"] += result.get("estimated_cost", 0)
            model_stats[model]["total_quality_score"] += result.get("quality_score", 0)
            if result.get("executable") is not None:
                model_stats[model]["executable_tests"] += 1
                if result["executable"]:
                    model_stats[model]["executable_count"] += 1
    
    model_rankings = []
    for model, stats in model_stats.items():
        if stats["success_count"] == 0:
            continue
        avg_time = stats["total_time"] / stats["success_count"]
        avg_cost = stats["total_cost"] / stats["success_count"]
        avg_quality = stats["total_quality_score"] / stats["success_count"]
        success_rate = (stats["success_count"] / stats["total_tests"]) * 100
        executable_rate = (stats["executable_count"] / stats["executable_tests"] * 100) if stats["executable_tests"] > 0 else 0
        
        quality_normalized = (avg_quality / 10) * 100
        cost_efficiency = 100 / (avg_cost * 1000000 + 1)
        speed_score = 100 / (avg_time * 10 + 1)
        composite_score = (
            quality_normalized * 0.4 +
            cost_efficiency * 0.25 +
            speed_score * 0.15 +
            success_rate * 0.1 +
            executable_rate * 0.1
        )
        model_rankings.append({
            "model": model,
            "display": models.get(model, {}).get("display", model),
            "success_rate": success_rate,
            "avg_quality": avg_quality,
            "avg_time": avg_time,
            "avg_cost": avg_cost,
            "executable_rate": executable_rate,
            "composite_score": composite_score,
        })
    
    model_rankings.sort(key=lambda x: x["composite_score"], reverse=True)
    return model_rankings


def generate_full_report(
    session,
    results: List[Dict[str, Any]],
    output_dir: Optional[Path] = None,
    include_llm_report: bool = True,
    include_word_doc: bool = True,
) -> Dict[str, Any]:
    """
    Generate a complete benchmarking report with LLM analysis and Word document
    
    This is a convenience function that:
    1. Analyzes benchmark results
    2. Generates LLM report (if requested)
    3. Creates Word document (if requested)
    
    Args:
        session: Snowflake session
        results: List of test results from run_model_benchmarks
        output_dir: Directory to save output files (default: current directory)
        include_llm_report: Whether to generate LLM report
        include_word_doc: Whether to create Word document
    
    Returns:
        Dictionary containing:
        - model_rankings: List of ranked models
        - best_model: Best model dictionary
        - unavailable_models: List of unavailable models
        - llm_report: LLM-generated report (if generated)
        - word_doc_path: Path to Word document (if created)
    """
    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze results (this prints rankings but doesn't return them)
    best_model_id, unavailable_models = analyze_benchmark_results(results)
    
    # Calculate model rankings from results
    model_rankings = _calculate_model_rankings(results)
    
    # Get best model dict from rankings
    best_model = model_rankings[0] if model_rankings else None
    
    llm_report = None
    if include_llm_report:
        try:
            llm_report = generate_llm_report(session, results, model_rankings, best_model)
        except Exception as e:
            print(f"⚠️  LLM report generation failed: {e}")
    
    word_doc_path = None
    if include_word_doc:
        try:
            word_doc_path = create_word_document(
                results,
                model_rankings,
                best_model,
                llm_report,
                output_dir,
                unavailable_models,
            )
        except Exception as e:
            print(f"⚠️  Word document generation failed: {e}")
    
    return {
        "best_model": best_model,
        "unavailable_models": unavailable_models,
        "llm_report": llm_report,
        "word_doc_path": word_doc_path,
    }
