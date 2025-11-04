"""
LLM Evaluation with Few-Shot Learning

This module provides a thin wrapper around the main LLM pipeline evaluation.
All LLM evaluation logic is centralized in ap_ai_classifier.models.llm.pipeline
to ensure single source of truth.
"""
from ap_ai_classifier.models.llm.pipeline import evaluate_llm_pipeline
from ap_ai_classifier.config import LLM_MODEL


def main():
    """
    Main entry point for LLM evaluation.
    
    This is a thin wrapper that delegates to evaluate_llm_pipeline()
    to maintain backward compatibility with evaluate.py while ensuring
    all LLM evaluation logic comes from a single source.
    
    Returns:
        Dictionary with evaluation results (compatible with evaluate.py format)
    """
    # Delegate to the main pipeline evaluation
    results = evaluate_llm_pipeline()
    
    # Convert to format expected by evaluate.py
    return {
        'experiment': f'LLM with Few-Shot Learning ({LLM_MODEL})',
        'overall_f1_macro': results['overall_f1'],
        'sample_wise_f1': results['sample_wise_f1'],  # Include sample-wise F1 for consistency
        'nominal_f1': results['nominal_f1'],
        'department_f1': results['department_f1'],
        'tc_f1': results['tax_code_f1'],
        'time_seconds': results['time_elapsed']
    }


if __name__ == "__main__":
    main()
