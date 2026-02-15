"""
Experiments Module - Comparison Studies for Guitar Generation

This module contains functions for running comparison experiments:
1. Rule-Based vs Neural vs Hybrid
2. Ablation studies (conditioning effects)
3. Statistical significance tests

Author: Rohan Rajendra Dhanawade
Project: Master's Thesis - SRH Berlin University of Applied Sciences
Chat: 10 - Evaluation Metrics & Experiments
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import Counter
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (
    compute_all_metrics,
    format_metrics_for_thesis,
    EvaluationReport,
    MetricResult,
)


# =============================================================================
# EXPERIMENT 1: SYSTEM COMPARISON
# =============================================================================

def run_system_comparison(
    test_samples: List[Dict],
    checkpoint_path: str = "checkpoints/guitar_lstm_final.pt",
    output_dir: str = "evaluation_results"
) -> Dict:
    """
    Compare Rule-Based vs Neural vs Hybrid systems.
    
    This is the main experiment for the thesis. It runs all three
    systems on the same test set and produces comparison tables.
    
    Args:
        test_samples: Test samples to evaluate
        checkpoint_path: Path to neural model checkpoint
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing all results and comparison metrics
    """
    from src.evaluation.run_evaluation import (
        generate_with_rule_based,
        generate_with_hybrid,
        generate_neural_only,
    )
    
    results = {}
    ground_truth = [{
        'key': s.get('key'),
        'mode': s.get('mode'),
        'genre': s.get('genre'),
        'emotion': s.get('emotion'),
    } for s in test_samples]
    
    # ─────────────────────────────────────────────────────────────────────────
    # System 1: Rule-Based
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXPERIMENT: Rule-Based System")
    print("=" * 60)
    
    rule_generated = generate_with_rule_based(test_samples, verbose=True)
    rule_report = compute_all_metrics(rule_generated, ground_truth)
    results['rule_based'] = {
        'report': rule_report,
        'generated': rule_generated,
        'source_breakdown': dict(Counter(s.get('source', 'unknown') for s in rule_generated))
    }
    print(format_metrics_for_thesis(rule_report, "RULE-BASED"))
    
    # ─────────────────────────────────────────────────────────────────────────
    # System 2: Neural Only (no fallback)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXPERIMENT: Neural Only (No Fallback)")
    print("=" * 60)
    
    try:
        neural_generated = generate_neural_only(
            test_samples,
            checkpoint_path=checkpoint_path,
            auto_fix=False,
            verbose=True
        )
        
        if neural_generated:
            # Filter valid samples for metrics
            valid_neural = [s for s in neural_generated if s.get('is_valid', False)]
            neural_report = compute_all_metrics(neural_generated, ground_truth)
            results['neural_only'] = {
                'report': neural_report,
                'generated': neural_generated,
                'valid_count': len(valid_neural),
                'total_count': len(neural_generated),
                'validity_rate': len(valid_neural) / len(neural_generated) if neural_generated else 0
            }
            print(format_metrics_for_thesis(neural_report, "NEURAL ONLY"))
        else:
            print("⚠️ Neural generation failed - skipping")
            results['neural_only'] = {'error': 'Generation failed'}
            
    except Exception as e:
        print(f"⚠️ Neural evaluation failed: {e}")
        results['neural_only'] = {'error': str(e)}
    
    # ─────────────────────────────────────────────────────────────────────────
    # System 3: Neural with Auto-Fix
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXPERIMENT: Neural with Auto-Fix")
    print("=" * 60)
    
    try:
        neural_fix_generated = generate_neural_only(
            test_samples,
            checkpoint_path=checkpoint_path,
            auto_fix=True,
            verbose=True
        )
        
        if neural_fix_generated:
            neural_fix_report = compute_all_metrics(neural_fix_generated, ground_truth)
            results['neural_with_fix'] = {
                'report': neural_fix_report,
                'generated': neural_fix_generated,
            }
            print(format_metrics_for_thesis(neural_fix_report, "NEURAL + AUTO-FIX"))
        else:
            results['neural_with_fix'] = {'error': 'Generation failed'}
            
    except Exception as e:
        print(f"⚠️ Neural+fix evaluation failed: {e}")
        results['neural_with_fix'] = {'error': str(e)}
    
    # ─────────────────────────────────────────────────────────────────────────
    # System 4: Hybrid (Neural + Rule Fallback)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXPERIMENT: Hybrid System")
    print("=" * 60)
    
    try:
        hybrid_generated = generate_with_hybrid(
            test_samples,
            checkpoint_path=checkpoint_path,
            prefer_neural=True,
            verbose=True
        )
        
        hybrid_report = compute_all_metrics(hybrid_generated, ground_truth)
        source_breakdown = dict(Counter(s.get('source', 'unknown') for s in hybrid_generated))
        
        results['hybrid'] = {
            'report': hybrid_report,
            'generated': hybrid_generated,
            'source_breakdown': source_breakdown,
            'neural_rate': source_breakdown.get('neural', 0) / len(hybrid_generated),
            'fallback_rate': source_breakdown.get('rule_based', 0) / len(hybrid_generated),
        }
        print(format_metrics_for_thesis(hybrid_report, "HYBRID"))
        
    except Exception as e:
        print(f"⚠️ Hybrid evaluation failed: {e}")
        results['hybrid'] = {'error': str(e)}
    
    # ─────────────────────────────────────────────────────────────────────────
    # Generate Comparison Table
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY TABLE")
    print("=" * 70)
    
    comparison_table = generate_comparison_table(results)
    print(comparison_table)
    
    # Save results
    save_experiment_results(results, output_dir, "system_comparison")
    
    return results


def generate_comparison_table(results: Dict) -> str:
    """Generate a formatted comparison table for thesis."""
    lines = []
    
    systems = ['rule_based', 'neural_only', 'neural_with_fix', 'hybrid']
    system_names = ['Rule-Based', 'Neural Only', 'Neural+Fix', 'Hybrid']
    
    # Header
    header = f"{'Metric':<30}"
    for name in system_names:
        header += f" {name:>12}"
    lines.append(header)
    lines.append("-" * (30 + 13 * len(systems)))
    
    # Metrics to compare
    metrics = [
        ('Chord Validity', 'correctness', 'chord_validity'),
        ('Pattern Validity', 'correctness', 'pattern_validity'),
        ('Key Adherence', 'correctness', 'key_adherence'),
        ('Key Match', 'prompt_adherence', 'key_match'),
        ('Genre Match', 'prompt_adherence', 'genre_match'),
        ('Emotion Match', 'prompt_adherence', 'emotion_match'),
        ('Unique Progressions', 'diversity', 'unique_progressions'),
        ('Unique Patterns', 'diversity', 'unique_patterns'),
    ]
    
    for metric_name, category, key in metrics:
        row = f"{metric_name:<30}"
        for system in systems:
            if system in results and 'report' in results[system]:
                report = results[system]['report']
                if isinstance(report, EvaluationReport):
                    report_dict = report.to_dict()
                else:
                    report_dict = report
                    
                value = report_dict.get(category, {}).get(key, {}).get('value', 'N/A')
                if isinstance(value, float):
                    row += f" {value:>11.1%}"
                else:
                    row += f" {str(value):>12}"
            else:
                row += f" {'N/A':>12}"
        lines.append(row)
    
    # Add source breakdown for hybrid
    if 'hybrid' in results and 'source_breakdown' in results['hybrid']:
        lines.append("-" * (30 + 13 * len(systems)))
        breakdown = results['hybrid']['source_breakdown']
        lines.append(f"Hybrid Source: Neural={breakdown.get('neural', 0)}, "
                     f"Rule-Based={breakdown.get('rule_based', 0)}")
    
    lines.append("=" * (30 + 13 * len(systems)))
    
    return "\n".join(lines)


# =============================================================================
# EXPERIMENT 2: ABLATION STUDY
# =============================================================================

def run_ablation_study(
    test_samples: List[Dict],
    checkpoint_path: str = "checkpoints/guitar_lstm_final.pt",
    output_dir: str = "evaluation_results"
) -> Dict:
    """
    Ablation study: Effect of different conditioning features.
    
    Tests:
    1. Full conditioning (key, mode, genre, emotion, tempo)
    2. No genre conditioning
    3. No emotion conditioning
    4. No key/mode conditioning
    
    This helps understand which features contribute most to quality.
    """
    # This would require modifying the generation to selectively remove features
    # For now, we'll document the approach
    
    print("\n" + "=" * 60)
    print("ABLATION STUDY: Feature Importance")
    print("=" * 60)
    print("\nNote: Full ablation study requires model modifications.")
    print("This experiment would test:")
    print("  1. Full features vs no genre")
    print("  2. Full features vs no emotion")
    print("  3. Full features vs no key/mode")
    print("\nTo implement: Modify generator to mask specific features.")
    
    return {"note": "Ablation study requires model modifications"}


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def compute_significance_tests(results: Dict) -> Dict:
    """
    Compute statistical significance between systems.
    
    Uses McNemar's test for paired binary comparisons
    (e.g., "Did system A get this sample right when system B got it wrong?")
    """
    print("\n" + "=" * 60)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 60)
    
    # For a more rigorous analysis, you'd compute:
    # 1. McNemar's test for binary metrics (valid/invalid)
    # 2. Paired t-test for continuous metrics
    # 3. Bootstrap confidence intervals
    
    print("\nNote: For thesis, consider:")
    print("  - McNemar's test for validity comparisons")
    print("  - Bootstrap confidence intervals for diversity")
    print("  - Effect size (Cohen's d) for continuous metrics")
    
    return {"note": "Statistical tests to be implemented with scipy"}


# =============================================================================
# SAVING RESULTS
# =============================================================================

def save_experiment_results(
    results: Dict,
    output_dir: str,
    experiment_name: str
) -> str:
    """Save experiment results to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"
    filepath = output_path / filename
    
    # Convert EvaluationReport objects to dicts
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            serializable_value = {}
            for k, v in value.items():
                if isinstance(v, EvaluationReport):
                    serializable_value[k] = v.to_dict()
                else:
                    serializable_value[k] = v
            serializable_results[key] = serializable_value
        else:
            serializable_results[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to: {filepath}")
    return str(filepath)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Run all experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run evaluation experiments")
    parser.add_argument(
        "--test-file",
        type=str,
        default="data/processed/test.jsonl",
        help="Path to test JSONL file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/guitar_lstm_final.pt",
        help="Path to neural model checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="comparison",
        choices=["comparison", "ablation", "all"],
        help="Which experiment to run"
    )
    
    args = parser.parse_args()
    
    # Load test data
    from src.evaluation.run_evaluation import load_test_set
    test_samples = load_test_set(args.test_file)
    
    if args.experiment in ["comparison", "all"]:
        run_system_comparison(
            test_samples,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir
        )
    
    if args.experiment in ["ablation", "all"]:
        run_ablation_study(
            test_samples,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
