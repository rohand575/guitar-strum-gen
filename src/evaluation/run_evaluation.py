"""
Run Evaluation Script
=====================

This script runs the complete evaluation pipeline:
1. Load the test set
2. Generate outputs using different systems (neural, rule-based, hybrid)
3. Compute all metrics
4. Save results to JSON and print thesis-ready tables

Usage:
    # From project root:
    python -m src.evaluation.run_evaluation
    
    # Or with specific options:
    python -m src.evaluation.run_evaluation --system hybrid --output results.json

Author: Rohan Rajendra Dhanawade
Project: Master's Thesis - SRH Berlin University of Applied Sciences
Chat: 10 - Evaluation Metrics & Experiments
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (
    compute_all_metrics,
    format_metrics_for_thesis,
    EvaluationReport,
    chord_validity_rate,
    pattern_validity_rate,
    key_adherence_rate
)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_test_set(path: str = "data/processed/test.jsonl") -> List[Dict]:
    """
    Load the test set from a JSONL file.
    
    Args:
        path: Path to test.jsonl file
        
    Returns:
        List of test samples as dictionaries
        
    Each sample has:
        - id: Unique identifier
        - prompt: Natural language prompt
        - chords: Ground truth chord progression
        - strum_pattern: Ground truth strumming pattern
        - tempo, genre, emotion, key, mode: Musical attributes
    """
    test_path = PROJECT_ROOT / path
    
    if not test_path.exists():
        raise FileNotFoundError(f"Test set not found at: {test_path}")
    
    samples = []
    with open(test_path, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    print(f"✓ Loaded {len(samples)} test samples from {test_path}")
    return samples


def load_test_set_from_file(filepath: str) -> List[Dict]:
    """
    Load test set from an explicit file path.
    
    Args:
        filepath: Full path to the JSONL file
        
    Returns:
        List of test samples
    """
    samples = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    print(f"✓ Loaded {len(samples)} test samples from {filepath}")
    return samples


# =============================================================================
# GENERATION FUNCTIONS
# =============================================================================

def generate_with_rule_based(test_samples: List[Dict], verbose: bool = False) -> List[Dict]:
    """
    Generate outputs using only the rule-based system.
    
    Args:
        test_samples: List of test samples with prompts
        verbose: Print progress
        
    Returns:
        List of generated outputs
    """
    from src.rules.generate_rule_based import generate_rule_based
    
    results = []
    total = len(test_samples)
    
    for i, sample in enumerate(test_samples):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Rule-based generation: {i+1}/{total}")
        
        try:
            output = generate_rule_based(sample["prompt"], verbose=False)
            
            results.append({
                "id": sample.get("id", f"sample_{i}"),
                "prompt": sample["prompt"],
                "chords": output.chords,
                "strum_pattern": output.strum_pattern,
                "tempo": output.tempo,
                "key": output.key,
                "mode": output.mode,
                "genre": output.genre,
                "emotion": output.emotion,
                "source": "rule_based",
                "ground_truth": {
                    "chords": sample.get("chords"),
                    "strum_pattern": sample.get("strum_pattern"),
                    "key": sample.get("key"),
                    "mode": sample.get("mode"),
                    "genre": sample.get("genre"),
                    "emotion": sample.get("emotion"),
                }
            })
        except Exception as e:
            print(f"  ⚠️ Error on sample {i}: {e}")
            results.append({
                "id": sample.get("id", f"sample_{i}"),
                "prompt": sample["prompt"],
                "error": str(e),
                "source": "rule_based"
            })
    
    return results


def generate_with_hybrid(
    test_samples: List[Dict],
    checkpoint_path: str = "checkpoints/guitar_lstm_final.pt",
    prefer_neural: bool = True,
    verbose: bool = False
) -> List[Dict]:
    """
    Generate outputs using the hybrid system (neural + rule fallback).
    
    Args:
        test_samples: List of test samples with prompts
        checkpoint_path: Path to neural model checkpoint
        prefer_neural: If True, try neural first; if False, use rules only
        verbose: Print progress
        
    Returns:
        List of generated outputs
    """
    from src.app.generate import generate_guitar_part
    
    results = []
    total = len(test_samples)
    neural_count = 0
    rule_count = 0
    
    for i, sample in enumerate(test_samples):
        if verbose and (i + 1) % 5 == 0:
            print(f"  Hybrid generation: {i+1}/{total} (neural: {neural_count}, rules: {rule_count})")
        
        try:
            output = generate_guitar_part(
                sample["prompt"],
                prefer_neural=prefer_neural,
                checkpoint_path=checkpoint_path,
                verbose=False
            )
            
            # Track source
            if output.get("source") == "neural":
                neural_count += 1
            else:
                rule_count += 1
            
            results.append({
                "id": sample.get("id", f"sample_{i}"),
                "prompt": sample["prompt"],
                "chords": output["chords"],
                "strum_pattern": output["strum_pattern"],
                "tempo": output.get("tempo"),
                "key": output.get("key"),
                "mode": output.get("mode"),
                "genre": output.get("genre"),
                "emotion": output.get("emotion"),
                "source": output.get("source", "unknown"),
                "validation": str(output.get("validation", "")),
                "fallback_reason": output.get("fallback_reason"),
                "ground_truth": {
                    "chords": sample.get("chords"),
                    "strum_pattern": sample.get("strum_pattern"),
                    "key": sample.get("key"),
                    "mode": sample.get("mode"),
                    "genre": sample.get("genre"),
                    "emotion": sample.get("emotion"),
                }
            })
        except Exception as e:
            print(f"  ⚠️ Error on sample {i}: {e}")
            rule_count += 1
            results.append({
                "id": sample.get("id", f"sample_{i}"),
                "prompt": sample["prompt"],
                "error": str(e),
                "source": "error"
            })
    
    if verbose:
        print(f"\n  Final counts - Neural: {neural_count}, Rule-based: {rule_count}")
    
    return results


def generate_neural_only(
    test_samples: List[Dict],
    checkpoint_path: str = "checkpoints/guitar_lstm_final.pt",
    auto_fix: bool = False,
    verbose: bool = False
) -> List[Dict]:
    """
    Generate outputs using ONLY the neural model (no fallback).
    
    This is useful for measuring raw neural model performance.
    Invalid outputs are still recorded (not replaced with rules).
    
    Args:
        test_samples: List of test samples with prompts
        checkpoint_path: Path to neural model checkpoint
        auto_fix: Whether to apply pattern auto-fix
        verbose: Print progress
        
    Returns:
        List of generated outputs (may include invalid ones)
    """
    try:
        from src.app.generate import NeuralGenerator, validate_output, fix_strum_pattern
        from src.models.prompt_parser import PromptParser
    except ImportError as e:
        print(f"⚠️ Could not import neural components: {e}")
        print("   Neural-only evaluation requires PyTorch and trained models.")
        return []
    
    # Initialize components
    try:
        generator = NeuralGenerator(checkpoint_path)
        parser = PromptParser()
    except Exception as e:
        print(f"⚠️ Could not initialize neural generator: {e}")
        return []
    
    results = []
    total = len(test_samples)
    valid_count = 0
    
    for i, sample in enumerate(test_samples):
        if verbose and (i + 1) % 5 == 0:
            print(f"  Neural generation: {i+1}/{total} (valid so far: {valid_count})")
        
        try:
            # Parse prompt to get features
            features = parser.parse(sample["prompt"])
            
            # Generate with neural model
            output = generator.generate(
                features={
                    "key": features.get("key", "C"),
                    "mode": features.get("mode", "major"),
                    "genre": features.get("genre", "pop"),
                    "emotion": features.get("emotion", "mellow"),
                    "tempo": features.get("tempo", 100)
                },
                temperature=0.8,
                top_k=10
            )
            
            if output is None:
                results.append({
                    "id": sample.get("id", f"sample_{i}"),
                    "prompt": sample["prompt"],
                    "error": "Generation returned None",
                    "source": "neural_failed"
                })
                continue
            
            chords = output.get("chords", [])
            pattern = output.get("strum_pattern", "")
            
            # Optionally apply auto-fix
            if auto_fix and pattern:
                pattern, was_fixed, _ = fix_strum_pattern(pattern)
            
            # Validate
            validation = validate_output(
                chords=chords,
                strum_pattern=pattern,
                key=features.get("key"),
                mode=features.get("mode")
            )
            
            if validation.is_valid:
                valid_count += 1
            
            results.append({
                "id": sample.get("id", f"sample_{i}"),
                "prompt": sample["prompt"],
                "chords": chords,
                "strum_pattern": pattern,
                "key": features.get("key"),
                "mode": features.get("mode"),
                "genre": features.get("genre"),
                "emotion": features.get("emotion"),
                "source": "neural",
                "is_valid": validation.is_valid,
                "validation_errors": validation.errors,
                "ground_truth": {
                    "chords": sample.get("chords"),
                    "strum_pattern": sample.get("strum_pattern"),
                    "key": sample.get("key"),
                    "mode": sample.get("mode"),
                    "genre": sample.get("genre"),
                    "emotion": sample.get("emotion"),
                }
            })
            
        except Exception as e:
            print(f"  ⚠️ Error on sample {i}: {e}")
            results.append({
                "id": sample.get("id", f"sample_{i}"),
                "prompt": sample["prompt"],
                "error": str(e),
                "source": "neural_error"
            })
    
    if verbose:
        print(f"\n  Valid outputs: {valid_count}/{total} ({valid_count/total*100:.1f}%)")
    
    return results


# =============================================================================
# EVALUATION RUNNER
# =============================================================================

def run_evaluation(
    test_samples: List[Dict],
    system: str = "rule_based",
    checkpoint_path: str = "checkpoints/guitar_lstm_final.pt",
    verbose: bool = True
) -> Tuple[List[Dict], EvaluationReport]:
    """
    Run complete evaluation for a system.
    
    Args:
        test_samples: List of test samples
        system: One of "rule_based", "hybrid", "neural_only"
        checkpoint_path: Path to neural model (for hybrid/neural_only)
        verbose: Print progress
        
    Returns:
        Tuple of (generated_samples, evaluation_report)
    """
    print(f"\n{'='*60}")
    print(f"Running evaluation for: {system.upper()}")
    print(f"{'='*60}")
    
    # Generate outputs
    if system == "rule_based":
        generated = generate_with_rule_based(test_samples, verbose=verbose)
    elif system == "hybrid":
        generated = generate_with_hybrid(
            test_samples,
            checkpoint_path=checkpoint_path,
            prefer_neural=True,
            verbose=verbose
        )
    elif system == "neural_only":
        generated = generate_neural_only(
            test_samples,
            checkpoint_path=checkpoint_path,
            auto_fix=False,
            verbose=verbose
        )
    elif system == "neural_with_fix":
        generated = generate_neural_only(
            test_samples,
            checkpoint_path=checkpoint_path,
            auto_fix=True,
            verbose=verbose
        )
    else:
        raise ValueError(f"Unknown system: {system}. Use 'rule_based', 'hybrid', 'neural_only', or 'neural_with_fix'")
    
    # Filter out error samples for metrics
    valid_generated = [s for s in generated if "error" not in s]
    
    if len(valid_generated) < len(generated):
        print(f"\n⚠️ {len(generated) - len(valid_generated)} samples had errors and were excluded from metrics")
    
    # Compute metrics
    print(f"\nComputing metrics on {len(valid_generated)} samples...")
    
    # Extract ground truth for prompt adherence metrics
    ground_truth = [s.get("ground_truth", {}) for s in valid_generated]
    
    report = compute_all_metrics(valid_generated, ground_truth)
    
    return generated, report


def save_results(
    results: Dict,
    output_path: str,
    include_samples: bool = False
) -> None:
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Dictionary containing metrics and optionally samples
        output_path: Path to save JSON file
        include_samples: Whether to include generated samples in output
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to: {output_file}")


# =============================================================================
# COMPARISON REPORT
# =============================================================================

def compare_systems(
    test_samples: List[Dict],
    systems: List[str] = ["rule_based"],
    checkpoint_path: str = "checkpoints/guitar_lstm_final.pt",
    output_dir: str = "evaluation_results"
) -> Dict:
    """
    Run evaluation on multiple systems and create comparison report.
    
    Args:
        test_samples: Test samples to evaluate
        systems: List of systems to compare
        checkpoint_path: Path to neural checkpoint
        output_dir: Directory to save results
        
    Returns:
        Dictionary with all results and comparison
    """
    all_results = {}
    
    for system in systems:
        try:
            generated, report = run_evaluation(
                test_samples,
                system=system,
                checkpoint_path=checkpoint_path,
                verbose=True
            )
            
            all_results[system] = {
                "report": report.to_dict(),
                "generated_count": len(generated),
                "error_count": len([s for s in generated if "error" in s]),
            }
            
            # Print formatted report
            print(format_metrics_for_thesis(report, system.upper()))
            
        except Exception as e:
            print(f"\n⚠️ Evaluation failed for {system}: {e}")
            all_results[system] = {"error": str(e)}
    
    # Create comparison summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    # Build comparison table
    metrics_to_compare = [
        ("Chord Validity", "correctness", "chord_validity"),
        ("Pattern Validity", "correctness", "pattern_validity"),
        ("Key Adherence", "correctness", "key_adherence"),
        ("Unique Progressions", "diversity", "unique_progressions"),
        ("Unique Patterns", "diversity", "unique_patterns"),
    ]
    
    # Header
    header = f"{'Metric':<25}"
    for system in systems:
        header += f" {system:>15}"
    print(header)
    print("-" * (25 + 16 * len(systems)))
    
    # Rows
    for metric_name, category, key in metrics_to_compare:
        row = f"{metric_name:<25}"
        for system in systems:
            if system in all_results and "report" in all_results[system]:
                value = all_results[system]["report"].get(category, {}).get(key, {}).get("value", "N/A")
                if isinstance(value, float):
                    row += f" {value:>14.1%}"
                else:
                    row += f" {str(value):>15}"
            else:
                row += f" {'ERROR':>15}"
        print(row)
    
    print("=" * 70)
    
    # Save comparison results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"comparison_{timestamp}.json"
    
    comparison_output = {
        "timestamp": timestamp,
        "test_sample_count": len(test_samples),
        "systems_evaluated": systems,
        "results": all_results
    }
    
    save_results(comparison_output, str(output_path))
    
    return all_results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation on guitar generation system"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="data/processed/test.jsonl",
        help="Path to test JSONL file"
    )
    parser.add_argument(
        "--system",
        type=str,
        default="rule_based",
        choices=["rule_based", "hybrid", "neural_only", "neural_with_fix", "all"],
        help="Which system to evaluate"
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
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )
    
    args = parser.parse_args()
    
    # Load test set
    try:
        test_samples = load_test_set(args.test_file)
    except FileNotFoundError:
        print(f"Test file not found at: {args.test_file}")
        print("Please provide the correct path using --test-file")
        return
    
    # Run evaluation
    if args.system == "all":
        # Compare all available systems
        systems = ["rule_based"]  # Start with rule_based (always available)
        
        # Check if neural is available
        try:
            import torch
            systems.extend(["neural_only", "hybrid"])
        except ImportError:
            print("⚠️ PyTorch not available, skipping neural evaluations")
        
        compare_systems(
            test_samples,
            systems=systems,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir
        )
    else:
        # Single system evaluation
        generated, report = run_evaluation(
            test_samples,
            system=args.system,
            checkpoint_path=args.checkpoint,
            verbose=args.verbose
        )
        
        # Print report
        print(format_metrics_for_thesis(report, args.system.upper()))
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(args.output_dir) / f"{args.system}_{timestamp}.json"
        
        save_results(
            {
                "system": args.system,
                "timestamp": timestamp,
                "sample_count": len(generated),
                "report": report.to_dict(),
                "samples": generated
            },
            str(output_path),
            include_samples=True
        )


if __name__ == "__main__":
    main()
