"""
Parser Validation Script
========================
Tests the prompt parser against the actual dataset to measure accuracy.

This generates metrics for your thesis:
- Key extraction accuracy
- Mode extraction accuracy  
- Genre detection accuracy
- Emotion detection accuracy
- Tempo category accuracy
- Overall confidence distribution
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.prompt_parser import PromptParser
from src.models.prompt_features import PromptFeatures


def load_dataset(filepath: Path) -> List[Dict]:
    """Load JSONL dataset file."""
    samples = []
    with open(filepath, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def compare_extraction(
    parser: PromptParser,
    sample: Dict
) -> Dict[str, bool]:
    """
    Compare parser extraction against ground truth.
    
    Returns dict with True/False for each field match.
    """
    # Parse the prompt
    features = parser.parse(sample['prompt'])
    
    # Compare each field
    results = {
        'key_match': features.key.upper() == sample['key'].upper(),
        'mode_match': features.mode.lower() == sample['mode'].lower(),
        'genre_match': features.genre.lower() == sample['genre'].lower(),
        'emotion_match': features.emotion.lower() == sample['emotion'].lower(),
        'tempo_category_match': get_tempo_category(features.tempo) == get_tempo_category(sample['tempo']),
    }
    
    return results, features


def get_tempo_category(tempo: int) -> str:
    """Convert BPM to category."""
    if tempo < 80:
        return 'slow'
    elif tempo < 120:
        return 'moderate'
    else:
        return 'fast'


def run_validation(dataset_path: Path, max_samples: int = None) -> Dict:
    """
    Run validation and compute metrics.
    
    Args:
        dataset_path: Path to JSONL dataset
        max_samples: Limit samples for quick testing
    
    Returns:
        Dictionary with accuracy metrics
    """
    print(f"\n{'='*70}")
    print(f"PARSER VALIDATION: {dataset_path.name}")
    print(f"{'='*70}")
    
    # Load data
    samples = load_dataset(dataset_path)
    if max_samples:
        samples = samples[:max_samples]
    
    print(f"Loaded {len(samples)} samples")
    
    # Initialize parser
    parser = PromptParser(verbose=False)
    
    # Track results
    field_correct = Counter()
    field_total = Counter()
    confidence_levels = []
    
    # Detailed error tracking
    errors = defaultdict(list)
    
    for i, sample in enumerate(samples):
        results, features = compare_extraction(parser, sample)
        
        for field, is_correct in results.items():
            field_total[field] += 1
            if is_correct:
                field_correct[field] += 1
            else:
                # Track errors for analysis
                field_name = field.replace('_match', '')
                errors[field_name].append({
                    'prompt': sample['prompt'][:50],
                    'expected': sample.get(field_name, sample.get('tempo')),
                    'got': getattr(features, field_name, features.tempo_category)
                })
        
        confidence_levels.append(features.confidence.overall())
    
    # Compute accuracies
    accuracies = {}
    for field in field_total:
        acc = field_correct[field] / field_total[field] * 100
        accuracies[field] = acc
    
    # Confidence distribution
    conf_dist = {
        'high': sum(1 for c in confidence_levels if c >= 0.8),
        'medium': sum(1 for c in confidence_levels if 0.5 <= c < 0.8),
        'low': sum(1 for c in confidence_levels if 0.2 <= c < 0.5),
        'default': sum(1 for c in confidence_levels if c < 0.2)
    }
    
    avg_confidence = sum(confidence_levels) / len(confidence_levels)
    
    # Print results
    print(f"\n{'='*70}")
    print("ACCURACY RESULTS")
    print(f"{'='*70}")
    
    for field, acc in sorted(accuracies.items(), key=lambda x: -x[1]):
        bar = '█' * int(acc / 5) + '░' * (20 - int(acc / 5))
        print(f"  {field:<25} {bar} {acc:5.1f}%")
    
    print(f"\n{'='*70}")
    print("CONFIDENCE DISTRIBUTION")
    print(f"{'='*70}")
    
    total = len(confidence_levels)
    for level, count in conf_dist.items():
        pct = count / total * 100
        bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
        print(f"  {level:<10} {bar} {count:3d} ({pct:5.1f}%)")
    
    print(f"\n  Average confidence: {avg_confidence:.3f}")
    
    # Show sample errors
    print(f"\n{'='*70}")
    print("SAMPLE ERRORS (first 3 per field)")
    print(f"{'='*70}")
    
    for field, field_errors in errors.items():
        if field_errors:
            print(f"\n  {field.upper()} mismatches:")
            for err in field_errors[:3]:
                print(f"    Prompt: \"{err['prompt']}...\"")
                print(f"    Expected: {err['expected']} | Got: {err['got']}")
    
    return {
        'accuracies': accuracies,
        'confidence_distribution': conf_dist,
        'average_confidence': avg_confidence,
        'total_samples': len(samples)
    }


if __name__ == "__main__":
    # Path to dataset
    data_dir = project_root / "data" / "processed"
    
    # Check if dataset exists
    train_path = data_dir / "train.jsonl"
    
    if not train_path.exists():
        print("Dataset not found at expected location.")
        print("Creating sample data for testing...")
        
        # Create sample test data
        data_dir.mkdir(parents=True, exist_ok=True)
        
        test_samples = [
            {
                "id": "test_001",
                "prompt": "Give me a sad folk song in E minor with a slow tempo",
                "key": "E", "mode": "minor", "genre": "folk",
                "emotion": "melancholic", "tempo": 65
            },
            {
                "id": "test_002", 
                "prompt": "upbeat pop progression in G major at 120 bpm",
                "key": "G", "mode": "major", "genre": "pop",
                "emotion": "upbeat", "tempo": 120
            },
            {
                "id": "test_003",
                "prompt": "peaceful acoustic ballad in C",
                "key": "C", "mode": "major", "genre": "ballad",
                "emotion": "peaceful", "tempo": 70
            },
            {
                "id": "test_004",
                "prompt": "energetic rock anthem in A minor",
                "key": "A", "mode": "minor", "genre": "rock",
                "emotion": "energetic", "tempo": 140
            },
            {
                "id": "test_005",
                "prompt": "mellow jazz in Dm at moderate tempo",
                "key": "D", "mode": "minor", "genre": "jazz",
                "emotion": "mellow", "tempo": 95
            },
        ]
        
        with open(train_path, 'w') as f:
            for sample in test_samples:
                f.write(json.dumps(sample) + '\n')
        
        print(f"Created test file with {len(test_samples)} samples")
    
    # Run validation
    results = run_validation(train_path, max_samples=50)
    
    print(f"\n{'='*70}")
    print("VALIDATION COMPLETE")
    print(f"{'='*70}")
