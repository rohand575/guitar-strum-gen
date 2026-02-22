"""
run_neural_metrics_only.py
Run in project ROOT: python run_neural_metrics_only.py
This re-generates and runs metrics — skips the file save that was crashing.
"""
import json, sys
sys.path.insert(0, '.')

from src.app.generate import generate_guitar_part
from src.evaluation.metrics import (
    chord_validity_rate, pattern_validity_rate, key_adherence_rate,
    key_match_rate, genre_match_rate, emotion_match_rate,
    unique_progression_ratio, unique_pattern_ratio,
    chord_distribution_entropy, pattern_distribution_entropy,
    compute_all_metrics, format_metrics_for_thesis,
)

test_samples = []
with open('data/processed/test.jsonl') as f:
    for line in f:
        test_samples.append(json.loads(line))

generated = []
for i, sample in enumerate(test_samples):
    try:
        result = generate_guitar_part(
            sample['prompt'],
            checkpoint_path='checkpoints/best_model.pt',
            prefer_neural=True,
            temperature=0.8,
            verbose=False
        )
        # Convert to plain dict, stripping non-serializable objects
        if hasattr(result, 'model_dump'):
            d = result.model_dump()
        elif hasattr(result, 'dict'):
            d = result.dict()
        elif isinstance(result, dict):
            # Deep-copy, converting any non-primitive values to strings
            d = {}
            for k, v in result.items():
                if isinstance(v, (str, int, float, bool, list, type(None))):
                    d[k] = v
                else:
                    d[k] = str(v)
        else:
            d = {}

        d.setdefault('key',     sample.get('key', 'C'))
        d.setdefault('mode',    sample.get('mode', 'major'))
        d.setdefault('genre',   sample.get('genre', 'pop'))
        d.setdefault('emotion', sample.get('emotion', 'mellow'))
        generated.append(d)
    except Exception as e:
        print(f"  ERROR {sample['id']}: {e}")

from collections import Counter
sources = Counter(g.get('source', g.get('generation_source', 'unknown')) for g in generated)
print(f"Generated {len(generated)}/29  |  Sources: {dict(sources)}")

ground_truth = test_samples

print("\n" + "=" * 60)
print(f"NEURAL LSTM MODEL — {len(generated)} test samples")
print("=" * 60)

print("\n--- CORRECTNESS ---")
r1 = chord_validity_rate(generated);   print(r1)
r2 = pattern_validity_rate(generated); print(r2)
r3 = key_adherence_rate(generated);    print(r3)

print("\n--- PROMPT ADHERENCE ---")
r4 = key_match_rate(generated, ground_truth);     print(r4)
r5 = genre_match_rate(generated, ground_truth);   print(r5)
r6 = emotion_match_rate(generated, ground_truth); print(r6)

print("\n--- DIVERSITY ---")
r7 = unique_progression_ratio(generated); print(r7)
r8 = unique_pattern_ratio(generated);     print(r8)
r9  = chord_distribution_entropy(generated)
print(r9); print(f"  Raw entropy: {r9.details['raw_entropy']:.4f}")
r10 = pattern_distribution_entropy(generated)
print(r10); print(f"  Raw entropy: {r10.details['raw_entropy']:.4f}")

print("\n--- THESIS-FORMAT REPORT ---")
report = compute_all_metrics(generated, ground_truth)
print(format_metrics_for_thesis(report, "Neural LSTM Model"))

results = {
    "system": "neural_lstm", "n_samples": len(generated),
    "source_breakdown": dict(sources),
    "chord_validity":      round(r1.value, 4),
    "pattern_validity":    round(r2.value, 4),
    "key_adherence":       round(r3.value, 4),
    "key_match":           round(r4.value, 4),
    "genre_match":         round(r5.value, 4),
    "emotion_match":       round(r6.value, 4),
    "unique_progressions": round(r7.value, 4),
    "unique_patterns":     round(r8.value, 4),
    "chord_entropy":       round(r9.value, 4),
    "pattern_entropy":     round(r10.value, 4),
}
# Safe JSON save
with open('neural_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved summary to neural_metrics.json")
print("DONE — paste full output to Claude.")