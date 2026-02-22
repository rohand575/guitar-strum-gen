"""
run_metrics.py  –  Run full evaluation on the 29 test samples.
Place this file in your project ROOT folder (same level as src/ and data/).
Run: python run_metrics.py
"""

import json
import sys
sys.path.insert(0, '.')

# ── Import using the REAL function names from your metrics.py ──────────────
from src.evaluation.metrics import (
    chord_validity_rate,
    pattern_validity_rate,
    key_adherence_rate,
    key_match_rate,
    genre_match_rate,
    emotion_match_rate,
    unique_progression_ratio,
    unique_pattern_ratio,
    chord_distribution_entropy,
    pattern_distribution_entropy,
    compute_all_metrics,
    format_metrics_for_thesis,
)

# ── Load rule-based outputs (produced by run_eval_rule.py) ─────────────────
with open('rule_based_outputs.json') as f:
    raw_outputs = json.load(f)

# Strip the internal _ref field — metrics.py doesn't expect it
generated = [{k: v for k, v in item.items() if k != '_ref'}
             for item in raw_outputs]

# Ground truth = the original test samples (from _ref)
ground_truth = [item['_ref'] for item in raw_outputs]

print(f"Loaded {len(generated)} generated samples")
print(f"Loaded {len(ground_truth)} ground-truth samples")

# ── Run every metric ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RULE-BASED BASELINE — Full evaluation on 29 test samples")
print("=" * 60)

print("\n--- CORRECTNESS ---")
r1 = chord_validity_rate(generated)
print(r1)

r2 = pattern_validity_rate(generated)
print(r2)

r3 = key_adherence_rate(generated)
print(r3)

print("\n--- PROMPT ADHERENCE ---")
r4 = key_match_rate(generated, ground_truth)
print(r4)

r5 = genre_match_rate(generated, ground_truth)
print(r5)

r6 = emotion_match_rate(generated, ground_truth)
print(r6)

print("\n--- DIVERSITY ---")
r7 = unique_progression_ratio(generated)
print(r7)

r8 = unique_pattern_ratio(generated)
print(r8)

r9 = chord_distribution_entropy(generated)
print(r9)
print(f"  Raw entropy: {r9.details['raw_entropy']:.4f}")

r10 = pattern_distribution_entropy(generated)
print(r10)
print(f"  Raw entropy: {r10.details['raw_entropy']:.4f}")

# ── Full report in thesis-table format ────────────────────────────────────
print("\n\n--- FULL THESIS-FORMAT REPORT ---")
report = compute_all_metrics(generated, ground_truth)
print(format_metrics_for_thesis(report, "Rule-Based Baseline System"))

# ── Save raw numbers to JSON for reference ────────────────────────────────
results = {
    "system": "rule_based",
    "n_samples": len(generated),
    "chord_validity":        round(r1.value, 4),
    "pattern_validity":      round(r2.value, 4),
    "key_adherence":         round(r3.value, 4),
    "key_match":             round(r4.value, 4),
    "genre_match":           round(r5.value, 4),
    "emotion_match":         round(r6.value, 4),
    "unique_progressions":   round(r7.value, 4),
    "unique_patterns":       round(r8.value, 4),
    "chord_entropy":         round(r9.value, 4),
    "pattern_entropy":       round(r10.value, 4),
}
with open('rule_based_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nSaved results to rule_based_metrics.json")
print("\n" + "=" * 60)
print("Done! Copy the output above and paste it back to Claude.")
print("=" * 60)