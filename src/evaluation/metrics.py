"""
Evaluation Metrics for Guitar Generation System
================================================

This module contains all objective metrics for evaluating the quality
of generated chord progressions and strumming patterns.

Metrics are organized into four categories:
1. CORRECTNESS - Is the output technically valid?
2. PROMPT ADHERENCE - Did it follow the instructions?
3. DIVERSITY - Is the output creative and varied?
4. COMPARISON - How do different systems compare?

Each metric returns a value between 0.0 and 1.0 (or a count/entropy value)
along with detailed breakdown information for thesis reporting.

Author: Rohan Rajendra Dhanawade
Project: Master's Thesis - SRH Berlin University of Applied Sciences
"""

from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import Counter
import math
import json


# =============================================================================
# CONSTANTS (imported from generate.py concepts)
# =============================================================================

# All valid chords that our system recognizes (29 chords)
VALID_CHORDS = {
    # Natural major chords
    "C", "D", "E", "F", "G", "A", "B",
    # Sharp major chords
    "A#", "C#", "D#", "F#", "G#",
    # Minor chords
    "Am", "Bm", "Cm", "Dm", "Em", "Fm", "Gm",
    "A#m", "C#m", "F#m", "G#m",
    # Seventh chords
    "A7", "B7", "D7", "E7",
    # Other types
    "Asus4", "C#dim", "Gdim",
}

# Valid strumming pattern characters
VALID_STRUM_CHARS = {"D", "U", "_"}

# Required pattern length
REQUIRED_PATTERN_LENGTH = 8

# Chromatic scale for key calculations
CHROMATIC_SCALE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Enharmonic equivalents (flats → sharps)
FLAT_TO_SHARP = {
    "Db": "C#", "Eb": "D#", "Fb": "E", "Gb": "F#", "Ab": "G#", "Bb": "A#"
}

# Valid genres and emotions (from the dataset)
VALID_GENRES = {"rock", "pop", "folk", "country", "ballad", "blues", "jazz", "indie", "acoustic"}
VALID_EMOTIONS = {"upbeat", "melancholic", "mellow", "energetic", "peaceful", "dramatic", "hopeful", "nostalgic"}


# =============================================================================
# DATA CLASSES FOR STRUCTURED RESULTS
# =============================================================================

@dataclass
class MetricResult:
    """
    Container for a single metric's result.
    
    Attributes:
        name: Human-readable metric name
        value: The metric value (usually 0.0 to 1.0, or a count)
        numerator: Count of items meeting the criteria
        denominator: Total count of items evaluated
        details: Additional information for debugging/reporting
    """
    name: str
    value: float
    numerator: int = 0
    denominator: int = 0
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        if self.denominator > 0:
            return f"{self.name}: {self.value:.2%} ({self.numerator}/{self.denominator})"
        else:
            return f"{self.name}: {self.value:.4f}"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "value": self.value,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "details": self.details
        }


@dataclass
class EvaluationReport:
    """
    Complete evaluation report containing all metrics.
    
    This is the main output of compute_all_metrics().
    """
    correctness: Dict[str, MetricResult] = field(default_factory=dict)
    prompt_adherence: Dict[str, MetricResult] = field(default_factory=dict)
    diversity: Dict[str, MetricResult] = field(default_factory=dict)
    sample_count: int = 0
    source_breakdown: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert entire report to dictionary."""
        return {
            "sample_count": self.sample_count,
            "source_breakdown": self.source_breakdown,
            "correctness": {k: v.to_dict() for k, v in self.correctness.items()},
            "prompt_adherence": {k: v.to_dict() for k, v in self.prompt_adherence.items()},
            "diversity": {k: v.to_dict() for k, v in self.diversity.items()},
        }
    
    def __str__(self) -> str:
        """Pretty print the report."""
        lines = []
        lines.append("=" * 60)
        lines.append("EVALUATION REPORT")
        lines.append("=" * 60)
        lines.append(f"Total samples evaluated: {self.sample_count}")
        
        if self.source_breakdown:
            lines.append(f"Source breakdown: {self.source_breakdown}")
        
        lines.append("\n--- CORRECTNESS METRICS ---")
        for metric in self.correctness.values():
            lines.append(f"  {metric}")
        
        lines.append("\n--- PROMPT ADHERENCE METRICS ---")
        for metric in self.prompt_adherence.values():
            lines.append(f"  {metric}")
        
        lines.append("\n--- DIVERSITY METRICS ---")
        for metric in self.diversity.values():
            lines.append(f"  {metric}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_key(key: str) -> str:
    """
    Normalize a key name to standard format.
    
    Examples:
        "Am" → "A" (root), "minor" (mode)
        "G" → "G" (root), "major" (mode) 
        "Bb" → "A#" (normalized to sharp)
    
    Args:
        key: Key string like "G", "Am", "Bb"
        
    Returns:
        Normalized root note (sharps only)
    """
    if not key:
        return ""
    
    # Handle minor keys like "Am", "Em"
    if key.endswith("m") and len(key) >= 2:
        root = key[:-1]
    else:
        root = key
    
    # Handle flats
    if root in FLAT_TO_SHARP:
        root = FLAT_TO_SHARP[root]
    
    # Handle sharps in key names like "F#m" → root is "F#"
    if len(root) == 2 and root[1] == '#':
        pass  # Already in correct format
    elif len(root) == 2 and root[1] == 'b':
        # Flat notation
        flat_key = root[0].upper() + 'b'
        if flat_key in FLAT_TO_SHARP:
            root = FLAT_TO_SHARP[flat_key]
    
    return root.upper() if len(root) == 1 else root[0].upper() + root[1]


def get_diatonic_chords(key: str, mode: str) -> List[str]:
    """
    Get the 7 diatonic chords for a given key and mode.
    
    This tells us which chords "belong" to a key according to music theory.
    
    Args:
        key: Root note (e.g., "G", "A")
        mode: "major" or "minor"
        
    Returns:
        List of 7 chord symbols that are diatonic to the key
        
    Example:
        >>> get_diatonic_chords("G", "major")
        ['G', 'Am', 'Bm', 'C', 'D', 'Em', 'F#dim']
        
        >>> get_diatonic_chords("A", "minor")
        ['Am', 'Bdim', 'C', 'Dm', 'Em', 'F', 'G']
    """
    # Normalize the key
    root = normalize_key(key)
    
    if root not in CHROMATIC_SCALE:
        return []  # Unknown key
    
    root_idx = CHROMATIC_SCALE.index(root)
    
    # Define intervals and chord qualities for each mode
    if mode.lower() == "minor":
        # Natural minor scale intervals: W-H-W-W-H-W-W
        intervals = [0, 2, 3, 5, 7, 8, 10]
        qualities = ["m", "dim", "", "m", "m", "", ""]
    else:  # major
        # Major scale intervals: W-W-H-W-W-W-H
        intervals = [0, 2, 4, 5, 7, 9, 11]
        qualities = ["", "m", "m", "", "", "m", "dim"]
    
    # Build the chords
    chords = []
    for interval, quality in zip(intervals, qualities):
        note_idx = (root_idx + interval) % 12
        note = CHROMATIC_SCALE[note_idx]
        chords.append(note + quality)
    
    return chords


def calculate_entropy(items: List[str]) -> float:
    """
    Calculate Shannon entropy of a distribution.
    
    Entropy measures how "spread out" or "uniform" a distribution is.
    - High entropy = items are evenly distributed (diverse)
    - Low entropy = a few items dominate (not diverse)
    
    The formula is: H = -Σ p(x) * log2(p(x))
    
    Args:
        items: List of items (can have duplicates)
        
    Returns:
        Entropy value (0 = all same, higher = more diverse)
        Maximum possible entropy is log2(n) where n = number of unique items
        
    Example:
        >>> calculate_entropy(["A", "A", "A", "A"])  # All same
        0.0
        
        >>> calculate_entropy(["A", "B", "C", "D"])  # All different
        2.0  # log2(4) = 2
    """
    if not items:
        return 0.0
    
    # Count occurrences
    counts = Counter(items)
    total = len(items)
    
    # Calculate entropy
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            probability = count / total
            entropy -= probability * math.log2(probability)
    
    return entropy


def normalize_entropy(entropy: float, num_unique: int) -> float:
    """
    Normalize entropy to 0-1 range.
    
    Normalized entropy = entropy / max_possible_entropy
    where max_possible_entropy = log2(num_unique)
    
    Args:
        entropy: Raw entropy value
        num_unique: Number of unique items
        
    Returns:
        Normalized entropy between 0 and 1
    """
    if num_unique <= 1:
        return 0.0
    
    max_entropy = math.log2(num_unique)
    if max_entropy == 0:
        return 0.0
    
    return entropy / max_entropy


# =============================================================================
# SECTION 1: CORRECTNESS METRICS
# =============================================================================

def chord_validity_rate(samples: List[Dict]) -> MetricResult:
    """
    Calculate the percentage of samples with all valid chord symbols.
    
    A chord is "valid" if it exists in our VALID_CHORDS vocabulary.
    This is a TECHNICAL check, not a musical one.
    
    Args:
        samples: List of generation result dictionaries
                 Each must have a "chords" key with a list of chord strings
                 
    Returns:
        MetricResult with validity rate and details about invalid chords
        
    Example:
        >>> samples = [
        ...     {"chords": ["G", "D", "Em", "C"]},  # All valid
        ...     {"chords": ["G", "Xm", "Em"]},      # Xm is invalid
        ... ]
        >>> result = chord_validity_rate(samples)
        >>> result.value
        0.5  # 50% of samples have all valid chords
    """
    if not samples:
        return MetricResult(
            name="Chord Validity Rate",
            value=0.0,
            numerator=0,
            denominator=0,
            details={"error": "No samples provided"}
        )
    
    valid_count = 0
    invalid_chords_found = []
    
    for sample in samples:
        chords = sample.get("chords", [])
        
        # Check if all chords in this sample are valid
        sample_invalid = [c for c in chords if c not in VALID_CHORDS]
        
        if not sample_invalid:
            valid_count += 1
        else:
            invalid_chords_found.extend(sample_invalid)
    
    total = len(samples)
    rate = valid_count / total if total > 0 else 0.0
    
    return MetricResult(
        name="Chord Validity Rate",
        value=rate,
        numerator=valid_count,
        denominator=total,
        details={
            "invalid_chords_found": list(set(invalid_chords_found)),
            "invalid_chord_count": len(invalid_chords_found)
        }
    )


def pattern_validity_rate(samples: List[Dict]) -> MetricResult:
    """
    Calculate the percentage of samples with valid strumming patterns.
    
    A pattern is "valid" if:
    1. It is exactly 8 characters long
    2. Each character is D, U, or _
    3. It's not all rests (________)
    
    Args:
        samples: List of generation result dictionaries
                 Each must have a "strum_pattern" key
                 
    Returns:
        MetricResult with validity rate and details about failures
    """
    if not samples:
        return MetricResult(
            name="Pattern Validity Rate",
            value=0.0,
            numerator=0,
            denominator=0,
            details={"error": "No samples provided"}
        )
    
    valid_count = 0
    invalid_reasons = {"wrong_length": 0, "invalid_chars": 0, "all_rests": 0}
    
    for sample in samples:
        pattern = sample.get("strum_pattern", "")
        is_valid = True
        
        # Check length
        if len(pattern) != REQUIRED_PATTERN_LENGTH:
            is_valid = False
            invalid_reasons["wrong_length"] += 1
        
        # Check characters
        elif any(c not in VALID_STRUM_CHARS for c in pattern):
            is_valid = False
            invalid_reasons["invalid_chars"] += 1
        
        # Check not all rests
        elif all(c == "_" for c in pattern):
            is_valid = False
            invalid_reasons["all_rests"] += 1
        
        if is_valid:
            valid_count += 1
    
    total = len(samples)
    rate = valid_count / total if total > 0 else 0.0
    
    return MetricResult(
        name="Pattern Validity Rate",
        value=rate,
        numerator=valid_count,
        denominator=total,
        details={"invalid_reasons": invalid_reasons}
    )


def key_adherence_rate(samples: List[Dict]) -> MetricResult:
    """
    Calculate what percentage of chords are diatonic to their stated key.
    
    This measures MUSICAL correctness:
    - If a sample says key="G", mode="major"
    - We check if all chords belong to G major's diatonic chords
    
    "Borrowed chords" (from parallel keys) are counted as non-adherent,
    even though they're musically acceptable. This is a strict metric.
    
    Args:
        samples: List of generation results with "chords", "key", "mode"
        
    Returns:
        MetricResult with adherence rate (chord-level, not sample-level)
        
    Example:
        Sample with G major: ["G", "D", "Em", "A"]
        - G, D, Em are diatonic to G major ✓
        - A major is NOT diatonic (Am would be) ✗
        - Adherence for this sample = 3/4 = 75%
    """
    if not samples:
        return MetricResult(
            name="Key Adherence Rate",
            value=0.0,
            numerator=0,
            denominator=0,
            details={"error": "No samples provided"}
        )
    
    total_chords = 0
    diatonic_chords = 0
    non_diatonic_examples = []
    
    for sample in samples:
        chords = sample.get("chords", [])
        key = sample.get("key", "")
        mode = sample.get("mode", "major")
        
        if not key or not chords:
            continue
        
        # Get diatonic chords for this key
        diatonic_set = set(get_diatonic_chords(key, mode))
        
        for chord in chords:
            total_chords += 1
            if chord in diatonic_set:
                diatonic_chords += 1
            else:
                non_diatonic_examples.append({
                    "chord": chord,
                    "key": key,
                    "mode": mode,
                    "diatonic_options": list(diatonic_set)
                })
    
    rate = diatonic_chords / total_chords if total_chords > 0 else 0.0
    
    # Limit examples for readability
    if len(non_diatonic_examples) > 10:
        non_diatonic_examples = non_diatonic_examples[:10]
        non_diatonic_examples.append({"note": "... and more"})
    
    return MetricResult(
        name="Key Adherence Rate",
        value=rate,
        numerator=diatonic_chords,
        denominator=total_chords,
        details={
            "non_diatonic_examples": non_diatonic_examples,
            "total_non_diatonic": total_chords - diatonic_chords
        }
    )


# =============================================================================
# SECTION 2: PROMPT ADHERENCE METRICS
# =============================================================================

def key_match_rate(samples: List[Dict], ground_truth: List[Dict]) -> MetricResult:
    """
    Calculate how often the generated key matches the requested key.
    
    Args:
        samples: Generated outputs (each with "key" field)
        ground_truth: Original prompts/labels (each with "key" field)
        
    Returns:
        MetricResult with match rate
        
    Note:
        Keys are normalized before comparison (e.g., "Bb" == "A#")
    """
    if not samples or not ground_truth:
        return MetricResult(
            name="Key Match Rate",
            value=0.0,
            numerator=0,
            denominator=0,
            details={"error": "Missing samples or ground truth"}
        )
    
    if len(samples) != len(ground_truth):
        return MetricResult(
            name="Key Match Rate",
            value=0.0,
            numerator=0,
            denominator=0,
            details={"error": f"Length mismatch: {len(samples)} vs {len(ground_truth)}"}
        )
    
    matches = 0
    mismatches = []
    
    for gen, gt in zip(samples, ground_truth):
        gen_key = normalize_key(gen.get("key", ""))
        gt_key = normalize_key(gt.get("key", ""))
        
        # Also compare mode
        gen_mode = gen.get("mode", "major").lower()
        gt_mode = gt.get("mode", "major").lower()
        
        if gen_key == gt_key and gen_mode == gt_mode:
            matches += 1
        else:
            mismatches.append({
                "expected": f"{gt_key} {gt_mode}",
                "got": f"{gen_key} {gen_mode}"
            })
    
    total = len(samples)
    rate = matches / total if total > 0 else 0.0
    
    return MetricResult(
        name="Key Match Rate",
        value=rate,
        numerator=matches,
        denominator=total,
        details={"mismatches": mismatches[:10]}  # Limit to 10 examples
    )


def genre_match_rate(samples: List[Dict], ground_truth: List[Dict]) -> MetricResult:
    """
    Calculate how often the generated genre matches the requested genre.
    
    Args:
        samples: Generated outputs (each with "genre" field)
        ground_truth: Original prompts/labels (each with "genre" field)
        
    Returns:
        MetricResult with match rate
    """
    if not samples or not ground_truth:
        return MetricResult(
            name="Genre Match Rate",
            value=0.0,
            numerator=0,
            denominator=0,
            details={"error": "Missing samples or ground truth"}
        )
    
    matches = 0
    mismatches = []
    
    for gen, gt in zip(samples, ground_truth):
        gen_genre = gen.get("genre", "").lower()
        gt_genre = gt.get("genre", "").lower()
        
        if gen_genre == gt_genre:
            matches += 1
        else:
            mismatches.append({
                "expected": gt_genre,
                "got": gen_genre
            })
    
    total = len(samples)
    rate = matches / total if total > 0 else 0.0
    
    return MetricResult(
        name="Genre Match Rate",
        value=rate,
        numerator=matches,
        denominator=total,
        details={"mismatches": mismatches[:10]}
    )


def emotion_match_rate(samples: List[Dict], ground_truth: List[Dict]) -> MetricResult:
    """
    Calculate how often the generated emotion matches the requested emotion.
    
    Args:
        samples: Generated outputs (each with "emotion" field)
        ground_truth: Original prompts/labels (each with "emotion" field)
        
    Returns:
        MetricResult with match rate
    """
    if not samples or not ground_truth:
        return MetricResult(
            name="Emotion Match Rate",
            value=0.0,
            numerator=0,
            denominator=0,
            details={"error": "Missing samples or ground truth"}
        )
    
    matches = 0
    mismatches = []
    
    for gen, gt in zip(samples, ground_truth):
        gen_emotion = gen.get("emotion", "").lower()
        gt_emotion = gt.get("emotion", "").lower()
        
        if gen_emotion == gt_emotion:
            matches += 1
        else:
            mismatches.append({
                "expected": gt_emotion,
                "got": gen_emotion
            })
    
    total = len(samples)
    rate = matches / total if total > 0 else 0.0
    
    return MetricResult(
        name="Emotion Match Rate",
        value=rate,
        numerator=matches,
        denominator=total,
        details={"mismatches": mismatches[:10]}
    )


# =============================================================================
# SECTION 3: DIVERSITY METRICS
# =============================================================================

def unique_progression_ratio(samples: List[Dict]) -> MetricResult:
    """
    Calculate the ratio of unique chord progressions to total samples.
    
    A chord progression is represented as a tuple of chords.
    Two progressions are "the same" if they have the exact same chords
    in the exact same order.
    
    Args:
        samples: List of generation results with "chords" field
        
    Returns:
        MetricResult with uniqueness ratio
        
    Example:
        If 100 samples produce 80 unique progressions → ratio = 0.80
        
    Interpretation:
        - 1.0 = Every output is unique (maximum diversity)
        - 0.1 = Only 10% of outputs are unique (low diversity)
    """
    if not samples:
        return MetricResult(
            name="Unique Progression Ratio",
            value=0.0,
            numerator=0,
            denominator=0,
            details={"error": "No samples provided"}
        )
    
    # Convert chord lists to tuples for hashing
    progressions = []
    for sample in samples:
        chords = sample.get("chords", [])
        if chords:
            progressions.append(tuple(chords))
    
    unique_progressions = set(progressions)
    
    total = len(progressions)
    unique = len(unique_progressions)
    ratio = unique / total if total > 0 else 0.0
    
    # Find most common progressions
    progression_counts = Counter(progressions)
    most_common = progression_counts.most_common(5)
    
    return MetricResult(
        name="Unique Progression Ratio",
        value=ratio,
        numerator=unique,
        denominator=total,
        details={
            "unique_count": unique,
            "total_count": total,
            "most_common": [
                {"progression": list(prog), "count": count}
                for prog, count in most_common
            ]
        }
    )


def unique_pattern_ratio(samples: List[Dict]) -> MetricResult:
    """
    Calculate the ratio of unique strumming patterns to total samples.
    
    Args:
        samples: List of generation results with "strum_pattern" field
        
    Returns:
        MetricResult with uniqueness ratio
        
    Note:
        With only 8 positions and 3 characters (D, U, _), there are
        theoretically 3^8 = 6,561 possible patterns. In practice,
        musically sensible patterns are far fewer.
    """
    if not samples:
        return MetricResult(
            name="Unique Pattern Ratio",
            value=0.0,
            numerator=0,
            denominator=0,
            details={"error": "No samples provided"}
        )
    
    patterns = [sample.get("strum_pattern", "") for sample in samples]
    patterns = [p for p in patterns if p]  # Filter empty
    
    unique_patterns = set(patterns)
    
    total = len(patterns)
    unique = len(unique_patterns)
    ratio = unique / total if total > 0 else 0.0
    
    # Find most common patterns
    pattern_counts = Counter(patterns)
    most_common = pattern_counts.most_common(5)
    
    return MetricResult(
        name="Unique Pattern Ratio",
        value=ratio,
        numerator=unique,
        denominator=total,
        details={
            "unique_count": unique,
            "total_count": total,
            "most_common": [
                {"pattern": pat, "count": count}
                for pat, count in most_common
            ]
        }
    )


def chord_distribution_entropy(samples: List[Dict]) -> MetricResult:
    """
    Calculate the entropy of chord usage across all samples.
    
    Higher entropy = chords are more evenly distributed (diverse)
    Lower entropy = a few chords dominate (less diverse)
    
    Args:
        samples: List of generation results with "chords" field
        
    Returns:
        MetricResult with raw entropy and normalized entropy (0-1)
    """
    if not samples:
        return MetricResult(
            name="Chord Distribution Entropy",
            value=0.0,
            details={"error": "No samples provided"}
        )
    
    # Collect all chords
    all_chords = []
    for sample in samples:
        chords = sample.get("chords", [])
        all_chords.extend(chords)
    
    if not all_chords:
        return MetricResult(
            name="Chord Distribution Entropy",
            value=0.0,
            details={"error": "No chords found in samples"}
        )
    
    # Calculate entropy
    raw_entropy = calculate_entropy(all_chords)
    unique_count = len(set(all_chords))
    normalized = normalize_entropy(raw_entropy, unique_count)
    
    # Chord frequency distribution
    chord_counts = Counter(all_chords)
    total_chords = len(all_chords)
    
    # Top 10 most used chords
    top_chords = chord_counts.most_common(10)
    
    return MetricResult(
        name="Chord Distribution Entropy",
        value=normalized,  # Use normalized as primary value
        details={
            "raw_entropy": raw_entropy,
            "normalized_entropy": normalized,
            "unique_chords_used": unique_count,
            "total_chord_tokens": total_chords,
            "max_possible_entropy": math.log2(unique_count) if unique_count > 1 else 0,
            "top_10_chords": [
                {"chord": chord, "count": count, "percentage": count/total_chords}
                for chord, count in top_chords
            ]
        }
    )


def pattern_distribution_entropy(samples: List[Dict]) -> MetricResult:
    """
    Calculate the entropy of strumming pattern usage.
    
    Args:
        samples: List of generation results with "strum_pattern" field
        
    Returns:
        MetricResult with entropy values
    """
    if not samples:
        return MetricResult(
            name="Pattern Distribution Entropy",
            value=0.0,
            details={"error": "No samples provided"}
        )
    
    patterns = [sample.get("strum_pattern", "") for sample in samples]
    patterns = [p for p in patterns if p]
    
    if not patterns:
        return MetricResult(
            name="Pattern Distribution Entropy",
            value=0.0,
            details={"error": "No patterns found"}
        )
    
    raw_entropy = calculate_entropy(patterns)
    unique_count = len(set(patterns))
    normalized = normalize_entropy(raw_entropy, unique_count)
    
    pattern_counts = Counter(patterns)
    top_patterns = pattern_counts.most_common(5)
    
    return MetricResult(
        name="Pattern Distribution Entropy",
        value=normalized,
        details={
            "raw_entropy": raw_entropy,
            "normalized_entropy": normalized,
            "unique_patterns": unique_count,
            "total_patterns": len(patterns),
            "top_5_patterns": [
                {"pattern": pat, "count": count}
                for pat, count in top_patterns
            ]
        }
    )


# =============================================================================
# SECTION 4: AGGREGATE FUNCTIONS
# =============================================================================

def compute_all_metrics(
    generated_samples: List[Dict],
    ground_truth: Optional[List[Dict]] = None
) -> EvaluationReport:
    """
    Compute all evaluation metrics at once.
    
    This is the main entry point for evaluation. It computes:
    - All correctness metrics
    - All prompt adherence metrics (if ground_truth provided)
    - All diversity metrics
    
    Args:
        generated_samples: List of generation output dictionaries
        ground_truth: Optional list of original test samples for comparison
        
    Returns:
        EvaluationReport containing all metrics
        
    Usage:
        >>> samples = [generate_guitar_part(p) for p in test_prompts]
        >>> report = compute_all_metrics(samples, test_data)
        >>> print(report)
    """
    report = EvaluationReport()
    report.sample_count = len(generated_samples)
    
    # Count sources
    sources = [s.get("source", "unknown") for s in generated_samples]
    report.source_breakdown = dict(Counter(sources))
    
    # ─────────────────────────────────────────────────────────────────────────
    # Correctness Metrics
    # ─────────────────────────────────────────────────────────────────────────
    report.correctness["chord_validity"] = chord_validity_rate(generated_samples)
    report.correctness["pattern_validity"] = pattern_validity_rate(generated_samples)
    report.correctness["key_adherence"] = key_adherence_rate(generated_samples)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Prompt Adherence Metrics (require ground truth)
    # ─────────────────────────────────────────────────────────────────────────
    if ground_truth:
        report.prompt_adherence["key_match"] = key_match_rate(
            generated_samples, ground_truth
        )
        report.prompt_adherence["genre_match"] = genre_match_rate(
            generated_samples, ground_truth
        )
        report.prompt_adherence["emotion_match"] = emotion_match_rate(
            generated_samples, ground_truth
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Diversity Metrics
    # ─────────────────────────────────────────────────────────────────────────
    report.diversity["unique_progressions"] = unique_progression_ratio(generated_samples)
    report.diversity["unique_patterns"] = unique_pattern_ratio(generated_samples)
    report.diversity["chord_entropy"] = chord_distribution_entropy(generated_samples)
    report.diversity["pattern_entropy"] = pattern_distribution_entropy(generated_samples)
    
    return report


def format_metrics_for_thesis(report: EvaluationReport, system_name: str = "System") -> str:
    """
    Format metrics as a LaTeX-ready table for thesis.
    
    Args:
        report: EvaluationReport to format
        system_name: Name to display in table header
        
    Returns:
        String formatted for thesis inclusion
    """
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"METRICS FOR: {system_name}")
    lines.append(f"Samples evaluated: {report.sample_count}")
    lines.append(f"{'='*60}")
    
    # Correctness table
    lines.append("\nTable: Correctness Metrics")
    lines.append("-" * 50)
    lines.append(f"{'Metric':<30} {'Value':>10} {'Count':>10}")
    lines.append("-" * 50)
    for name, metric in report.correctness.items():
        lines.append(f"{metric.name:<30} {metric.value:>9.1%} {metric.numerator:>4}/{metric.denominator:<4}")
    
    # Prompt adherence table
    if report.prompt_adherence:
        lines.append("\nTable: Prompt Adherence Metrics")
        lines.append("-" * 50)
        lines.append(f"{'Metric':<30} {'Value':>10} {'Count':>10}")
        lines.append("-" * 50)
        for name, metric in report.prompt_adherence.items():
            lines.append(f"{metric.name:<30} {metric.value:>9.1%} {metric.numerator:>4}/{metric.denominator:<4}")
    
    # Diversity table
    lines.append("\nTable: Diversity Metrics")
    lines.append("-" * 50)
    lines.append(f"{'Metric':<35} {'Value':>15}")
    lines.append("-" * 50)
    for name, metric in report.diversity.items():
        if "ratio" in name.lower():
            lines.append(f"{metric.name:<35} {metric.value:>14.1%}")
        else:
            lines.append(f"{metric.name:<35} {metric.value:>14.3f}")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing metrics.py")
    print("=" * 60)
    
    # Create some test samples
    test_samples = [
        {
            "chords": ["G", "D", "Em", "C"],
            "strum_pattern": "D_DU_DU_",
            "key": "G",
            "mode": "major",
            "genre": "folk",
            "emotion": "upbeat",
            "source": "neural"
        },
        {
            "chords": ["G", "C", "D", "G"],
            "strum_pattern": "DDUUDDUU",
            "key": "G",
            "mode": "major",
            "genre": "rock",
            "emotion": "energetic",
            "source": "neural"
        },
        {
            "chords": ["Am", "F", "C", "G"],
            "strum_pattern": "D_D_D_D_",
            "key": "Am",
            "mode": "minor",
            "genre": "pop",
            "emotion": "melancholic",
            "source": "rule_based"
        },
        {
            "chords": ["Am", "Dm", "G", "C"],
            "strum_pattern": "D___D___",
            "key": "Am",
            "mode": "minor",
            "genre": "ballad",
            "emotion": "peaceful",
            "source": "neural"
        },
    ]
    
    ground_truth = [
        {"key": "G", "mode": "major", "genre": "folk", "emotion": "upbeat"},
        {"key": "G", "mode": "major", "genre": "rock", "emotion": "energetic"},
        {"key": "Am", "mode": "minor", "genre": "pop", "emotion": "melancholic"},
        {"key": "Am", "mode": "minor", "genre": "ballad", "emotion": "peaceful"},
    ]
    
    print("\n--- Test 1: Individual Metrics ---")
    
    print("\n✓ Chord Validity:")
    print(f"  {chord_validity_rate(test_samples)}")
    
    print("\n✓ Pattern Validity:")
    print(f"  {pattern_validity_rate(test_samples)}")
    
    print("\n✓ Key Adherence:")
    result = key_adherence_rate(test_samples)
    print(f"  {result}")
    
    print("\n✓ Unique Progressions:")
    print(f"  {unique_progression_ratio(test_samples)}")
    
    print("\n✓ Chord Entropy:")
    entropy_result = chord_distribution_entropy(test_samples)
    print(f"  {entropy_result}")
    print(f"  Raw entropy: {entropy_result.details['raw_entropy']:.3f}")
    
    print("\n--- Test 2: Full Evaluation Report ---")
    report = compute_all_metrics(test_samples, ground_truth)
    print(report)
    
    print("\n--- Test 3: Thesis-Formatted Output ---")
    print(format_metrics_for_thesis(report, "Test System"))
    
    print("\n--- Test 4: Helper Functions ---")
    print(f"  Diatonic chords in G major: {get_diatonic_chords('G', 'major')}")
    print(f"  Diatonic chords in Am minor: {get_diatonic_chords('A', 'minor')}")
    print(f"  Entropy of ['A','A','A','A']: {calculate_entropy(['A','A','A','A']):.3f}")
    print(f"  Entropy of ['A','B','C','D']: {calculate_entropy(['A','B','C','D']):.3f}")
    
    print("\n" + "=" * 60)
    print("All tests complete!")
    print("=" * 60)
