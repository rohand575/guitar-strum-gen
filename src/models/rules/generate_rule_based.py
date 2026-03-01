"""
Rule-Based Generator - Main Entry Point for Rule-Based System

This module combines all rule-based components to generate complete
guitar chord progressions and strumming patterns from natural language.

Author: Rohan Rajendra Dhanawade
"""

from typing import Optional
import uuid

from src.rules.harmony import select_progression, validate_progression
from src.rules.strumming import select_strumming_pattern, pick_tempo
from src.rules.prompt_parser import parse_prompt, apply_defaults, ParsedFeatures
from src.data.schema import GuitarSample, create_sample


# =============================================================================
# MAIN GENERATOR FUNCTION
# =============================================================================

def generate_rule_based(
    prompt: str,
    verbose: bool = False
) -> GuitarSample:
    """Generate a complete guitar part from a natural language prompt."""
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"RULE-BASED GENERATOR")
        print(f"{'='*60}")
        print(f"\nInput prompt: \"{prompt}\"")
        print(f"\n--- Step 1: Parsing prompt ---")
    
    features = parse_prompt(prompt)
    
    if verbose:
        print(f"  Raw extraction: {features}")
    
    features = apply_defaults(features)
    
    if verbose:
        print(f"  After defaults: {features}")
    
    if verbose:
        print(f"\n--- Step 2: Generating chord progression ---")
        print(f"  Key: {features.key} {features.mode}")
        print(f"  Genre: {features.genre}, Emotion: {features.emotion}")
    
    chords = select_progression(
        key=features.key,
        mode=features.mode,
        genre=features.genre,
        emotion=features.emotion
    )
    
    if verbose:
        print(f"  Generated chords: {chords}")
        valid, invalid = validate_progression(chords, features.key, features.mode)
        print(f"  Validation: valid={valid}, invalid_chords={invalid}")
    
    if verbose:
        print(f"\n--- Step 3: Selecting strumming pattern ---")
        print(f"  Genre: {features.genre}, Tempo: {features.tempo}, Emotion: {features.emotion}")
    
    strum_pattern = select_strumming_pattern(
        genre=features.genre,
        tempo=features.tempo,
        emotion=features.emotion
    )
    
    if verbose:
        print(f"  Selected pattern: {strum_pattern}")
    
    if verbose:
        print(f"\n--- Step 4: Creating GuitarSample ---")
    
    sample_id = f"rule_{uuid.uuid4().hex[:8]}"
    
    if features.mode == "minor":
        key_str = features.key + "m"
    else:
        key_str = features.key
    
    sample = create_sample(
        id=sample_id,
        prompt=prompt,
        chords=chords,
        strum_pattern=strum_pattern,
        tempo=features.tempo,
        genre=features.genre,
        emotion=features.emotion,
        key=key_str,
        mode=features.mode,
        time_signature="4/4"
    )
    
    if verbose:
        print(f"  Sample ID: {sample.id}")
        print(f"  Final output:")
        print(f"    Chords: {sample.chords}")
        print(f"    Pattern: {sample.strum_pattern}")
        print(f"    Tempo: {sample.tempo} BPM")
        print(f"    Key: {sample.key} {sample.mode}")
        print(f"\n{'='*60}")
    
    return sample


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def format_as_chord_sheet(sample: GuitarSample) -> str:
    """Format a GuitarSample as a human-readable chord sheet."""
    lines = []
    lines.append("╔" + "═" * 50 + "╗")
    lines.append("║" + " CHORD SHEET ".center(50) + "║")
    lines.append("╠" + "═" * 50 + "╣")
    
    prompt_display = sample.prompt[:45] + "..." if len(sample.prompt) > 45 else sample.prompt
    lines.append(f"║ Prompt: {prompt_display}".ljust(51) + "║")
    lines.append("╠" + "─" * 50 + "╣")
    
    lines.append(f"║ Key: {sample.key} {sample.mode}".ljust(51) + "║")
    lines.append(f"║ Tempo: {sample.tempo} BPM".ljust(51) + "║")
    lines.append(f"║ Genre: {sample.genre} | Emotion: {sample.emotion}".ljust(51) + "║")
    lines.append("╠" + "─" * 50 + "╣")
    
    chord_str = " → ".join(sample.chords)
    lines.append(f"║ Chords: {chord_str}".ljust(51) + "║")
    lines.append("╠" + "─" * 50 + "╣")
    
    lines.append(f"║ Strum Pattern:".ljust(51) + "║")
    lines.append(f"║   Beat:  1  &  2  &  3  &  4  &".ljust(51) + "║")
    pattern_display = "   ".join(sample.strum_pattern)
    lines.append(f"║   Strum: {pattern_display}".ljust(51) + "║")
    
    lines.append("╚" + "═" * 50 + "╝")
    
    return "\n".join(lines)


def format_as_json(sample: GuitarSample, indent: int = 2) -> str:
    """Format a GuitarSample as JSON."""
    return sample.model_dump_json(indent=indent)


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command-line interface for the rule-based generator."""
    import sys
    
    if len(sys.argv) < 2:
        print("=" * 60)
        print("RULE-BASED GUITAR GENERATOR - Demo Mode")
        print("=" * 60)
        print("\nUsage: python -m src.rules.generate_rule_based \"your prompt\"")
        print("\nRunning demo with example prompts...\n")
        
        demo_prompts = [
            "upbeat folk song in G major",
            "melancholic ballad in Am, slow tempo",
            "energetic rock progression in E minor",
            "chill acoustic vibes, moderate tempo",
        ]
        
        for prompt in demo_prompts:
            sample = generate_rule_based(prompt, verbose=False)
            print(format_as_chord_sheet(sample))
            print()
    else:
        prompt = " ".join(sys.argv[1:])
        sample = generate_rule_based(prompt, verbose=True)
        print("\n" + format_as_chord_sheet(sample))


if __name__ == "__main__":
    main()