"""
Harmony Module - Music Theory Rules for Chord Progressions

This module encodes music theory knowledge for generating chord progressions.
It can:
    1. Build major and minor scales from any root note
    2. Derive diatonic chords for any key
    3. Select appropriate chord progressions based on genre/emotion
    4. Validate if a chord belongs to a given key

This is the "brain" of our rule-based baseline system.

Author: Rohan Rajendra Dhanawade
Thesis: A Conversational AI System for Symbolic Guitar Strumming Pattern 
        and Chord Progression Generation
"""

from typing import List, Dict, Tuple, Optional
import random


# =============================================================================
# CONSTANTS: The Building Blocks of Music Theory
# =============================================================================

# The 12 notes in Western music, using sharps (we'll handle flats separately)
CHROMATIC_SCALE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Mapping of flat notes to their sharp equivalents (enharmonic equivalents)
FLAT_TO_SHARP = {
    "Db": "C#",
    "Eb": "D#",
    "Gb": "F#",
    "Ab": "G#",
    "Bb": "A#",
}

# Scale formulas as semitone intervals from the root
SCALE_FORMULAS = {
    "major": [0, 2, 4, 5, 7, 9, 11],      # W-W-H-W-W-W-H
    "minor": [0, 2, 3, 5, 7, 8, 10],      # W-H-W-W-H-W-W (natural minor)
}

# Chord qualities for each scale degree
CHORD_QUALITIES = {
    "major": ["", "m", "m", "", "", "m", "dim"],
    "minor": ["m", "dim", "", "m", "m", "", ""],
}

# Roman numeral labels (for documentation/display purposes)
ROMAN_NUMERALS = {
    "major": ["I", "ii", "iii", "IV", "V", "vi", "vii°"],
    "minor": ["i", "ii°", "III", "iv", "v", "VI", "VII"],
}


# =============================================================================
# COMMON CHORD PROGRESSIONS
# =============================================================================

PROGRESSIONS = {
    "major": {
        "I-IV-V-I": [1, 4, 5, 1],
        "I-V-vi-IV": [1, 5, 6, 4],
        "I-vi-IV-V": [1, 6, 4, 5],
        "I-IV-vi-V": [1, 4, 6, 5],
        "I-V-IV-I": [1, 5, 4, 1],
        "I-ii-V-I": [1, 2, 5, 1],
        "I-IV-I-V": [1, 4, 1, 5],
        "I-iii-IV-V": [1, 3, 4, 5],
    },
    "minor": {
        "i-iv-v-i": [1, 4, 5, 1],
        "i-VI-III-VII": [1, 6, 3, 7],
        "i-iv-VII-III": [1, 4, 7, 3],
        "i-VII-VI-VII": [1, 7, 6, 7],
        "i-iv-i-v": [1, 4, 1, 5],
        "i-VI-iv-V": [1, 6, 4, 5],
        "i-III-VII-VI": [1, 3, 7, 6],
        "i-ii°-V-i": [1, 2, 5, 1],
    },
}

# Map genres and emotions to preferred progressions
GENRE_PROGRESSIONS = {
    "major": {
        "pop": ["I-V-vi-IV", "I-IV-vi-V", "I-vi-IV-V"],
        "rock": ["I-IV-V-I", "I-V-IV-I", "I-IV-I-V"],
        "folk": ["I-IV-V-I", "I-V-IV-I", "I-IV-I-V", "I-iii-IV-V"],
        "country": ["I-IV-V-I", "I-V-IV-I", "I-IV-I-V"],
        "ballad": ["I-V-vi-IV", "I-vi-IV-V", "I-iii-IV-V"],
        "blues": ["I-IV-V-I", "I-IV-I-V"],
        "jazz": ["I-ii-V-I", "I-vi-IV-V"],
        "indie": ["I-V-vi-IV", "I-iii-IV-V", "I-IV-vi-V"],
        "acoustic": ["I-V-vi-IV", "I-IV-V-I", "I-iii-IV-V"],
    },
    "minor": {
        "pop": ["i-VI-III-VII", "i-iv-VII-III"],
        "rock": ["i-VII-VI-VII", "i-iv-v-i", "i-iv-VII-III"],
        "folk": ["i-iv-v-i", "i-VII-VI-VII"],
        "country": ["i-iv-v-i", "i-VI-iv-V"],
        "ballad": ["i-VI-III-VII", "i-iv-VII-III", "i-III-VII-VI"],
        "blues": ["i-iv-v-i", "i-iv-i-v"],
        "jazz": ["i-ii°-V-i", "i-VI-iv-V"],
        "indie": ["i-VI-III-VII", "i-III-VII-VI"],
        "acoustic": ["i-iv-v-i", "i-VI-III-VII"],
    },
}

EMOTION_PROGRESSIONS = {
    "major": {
        "upbeat": ["I-V-vi-IV", "I-IV-V-I", "I-IV-I-V"],
        "melancholic": ["I-vi-IV-V", "I-iii-IV-V"],
        "mellow": ["I-iii-IV-V", "I-vi-IV-V"],
        "energetic": ["I-IV-V-I", "I-V-IV-I"],
        "peaceful": ["I-iii-IV-V", "I-IV-vi-V"],
        "dramatic": ["I-V-vi-IV", "I-IV-vi-V"],
        "hopeful": ["I-V-vi-IV", "I-IV-V-I"],
        "nostalgic": ["I-vi-IV-V", "I-V-vi-IV"],
    },
    "minor": {
        "upbeat": ["i-VII-VI-VII", "i-iv-VII-III"],
        "melancholic": ["i-VI-III-VII", "i-iv-VII-III", "i-III-VII-VI"],
        "mellow": ["i-iv-v-i", "i-VI-III-VII"],
        "energetic": ["i-VII-VI-VII", "i-iv-v-i"],
        "peaceful": ["i-iv-v-i", "i-VI-III-VII"],
        "dramatic": ["i-iv-VII-III", "i-VI-iv-V"],
        "hopeful": ["i-III-VII-VI", "i-VI-III-VII"],
        "nostalgic": ["i-VI-III-VII", "i-iv-VII-III"],
    },
}


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def normalize_note(note: str) -> str:
    """Convert a note name to its standard sharp form."""
    if len(note) == 1:
        note = note.upper()
    elif len(note) == 2:
        note = note[0].upper() + note[1].lower()
    else:
        raise ValueError(f"Invalid note format: '{note}'")
    
    if note in FLAT_TO_SHARP:
        note = FLAT_TO_SHARP[note]
    
    if note not in CHROMATIC_SCALE:
        raise ValueError(f"Unknown note: '{note}'. Valid notes are: {CHROMATIC_SCALE}")
    
    return note


def get_note_index(note: str) -> int:
    """Get the index of a note in the chromatic scale (0-11)."""
    normalized = normalize_note(note)
    return CHROMATIC_SCALE.index(normalized)


def build_scale(root: str, mode: str = "major") -> List[str]:
    """Build a scale from a root note."""
    if mode not in SCALE_FORMULAS:
        raise ValueError(f"Mode must be 'major' or 'minor'. Got: '{mode}'")
    
    root_index = get_note_index(root)
    formula = SCALE_FORMULAS[mode]
    
    scale = []
    for interval in formula:
        note_index = (root_index + interval) % 12
        scale.append(CHROMATIC_SCALE[note_index])
    
    return scale


def get_diatonic_chords(key: str, mode: str = "major") -> List[str]:
    """Get all 7 diatonic chords for a given key."""
    scale = build_scale(key, mode)
    qualities = CHORD_QUALITIES[mode]
    
    chords = []
    for note, quality in zip(scale, qualities):
        chords.append(note + quality)
    
    return chords


def get_chord_from_degree(key: str, mode: str, degree: int) -> str:
    """Get a specific chord by its scale degree."""
    if not 1 <= degree <= 7:
        raise ValueError(f"Degree must be 1-7. Got: {degree}")
    
    chords = get_diatonic_chords(key, mode)
    return chords[degree - 1]


def degrees_to_chords(key: str, mode: str, degrees: List[int]) -> List[str]:
    """Convert a list of scale degrees to actual chord symbols."""
    return [get_chord_from_degree(key, mode, d) for d in degrees]


# =============================================================================
# PROGRESSION SELECTION
# =============================================================================

def select_progression(
    key: str,
    mode: str = "major",
    genre: Optional[str] = None,
    emotion: Optional[str] = None
) -> List[str]:
    """Select and return a chord progression based on key, mode, genre, and emotion."""
    # If genre="folk" in major key, candidates include:
    # ["I-IV-V-I", "I-V-IV-I", "I-IV-I-V", "I-iii-IV-V"]

    # If emotion="upbeat" in major key, candidates include:
    # ["I-V-vi-IV", "I-IV-V-I", "I-IV-I-V"]

    # The function finds the intersection or union, picks one randomly,
    # then calls degrees_to_chords() to convert to actual chords
    mode = mode.lower()
    
    candidates = set()
    
    if genre and genre.lower() in GENRE_PROGRESSIONS.get(mode, {}):
        candidates.update(GENRE_PROGRESSIONS[mode][genre.lower()])
    
    if emotion and emotion.lower() in EMOTION_PROGRESSIONS.get(mode, {}):
        candidates.update(EMOTION_PROGRESSIONS[mode][emotion.lower()])
    
    if not candidates:
        candidates = set(PROGRESSIONS[mode].keys())
    
    progression_name = random.choice(list(candidates))
    degrees = PROGRESSIONS[mode][progression_name]
    
    return degrees_to_chords(key, mode, degrees)


def is_chord_in_key(chord: str, key: str, mode: str = "major") -> bool:
    """Check if a chord belongs to a given key."""
    diatonic = get_diatonic_chords(key, mode)
    return chord in diatonic


def validate_progression(chords: List[str], key: str, mode: str = "major") -> Tuple[bool, List[str]]:
    """Validate a chord progression against a key."""
    invalid = [c for c in chords if not is_chord_in_key(c, key, mode)]
    return (len(invalid) == 0, invalid)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_relative_key(key: str, mode: str) -> Tuple[str, str]:
    """Get the relative major/minor of a key."""
    if mode == "major":
        root_index = get_note_index(key)
        relative_index = (root_index + 9) % 12
        return (CHROMATIC_SCALE[relative_index], "minor")
    else:
        root_index = get_note_index(key)
        relative_index = (root_index + 3) % 12
        return (CHROMATIC_SCALE[relative_index], "major")


def get_parallel_key(key: str, mode: str) -> Tuple[str, str]:
    """Get the parallel major/minor of a key."""
    new_mode = "minor" if mode == "major" else "major"
    return (key, new_mode)


def print_key_info(key: str, mode: str = "major") -> None:
    """Print detailed information about a key."""
    print(f"\n{'='*60}")
    print(f"Key: {key} {mode}")
    print(f"{'='*60}")
    
    scale = build_scale(key, mode)
    print(f"\nScale: {' - '.join(scale)}")
    
    chords = get_diatonic_chords(key, mode)
    numerals = ROMAN_NUMERALS[mode]
    print(f"\nDiatonic Chords:")
    for numeral, chord in zip(numerals, chords):
        print(f"  {numeral:6} = {chord}")
    
    rel_key, rel_mode = get_relative_key(key, mode)
    print(f"\nRelative key: {rel_key} {rel_mode}")
    
    par_key, par_mode = get_parallel_key(key, mode)
    print(f"Parallel key: {par_key} {par_mode}")


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing harmony.py")
    print("=" * 60)
    
    print("\n✓ Test 1: Building scales")
    print(f"  G major: {build_scale('G', 'major')}")
    print(f"  A minor: {build_scale('A', 'minor')}")
    
    print("\n✓ Test 2: Diatonic chords")
    print(f"  G major: {get_diatonic_chords('G', 'major')}")
    print(f"  A minor: {get_diatonic_chords('A', 'minor')}")
    
    print("\n✓ Test 3: Degrees to chords")
    print(f"  G major [1,4,5,1]: {degrees_to_chords('G', 'major', [1,4,5,1])}")
    print(f"  A minor [1,6,3,7]: {degrees_to_chords('A', 'minor', [1,6,3,7])}")
    
    print("\n✓ Test 4: Selecting progressions")
    print(f"  G major folk: {select_progression('G', 'major', genre='folk')}")
    print(f"  A minor melancholic: {select_progression('A', 'minor', emotion='melancholic')}")
    
    print("\n✓ Test 5: Chord validation")
    valid, invalid = validate_progression(["G", "C", "D", "Em"], "G", "major")
    print(f"  ['G','C','D','Em'] in G major: valid={valid}, invalid={invalid}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)