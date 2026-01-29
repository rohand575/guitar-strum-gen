"""
Schema definitions for Guitar Strum Generator dataset.

This module defines the Pydantic models that validate and structure
our training data. Every sample in the dataset must conform to these schemas.

Author: Rohan Rajendra Dhanawade
Thesis: A Conversational AI System for Symbolic Guitar Strumming Pattern 
        and Chord Progression Generation
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
import re


# =============================================================================
# VALID OPTIONS (These define what's allowed in our dataset)
# =============================================================================

VALID_GENRES = [
    "pop", "rock", "folk", "ballad", "country", 
    "blues", "jazz", "indie", "acoustic"
]

VALID_EMOTIONS = [
    "upbeat", "melancholic", "mellow", "energetic", 
    "peaceful", "dramatic", "hopeful", "nostalgic"
]

VALID_MODES = ["major", "minor"]

VALID_TIME_SIGNATURES = ["4/4", "3/4", "6/8"]

# All valid chord symbols your system will recognize
# Major chords: C, D, E, F, G, A, B (and sharps/flats)
# Minor chords: Cm, Dm, Em, Fm, Gm, Am, Bm (and sharps/flats)
# Seventh chords: C7, Cmaj7, Cm7, etc.
VALID_CHORD_ROOTS = ["C", "C#", "Db", "D", "D#", "Eb", "E", "F", 
                     "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"]

VALID_CHORD_SUFFIXES = [
    "",      # Major (e.g., "C")
    "m",     # Minor (e.g., "Am")
    "7",     # Dominant 7th (e.g., "G7")
    "maj7",  # Major 7th (e.g., "Cmaj7")
    "m7",    # Minor 7th (e.g., "Am7")
    "dim",   # Diminished (e.g., "Bdim")
    "aug",   # Augmented (e.g., "Caug")
    "sus2",  # Suspended 2nd (e.g., "Dsus2")
    "sus4",  # Suspended 4th (e.g., "Dsus4")
    "add9",  # Added 9th (e.g., "Cadd9")
]

# Pattern for valid strum characters: exactly 8 characters of D, U, or _
STRUM_PATTERN_REGEX = re.compile(r'^[DU_]{8}$')


# =============================================================================
# HELPER FUNCTION
# =============================================================================

def is_valid_chord(chord: str) -> bool:
    """
    Check if a chord symbol is valid.
    
    Valid chords consist of:
    - A root note (C, C#, Db, D, etc.)
    - An optional suffix (m, 7, maj7, m7, dim, etc.)
    
    Examples:
        is_valid_chord("C")     → True
        is_valid_chord("Am")    → True
        is_valid_chord("G7")    → True
        is_valid_chord("Xyz")   → False
    
    Args:
        chord: The chord symbol to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Check each possible root
    for root in VALID_CHORD_ROOTS:
        if chord.startswith(root):
            # Get the suffix (everything after the root)
            suffix = chord[len(root):]
            # Check if suffix is valid
            if suffix in VALID_CHORD_SUFFIXES:
                return True
    return False


# =============================================================================
# MAIN SCHEMA
# =============================================================================

class GuitarSample(BaseModel):
    """
    A single training example for the guitar strum generator.
    
    This represents one prompt → output pair in our dataset.
    
    Attributes:
        id: Unique identifier for this sample
        prompt: Natural language description of desired output
        chords: List of chord symbols in the progression
        strum_pattern: 8-character strumming pattern using D, U, and _
        tempo: Tempo in beats per minute (BPM)
        time_signature: Time signature (default "4/4")
        genre: Musical genre/style
        emotion: Emotional quality of the piece
        key: Musical key (e.g., 'C', 'G', 'Am')
        mode: Major or minor mode
    
    Example:
        >>> sample = GuitarSample(
        ...     id="sample_001",
        ...     prompt="upbeat folk strum in G major at moderate tempo",
        ...     chords=["G", "D", "Em", "C"],
        ...     strum_pattern="D_DU_UD_",
        ...     tempo=110,
        ...     genre="folk",
        ...     emotion="upbeat",
        ...     key="G",
        ...     mode="major"
        ... )
    """
    
    # ---------------------------
    # Required Fields
    # ---------------------------
    
    id: str = Field(
        ...,  # ... means "required, no default"
        min_length=1,
        description="Unique identifier for this sample",
        examples=["sample_001", "folk_upbeat_003"]
    )
    
    prompt: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Natural language description of desired output",
        examples=["upbeat folk strum in G major at moderate tempo"]
    )
    
    chords: List[str] = Field(
        ...,
        min_length=1,
        max_length=8,
        description="List of chord symbols in the progression",
        examples=[["G", "D", "Em", "C"], ["Am", "F", "C", "G"]]
    )
    
    strum_pattern: str = Field(
        ...,
        description="8-character strumming pattern using D, U, and _",
        examples=["D_DU_UD_", "D_D_D_D_", "DDUUDDUU"]
    )
    
    tempo: int = Field(
        ...,
        ge=40,   # ge = greater than or equal (slowest reasonable tempo)
        le=200,  # le = less than or equal (fastest reasonable tempo)
        description="Tempo in beats per minute (BPM)",
        examples=[70, 110, 140]
    )
    
    time_signature: str = Field(
        default="4/4",
        description="Time signature of the pattern",
        examples=["4/4", "3/4"]
    )
    
    # ---------------------------
    # Metadata Fields
    # ---------------------------
    
    genre: str = Field(
        ...,
        description="Musical genre/style",
        examples=["folk", "pop", "rock", "ballad"]
    )
    
    emotion: str = Field(
        ...,
        description="Emotional quality of the piece",
        examples=["upbeat", "melancholic", "mellow"]
    )
    
    key: str = Field(
        ...,
        description="Musical key (e.g., 'C', 'G', 'Am')",
        examples=["C", "G", "Am", "Em"]
    )
    
    mode: str = Field(
        ...,
        description="Major or minor mode",
        examples=["major", "minor"]
    )
    
    # ---------------------------
    # Custom Validators
    # ---------------------------
    
    @field_validator('strum_pattern')
    @classmethod
    def validate_strum_pattern(cls, v: str) -> str:
        """Ensure strum pattern is exactly 8 characters of D, U, or _"""
        if not STRUM_PATTERN_REGEX.match(v):
            raise ValueError(
                f"Strum pattern must be exactly 8 characters using only D, U, and _. "
                f"Got: '{v}' (length: {len(v)})"
            )
        return v
    
    @field_validator('genre')
    @classmethod
    def validate_genre(cls, v: str) -> str:
        """Ensure genre is from our valid list (case-insensitive)"""
        v_lower = v.lower()
        if v_lower not in VALID_GENRES:
            raise ValueError(
                f"Genre must be one of {VALID_GENRES}. Got: '{v}'"
            )
        return v_lower  # Always store lowercase for consistency
    
    @field_validator('emotion')
    @classmethod
    def validate_emotion(cls, v: str) -> str:
        """Ensure emotion is from our valid list (case-insensitive)"""
        v_lower = v.lower()
        if v_lower not in VALID_EMOTIONS:
            raise ValueError(
                f"Emotion must be one of {VALID_EMOTIONS}. Got: '{v}'"
            )
        return v_lower
    
    @field_validator('mode')
    @classmethod
    def validate_mode(cls, v: str) -> str:
        """Ensure mode is major or minor (case-insensitive)"""
        v_lower = v.lower()
        if v_lower not in VALID_MODES:
            raise ValueError(
                f"Mode must be one of {VALID_MODES}. Got: '{v}'"
            )
        return v_lower
    
    @field_validator('time_signature')
    @classmethod
    def validate_time_signature(cls, v: str) -> str:
        """Ensure time signature is valid"""
        if v not in VALID_TIME_SIGNATURES:
            raise ValueError(
                f"Time signature must be one of {VALID_TIME_SIGNATURES}. Got: '{v}'"
            )
        return v
    
    @field_validator('chords')
    @classmethod
    def validate_chords(cls, v: List[str]) -> List[str]:
        """Ensure all chords in the list are valid chord symbols"""
        invalid_chords = [chord for chord in v if not is_valid_chord(chord)]
        if invalid_chords:
            raise ValueError(
                f"Invalid chord symbols found: {invalid_chords}. "
                f"Valid roots are: {VALID_CHORD_ROOTS}. "
                f"Valid suffixes are: {VALID_CHORD_SUFFIXES}"
            )
        return v
    
    @field_validator('key')
    @classmethod
    def validate_key(cls, v: str) -> str:
        """Ensure key is a valid chord symbol (can be major like 'C' or minor like 'Am')"""
        if not is_valid_chord(v):
            raise ValueError(
                f"Key must be a valid chord symbol (e.g., 'C', 'Am', 'G'). Got: '{v}'"
            )
        return v


# =============================================================================
# CONVENIENCE METHODS
# =============================================================================

def create_sample(
    id: str,
    prompt: str,
    chords: List[str],
    strum_pattern: str,
    tempo: int,
    genre: str,
    emotion: str,
    key: str,
    mode: str,
    time_signature: str = "4/4"
) -> GuitarSample:
    """
    Convenience function to create a validated GuitarSample.
    
    This is just a wrapper around GuitarSample() that makes the 
    parameter order clearer.
    
    Args:
        id: Unique identifier
        prompt: Natural language description
        chords: List of chord symbols
        strum_pattern: 8-character pattern (D, U, _)
        tempo: BPM (40-200)
        genre: One of VALID_GENRES
        emotion: One of VALID_EMOTIONS
        key: Musical key (e.g., "C", "Am")
        mode: "major" or "minor"
        time_signature: Default "4/4"
        
    Returns:
        Validated GuitarSample instance
        
    Raises:
        ValidationError: If any field fails validation
    """
    return GuitarSample(
        id=id,
        prompt=prompt,
        chords=chords,
        strum_pattern=strum_pattern,
        tempo=tempo,
        time_signature=time_signature,
        genre=genre,
        emotion=emotion,
        key=key,
        mode=mode
    )


# =============================================================================
# EXAMPLE USAGE (for testing)
# =============================================================================

if __name__ == "__main__":
    # This code runs when you execute: python schema.py
    
    print("=" * 60)
    print("Testing GuitarSample Schema")
    print("=" * 60)
    
    # Test 1: Create a valid sample
    print("\n✓ Test 1: Creating a valid sample...")
    try:
        sample = GuitarSample(
            id="test_001",
            prompt="upbeat folk progression in G major with moderate tempo",
            chords=["G", "D", "Em", "C"],
            strum_pattern="D_DU_UD_",
            tempo=110,
            genre="folk",
            emotion="upbeat",
            key="G",
            mode="major"
        )
        print(f"  ✅ Success! Created: {sample.id}")
        print(f"  Chords: {sample.chords}")
        print(f"  Pattern: {sample.strum_pattern}")
        print(f"  Tempo: {sample.tempo} BPM")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    # Test 2: Invalid strum pattern
    print("\n✓ Test 2: Testing invalid strum pattern...")
    try:
        bad_sample = GuitarSample(
            id="test_002",
            prompt="test prompt that is long enough to pass validation",
            chords=["C", "G"],
            strum_pattern="XYZ",  # Invalid!
            tempo=100,
            genre="pop",
            emotion="upbeat",
            key="C",
            mode="major"
        )
        print(f"  ❌ Should have failed but didn't!")
    except Exception as e:
        print(f"  ✅ Correctly rejected: {type(e).__name__}")
        print(f"     Message: {str(e)[:80]}...")
    
    # Test 3: Invalid tempo
    print("\n✓ Test 3: Testing invalid tempo (too high)...")
    try:
        bad_sample = GuitarSample(
            id="test_003",
            prompt="test prompt that is long enough to pass validation",
            chords=["C", "G"],
            strum_pattern="D_DU_UD_",
            tempo=300,  # Too high!
            genre="pop",
            emotion="upbeat",
            key="C",
            mode="major"
        )
        print(f"  ❌ Should have failed but didn't!")
    except Exception as e:
        print(f"  ✅ Correctly rejected: {type(e).__name__}")
    
    # Test 4: Invalid genre
    print("\n✓ Test 4: Testing invalid genre...")
    try:
        bad_sample = GuitarSample(
            id="test_004",
            prompt="test prompt that is long enough to pass validation",
            chords=["C", "G"],
            strum_pattern="D_DU_UD_",
            tempo=100,
            genre="heavy_metal",  # Not in our list!
            emotion="upbeat",
            key="C",
            mode="major"
        )
        print(f"  ❌ Should have failed but didn't!")
    except Exception as e:
        print(f"  ✅ Correctly rejected: {type(e).__name__}")
    
    # Test 5: Case insensitivity
    print("\n✓ Test 5: Testing case insensitivity (FOLK → folk)...")
    try:
        sample = GuitarSample(
            id="test_005",
            prompt="test prompt that is long enough to pass validation",
            chords=["C", "G"],
            strum_pattern="D_DU_UD_",
            tempo=100,
            genre="FOLK",  # Uppercase
            emotion="UPBEAT",  # Uppercase
            key="C",
            mode="MAJOR"  # Uppercase
        )
        print(f"  ✅ Success! Genre stored as: '{sample.genre}'")
        print(f"     Emotion stored as: '{sample.emotion}'")
        print(f"     Mode stored as: '{sample.mode}'")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    # Test 6: Invalid chord
    print("\n✓ Test 6: Testing invalid chord symbol...")
    try:
        bad_sample = GuitarSample(
            id="test_006",
            prompt="test prompt that is long enough to pass validation",
            chords=["C", "Xyz", "G"],  # Xyz is invalid!
            strum_pattern="D_DU_UD_",
            tempo=100,
            genre="pop",
            emotion="upbeat",
            key="C",
            mode="major"
        )
        print(f"  ❌ Should have failed but didn't!")
    except Exception as e:
        print(f"  ✅ Correctly rejected: {type(e).__name__}")
    
    # Test 7: Export to JSON
    print("\n✓ Test 7: Exporting to JSON...")
    sample = GuitarSample(
        id="export_test",
        prompt="melancholic ballad in Am with slow gentle rhythm",
        chords=["Am", "F", "C", "G"],
        strum_pattern="D___D_D_",
        tempo=70,
        genre="ballad",
        emotion="melancholic",
        key="Am",
        mode="minor"
    )
    json_str = sample.model_dump_json(indent=2)
    print(f"  JSON output:")
    print(f"  {json_str[:200]}...")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)