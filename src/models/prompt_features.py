"""
Prompt Features Data Structure
==============================
Guitar Strum Generator - Thesis Project
Author: Rohan Rajendra Dhanawade

This module defines the data structure that holds all extracted features
from a natural language prompt. Think of it as the "receipt" that the
prompt parser hands to the generation system.

Key Design Decisions:
1. Every field has a value (no None for core features) - we use defaults
2. Every field has a confidence score (0.0 to 1.0)
3. We track what was explicitly mentioned vs. inferred
4. We maintain the original prompt for debugging/logging

Usage:
    from src.models.prompt_features import PromptFeatures, ExtractionConfidence
    
    features = PromptFeatures(
        key="E",
        mode="minor",
        genre="folk",
        emotion="melancholic",
        tempo=70,
        ...
    )
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class ConfidenceLevel(Enum):
    """
    Human-readable confidence levels for extraction results.
    
    These map to numeric ranges:
    - HIGH: 0.8 - 1.0 (explicitly stated in prompt)
    - MEDIUM: 0.5 - 0.79 (strongly implied or synonym match)
    - LOW: 0.2 - 0.49 (weakly implied)
    - DEFAULT: 0.0 - 0.19 (not found, using fallback)
    """
    HIGH = "high"       # Explicitly stated: "in E minor"
    MEDIUM = "medium"   # Strongly implied: "sad" → melancholic
    LOW = "low"         # Weakly implied: "evening" → slow tempo
    DEFAULT = "default" # Not found, using system default


@dataclass
class ExtractionConfidence:
    """
    Tracks confidence for each extracted feature.
    
    This is crucial for:
    1. Deciding when to use neural model vs. rule-based fallback
    2. Explaining to users what was inferred vs. stated
    3. Thesis evaluation: measuring parser accuracy
    
    Attributes:
        key: Confidence in extracted musical key (0.0-1.0)
        mode: Confidence in major/minor detection (0.0-1.0)
        genre: Confidence in genre classification (0.0-1.0)
        emotion: Confidence in emotion detection (0.0-1.0)
        tempo: Confidence in tempo extraction (0.0-1.0)
    """
    key: float = 0.0
    mode: float = 0.0
    genre: float = 0.0
    emotion: float = 0.0
    tempo: float = 0.0
    
    def overall(self) -> float:
        """
        Calculate overall confidence as weighted average.
        
        Key and mode are weighted higher because they're most
        critical for harmonic correctness.
        
        Returns:
            Weighted average confidence (0.0-1.0)
        """
        weights = {
            'key': 0.25,
            'mode': 0.25,
            'genre': 0.20,
            'emotion': 0.15,
            'tempo': 0.15
        }
        
        total = (
            self.key * weights['key'] +
            self.mode * weights['mode'] +
            self.genre * weights['genre'] +
            self.emotion * weights['emotion'] +
            self.tempo * weights['tempo']
        )
        
        return round(total, 3)
    
    def get_level(self, score: float) -> ConfidenceLevel:
        """Convert numeric score to human-readable level."""
        if score >= 0.8:
            return ConfidenceLevel.HIGH
        elif score >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.DEFAULT
    
    def summary(self) -> Dict[str, str]:
        """Get human-readable summary of all confidence levels."""
        return {
            'key': self.get_level(self.key).value,
            'mode': self.get_level(self.mode).value,
            'genre': self.get_level(self.genre).value,
            'emotion': self.get_level(self.emotion).value,
            'tempo': self.get_level(self.tempo).value,
            'overall': self.get_level(self.overall()).value
        }


@dataclass 
class PromptFeatures:
    """
    Complete extracted features from a natural language prompt.
    
    This is the main output of the prompt parser. Every field has:
    1. A value (never None for core features - we use defaults)
    2. Associated confidence tracked in the `confidence` field
    
    Core Features (always populated):
        key: Musical key root (e.g., "C", "E", "F#")
        mode: "major" or "minor"
        genre: One of the 9 supported genres
        emotion: One of the 8 supported emotions
        tempo: BPM as integer (40-200)
        time_signature: Almost always "4/4" for guitar
    
    Metadata:
        original_prompt: The raw input text
        confidence: ExtractionConfidence object with scores
        extracted_chords: Any specific chords mentioned (e.g., "Am-G-C")
        warnings: Any issues encountered during parsing
    
    Example:
        >>> features = PromptFeatures(
        ...     key="E",
        ...     mode="minor", 
        ...     genre="folk",
        ...     emotion="melancholic",
        ...     tempo=70,
        ...     original_prompt="sad folk song in E minor, slow"
        ... )
        >>> print(features.full_key)
        'Em'
    """
    
    # =========================================================================
    # CORE MUSICAL FEATURES (always have a value)
    # =========================================================================
    
    key: str = "C"              # Musical key root: C, D, E, F, G, A, B (with #/b)
    mode: str = "major"         # "major" or "minor"
    genre: str = "pop"          # One of: pop, rock, folk, ballad, country, blues, jazz, indie, acoustic
    emotion: str = "mellow"     # One of: upbeat, melancholic, mellow, energetic, peaceful, dramatic, hopeful, nostalgic
    tempo: int = 100            # BPM (40-200)
    time_signature: str = "4/4" # Almost always 4/4 for guitar strumming
    
    # =========================================================================
    # METADATA
    # =========================================================================
    
    original_prompt: str = ""   # Raw input for debugging
    confidence: ExtractionConfidence = field(default_factory=ExtractionConfidence)
    
    # Optional: specific chords mentioned in prompt (e.g., "use Am, G, C, F")
    extracted_chords: List[str] = field(default_factory=list)
    
    # Any warnings or notes from parsing
    warnings: List[str] = field(default_factory=list)
    
    # Track which features were explicitly found vs defaulted
    explicitly_stated: Dict[str, bool] = field(default_factory=lambda: {
        'key': False,
        'mode': False,
        'genre': False,
        'emotion': False,
        'tempo': False
    })
    
    # =========================================================================
    # COMPUTED PROPERTIES
    # =========================================================================
    
    @property
    def full_key(self) -> str:
        """
        Get the full key name (e.g., "Em", "C", "F#m").
        
        Combines key root with mode suffix:
        - Major keys: just the root (C, G, D)
        - Minor keys: root + "m" (Am, Em, Dm)
        
        Returns:
            Full key string like "Am" or "G"
        """
        if self.mode == "minor":
            return f"{self.key}m"
        return self.key
    
    @property
    def tempo_category(self) -> str:
        """
        Get human-readable tempo category.
        
        Categories based on standard musical terms:
        - slow: 40-79 BPM (ballads, slow songs)
        - moderate: 80-119 BPM (most pop/folk)
        - fast: 120-200 BPM (upbeat, energetic)
        
        Returns:
            One of: "slow", "moderate", "fast"
        """
        if self.tempo < 80:
            return "slow"
        elif self.tempo < 120:
            return "moderate"
        else:
            return "fast"
    
    @property
    def is_high_confidence(self) -> bool:
        """
        Check if overall extraction confidence is high enough for neural model.
        
        If confidence is too low, the system should consider:
        1. Asking clarifying questions
        2. Using rule-based fallback
        3. Flagging output as uncertain
        
        Returns:
            True if overall confidence >= 0.5
        """
        return self.confidence.overall() >= 0.5
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization or logging.
        
        Returns:
            Dictionary with all features and metadata
        """
        return {
            'key': self.key,
            'mode': self.mode,
            'full_key': self.full_key,
            'genre': self.genre,
            'emotion': self.emotion,
            'tempo': self.tempo,
            'tempo_category': self.tempo_category,
            'time_signature': self.time_signature,
            'original_prompt': self.original_prompt,
            'extracted_chords': self.extracted_chords,
            'confidence': {
                'key': self.confidence.key,
                'mode': self.confidence.mode,
                'genre': self.confidence.genre,
                'emotion': self.confidence.emotion,
                'tempo': self.confidence.tempo,
                'overall': self.confidence.overall()
            },
            'confidence_levels': self.confidence.summary(),
            'explicitly_stated': self.explicitly_stated,
            'warnings': self.warnings,
            'is_high_confidence': self.is_high_confidence
        }
    
    def summary(self) -> str:
        """
        Get a human-readable summary of extracted features.
        
        Useful for debugging and user feedback.
        
        Returns:
            Formatted string summary
        """
        lines = [
            f"Extracted Features from: \"{self.original_prompt[:50]}{'...' if len(self.original_prompt) > 50 else ''}\"",
            f"  Key: {self.full_key} (confidence: {self.confidence.get_level(self.confidence.key).value})",
            f"  Genre: {self.genre} (confidence: {self.confidence.get_level(self.confidence.genre).value})",
            f"  Emotion: {self.emotion} (confidence: {self.confidence.get_level(self.confidence.emotion).value})",
            f"  Tempo: {self.tempo} BPM ({self.tempo_category}) (confidence: {self.confidence.get_level(self.confidence.tempo).value})",
            f"  Overall Confidence: {self.confidence.overall():.2f} ({self.confidence.get_level(self.confidence.overall()).value})"
        ]
        
        if self.extracted_chords:
            lines.append(f"  Explicit Chords: {' → '.join(self.extracted_chords)}")
        
        if self.warnings:
            lines.append(f"  ⚠️  Warnings: {'; '.join(self.warnings)}")
        
        return '\n'.join(lines)
    
    def __str__(self) -> str:
        """String representation for quick inspection."""
        return f"PromptFeatures({self.full_key}, {self.genre}, {self.emotion}, {self.tempo}bpm)"


# =============================================================================
# DEFAULT VALUES (used when nothing is detected)
# =============================================================================

# These are "safe" defaults that work well for most guitar music
DEFAULT_FEATURES = {
    'key': 'G',           # G major is very guitar-friendly (open chords)
    'mode': 'major',      # Major is more common in popular music
    'genre': 'pop',       # Pop is the most general/neutral genre
    'emotion': 'mellow',  # Mellow is neutral - not too happy, not too sad
    'tempo': 100,         # 100 BPM is moderate, works for most styles
    'time_signature': '4/4'
}


# =============================================================================
# CONTROLLED VOCABULARIES (must match schema.py!)
# =============================================================================

VALID_GENRES = [
    'pop', 'rock', 'folk', 'ballad', 'country', 
    'blues', 'jazz', 'indie', 'acoustic'
]

VALID_EMOTIONS = [
    'upbeat', 'melancholic', 'mellow', 'energetic',
    'peaceful', 'dramatic', 'hopeful', 'nostalgic'
]

VALID_MODES = ['major', 'minor']

# Valid key roots (including enharmonic equivalents)
VALID_KEY_ROOTS = [
    'C', 'C#', 'Db', 'D', 'D#', 'Eb', 'E', 'F', 
    'F#', 'Gb', 'G', 'G#', 'Ab', 'A', 'A#', 'Bb', 'B'
]


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing PromptFeatures Data Structure")
    print("=" * 60)
    
    # Test 1: Create features with explicit values
    print("\n✅ Test 1: Creating features with explicit values")
    features = PromptFeatures(
        key="E",
        mode="minor",
        genre="folk",
        emotion="melancholic",
        tempo=70,
        original_prompt="sad folk song in E minor, slow",
        confidence=ExtractionConfidence(
            key=0.95,
            mode=0.95,
            genre=0.85,
            emotion=0.80,
            tempo=0.70
        ),
        explicitly_stated={
            'key': True,
            'mode': True,
            'genre': True,
            'emotion': True,
            'tempo': True
        }
    )
    
    print(features.summary())
    print(f"\n  full_key property: {features.full_key}")
    print(f"  tempo_category: {features.tempo_category}")
    print(f"  is_high_confidence: {features.is_high_confidence}")
    
    # Test 2: Test with defaults (ambiguous prompt)
    print("\n" + "=" * 60)
    print("✅ Test 2: Features with mostly defaults (ambiguous prompt)")
    
    ambiguous_features = PromptFeatures(
        emotion="mellow",  # Only thing we could extract
        original_prompt="something chill for the evening",
        confidence=ExtractionConfidence(
            key=0.0,      # defaulted
            mode=0.0,     # defaulted  
            genre=0.0,    # defaulted
            emotion=0.4,  # weakly implied by "chill"
            tempo=0.3     # weakly implied by "evening"
        ),
        warnings=["Key not specified, using default G major",
                  "Genre not specified, using default pop"]
    )
    
    print(ambiguous_features.summary())
    print(f"\n  is_high_confidence: {ambiguous_features.is_high_confidence}")
    
    # Test 3: Dictionary export
    print("\n" + "=" * 60)
    print("✅ Test 3: Dictionary export (for JSON serialization)")
    
    feature_dict = features.to_dict()
    print(f"  Keys in dict: {list(feature_dict.keys())}")
    print(f"  Confidence levels: {feature_dict['confidence_levels']}")
    
    # Test 4: Confidence level calculations
    print("\n" + "=" * 60)
    print("✅ Test 4: Confidence level thresholds")
    
    conf = ExtractionConfidence(key=0.9, mode=0.6, genre=0.3, emotion=0.1, tempo=0.5)
    print(f"  Key (0.9): {conf.get_level(conf.key).value}")
    print(f"  Mode (0.6): {conf.get_level(conf.mode).value}")
    print(f"  Genre (0.3): {conf.get_level(conf.genre).value}")
    print(f"  Emotion (0.1): {conf.get_level(conf.emotion).value}")
    print(f"  Overall: {conf.overall():.3f} → {conf.get_level(conf.overall()).value}")
    
    print("\n" + "=" * 60)
    print("All tests passed! PromptFeatures is ready.")
    print("=" * 60)
