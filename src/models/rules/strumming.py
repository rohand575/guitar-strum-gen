"""
Strumming Module - Pattern Templates for Guitar Strumming

This module provides strumming pattern selection based on:
    - Genre (folk, rock, pop, ballad, etc.)
    - Tempo (slow, moderate, fast)
    - Emotion (upbeat, melancholic, energetic, etc.)

Pattern Format:
    - 8 characters representing one measure of 4/4 time
    - D = Downstroke, U = Upstroke, _ = Rest

Author: Rohan Rajendra Dhanawade
"""

from typing import List, Dict, Optional, Tuple
import random
import re


# =============================================================================
# CONSTANTS
# =============================================================================

TEMPO_RANGES = {
    "slow": (40, 80),
    "moderate": (81, 120),
    "fast": (121, 200),
}

STRUM_PATTERN_REGEX = re.compile(r'^[DU_]{8}$')


# =============================================================================
# PATTERN TEMPLATES BY GENRE AND TEMPO
# =============================================================================

GENRE_PATTERNS: Dict[str, Dict[str, List[str]]] = {
    "pop": {
        "slow": ["D___D___", "D___D_D_", "D__D__D_"],
        "moderate": ["D_DU_UD_", "D_DUDU__", "D_D_DUDU", "D_DU__DU"],
        "fast": ["DDUUDDUU", "D_DUD_DU", "DDUU_DUU"],
    },
    "rock": {
        "slow": ["D___D___", "D_D___D_", "D___D_DU"],
        "moderate": ["D_D_D_D_", "D_D_D_DU", "DD__DD__", "D_DUD_DU"],
        "fast": ["DDDDDDDD", "DD_DDD_D", "D_D_DDDD", "DDDD__DD"],
    },
    "folk": {
        "slow": ["D___D_D_", "D__D_D__", "D___DU__"],
        "moderate": ["D_DU_DU_", "D_D_DUDU", "D_DUDU__", "DUDU____"],
        "fast": ["DDUUDDUU", "D_DUDUDU", "DUDU_DUD"],
    },
    "ballad": {
        "slow": ["D_______", "D___D___", "D___D_D_", "D__D__D_"],
        "moderate": ["D___D_DU", "D_D___DU", "D_DU____", "D___DUDU"],
        "fast": ["D_DU_DU_", "D_D_D_DU"],
    },
    "country": {
        "slow": ["D___D_D_", "D__D__D_", "D___D_DU"],
        "moderate": ["D_DU_DU_", "D_D_DUDU", "D_DUD_D_", "DD__D_DU"],
        "fast": ["DDUUDDUU", "D_DUDUDU", "DD_DD_DU"],
    },
    "blues": {
        "slow": ["D___D___", "D___D_D_", "D__D__D_"],
        "moderate": ["D_D_D_D_", "D_D_D_DU", "D_DU_D__"],
        "fast": ["D_D_D_DU", "DDUU__DU", "D_DUD_DU"],
    },
    "jazz": {
        "slow": ["D_______", "D___D___", "D__D____"],
        "moderate": ["D___D_D_", "D_D___D_", "_D__D___", "D___D__U"],
        "fast": ["D_D_D_D_", "_D_D_D_D", "D__DD__D"],
    },
    "indie": {
        "slow": ["D_______", "D___D___", "____D___"],
        "moderate": ["D_DU_UD_", "D__DU_D_", "_D_D_DU_", "D_D___DU"],
        "fast": ["DDUUDDUU", "_DUDUD_D", "DU_DU_DU"],
    },
    "acoustic": {
        "slow": ["D___D___", "D___D_D_", "D__D__D_"],
        "moderate": ["D_DU_DU_", "D_D_DUDU", "D_DUDU__", "D_DU__DU"],
        "fast": ["DDUUDDUU", "D_DUDUDU", "DUDU_DUU"],
    },
}


# =============================================================================
# EMOTION MODIFIERS
# =============================================================================

EMOTION_PREFERENCES = {
    "upbeat": {"prefer_dense": True, "prefer_downbeats": False, "min_strokes": 4},
    "melancholic": {"prefer_dense": False, "prefer_downbeats": True, "max_strokes": 5},
    "mellow": {"prefer_dense": False, "prefer_downbeats": True, "max_strokes": 5},
    "energetic": {"prefer_dense": True, "prefer_downbeats": False, "min_strokes": 5},
    "peaceful": {"prefer_dense": False, "prefer_downbeats": True, "max_strokes": 4},
    "dramatic": {"prefer_dense": False, "prefer_downbeats": True, "max_strokes": 5},
    "hopeful": {"prefer_dense": False, "prefer_downbeats": False, "min_strokes": 3},
    "nostalgic": {"prefer_dense": False, "prefer_downbeats": True, "max_strokes": 5},
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def count_strokes(pattern: str) -> int:
    """Count the number of strokes (D or U) in a pattern."""
    return sum(1 for c in pattern if c in "DU")


def count_downstrokes(pattern: str) -> int:
    """Count downstrokes (D) in a pattern."""
    return pattern.count("D")


def count_upstrokes(pattern: str) -> int:
    """Count upstrokes (U) in a pattern."""
    return pattern.count("U")


def has_downbeat_emphasis(pattern: str) -> bool:
    """Check if pattern emphasizes downbeats (positions 0, 2, 4, 6)."""
    downbeat_positions = [0, 2, 4, 6]
    downbeat_strokes = sum(1 for i in downbeat_positions if pattern[i] in "DU")
    upbeat_strokes = sum(1 for i in [1, 3, 5, 7] if pattern[i] in "DU")
    return downbeat_strokes >= upbeat_strokes


def get_tempo_category(tempo: int) -> str:
    """Categorize a tempo (BPM) as slow, moderate, or fast."""
    for category, (low, high) in TEMPO_RANGES.items():
        if low <= tempo <= high:
            return category
    return "moderate"


def validate_pattern(pattern: str) -> bool:
    """Check if a pattern is valid (8 chars of D, U, _)."""
    return bool(STRUM_PATTERN_REGEX.match(pattern))


# =============================================================================
# MAIN SELECTION FUNCTION
# =============================================================================

### 3. The Selection Flow
"""
select_strumming_pattern("folk", 110, "melancholic")
         │
         ▼
    genre = "folk"
    tempo_category = "moderate"  (from 110 BPM)
         │
         ▼
    candidates = ["D_DU_DU_", "D_D_DUDU", "D_DUDU__", ...]
         │
         ▼
    Apply emotion filter (max 5 strokes, prefer downbeats)
         │
         ▼
    filtered = ["D_DU_DU_", ...]
         │
         ▼
    Pick one randomly → "D_DU_DU_"
"""


def select_strumming_pattern(
    genre: str = "pop",
    tempo: int = 100,
    emotion: Optional[str] = None
) -> str:
    """Select a strumming pattern based on genre, tempo, and emotion."""
    genre = genre.lower()
    
    if genre not in GENRE_PATTERNS:
        genre = "pop"
    
    tempo_category = get_tempo_category(tempo)
    candidates = GENRE_PATTERNS[genre].get(tempo_category, [])
    
    if not candidates:
        candidates = GENRE_PATTERNS[genre].get("moderate", ["D_DU_UD_"])
    
    if emotion and emotion.lower() in EMOTION_PREFERENCES:
        prefs = EMOTION_PREFERENCES[emotion.lower()]
        filtered = []
        
        for pattern in candidates:
            stroke_count = count_strokes(pattern)
            
            if "min_strokes" in prefs and stroke_count < prefs["min_strokes"]:
                continue
            if "max_strokes" in prefs and stroke_count > prefs["max_strokes"]:
                continue
            if prefs.get("prefer_downbeats") and not has_downbeat_emphasis(pattern):
                continue
            
            filtered.append(pattern)
        
        if filtered:
            candidates = filtered
    
    return random.choice(candidates)


def get_all_patterns_for_genre(genre: str) -> Dict[str, List[str]]:
    """Get all patterns for a genre, organized by tempo."""
    genre = genre.lower()
    if genre not in GENRE_PATTERNS:
        return {}
    return GENRE_PATTERNS[genre]


def suggest_tempo_for_emotion(emotion: str) -> Tuple[int, int]:
    """Suggest a tempo range based on emotion."""
    emotion = emotion.lower()
    
    emotion_tempos = {
        "upbeat": (100, 140),
        "melancholic": (60, 85),
        "mellow": (70, 95),
        "energetic": (120, 160),
        "peaceful": (60, 90),
        "dramatic": (70, 110),
        "hopeful": (90, 120),
        "nostalgic": (75, 100),
    }
    
    return emotion_tempos.get(emotion, (80, 120))


def pick_tempo(
    emotion: Optional[str] = None,
    tempo_hint: Optional[str] = None
) -> int:
    """Pick a specific tempo value based on hints."""
    if emotion:
        min_tempo, max_tempo = suggest_tempo_for_emotion(emotion)
    else:
        min_tempo, max_tempo = 80, 120
    
    if tempo_hint:
        tempo_hint = tempo_hint.lower()
        if tempo_hint in TEMPO_RANGES:
            min_tempo, max_tempo = TEMPO_RANGES[tempo_hint]
        elif "slow" in tempo_hint:
            min_tempo, max_tempo = TEMPO_RANGES["slow"]
        elif "fast" in tempo_hint or "quick" in tempo_hint:
            min_tempo, max_tempo = TEMPO_RANGES["fast"]
    
    return random.randint(min_tempo, max_tempo)


# =============================================================================
# PATTERN ANALYSIS
# =============================================================================

def analyze_pattern(pattern: str) -> Dict:
    """Analyze a strumming pattern and return its characteristics."""
    return {
        "pattern": pattern,
        "total_strokes": count_strokes(pattern),
        "downstrokes": count_downstrokes(pattern),
        "upstrokes": count_upstrokes(pattern),
        "rests": pattern.count("_"),
        "density": count_strokes(pattern) / 8.0,
        "downbeat_emphasis": has_downbeat_emphasis(pattern),
        "starts_with_down": pattern[0] == "D",
        "valid": validate_pattern(pattern),
    }


def pattern_to_beats(pattern: str) -> str:
    """Convert a pattern to a readable beat representation."""
    beats = "Beat:    1   &   2   &   3   &   4   &"
    strums = "Strum:   " + "   ".join(pattern)
    return f"{beats}\n{strums}"


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing strumming.py")
    print("=" * 60)
    
    print("\n✓ Test 1: Tempo categorization")
    print(f"  70 BPM → {get_tempo_category(70)}")
    print(f"  110 BPM → {get_tempo_category(110)}")
    print(f"  150 BPM → {get_tempo_category(150)}")
    
    print("\n✓ Test 2: Stroke counting")
    test_pattern = "D_DU_UD_"
    print(f"  Pattern: {test_pattern}")
    print(f"  Total strokes: {count_strokes(test_pattern)}")
    
    print("\n✓ Test 3: Pattern selection by genre")
    for genre in ["pop", "rock", "folk", "ballad"]:
        pattern = select_strumming_pattern(genre, tempo=110)
        print(f"  {genre:8} → {pattern}")
    
    print("\n✓ Test 4: Beat visualization")
    print(pattern_to_beats("D_DU_UD_"))
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)