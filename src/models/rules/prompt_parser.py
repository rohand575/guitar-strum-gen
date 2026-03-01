"""
Prompt Parser Module - Extract Musical Features from Natural Language

This module parses natural language prompts to extract musical parameters:
    - Key and mode (e.g., "in G major" → key="G", mode="major")
    - Genre (e.g., "folk song" → genre="folk")
    - Emotion (e.g., "melancholic" → emotion="melancholic")
    - Tempo (e.g., "slow tempo" → tempo=70, or "120 BPM" → tempo=120)

Author: Rohan Rajendra Dhanawade
"""

from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import re


# =============================================================================
# PARSED FEATURES DATA CLASS
# =============================================================================

@dataclass
class ParsedFeatures:
    """Container for features extracted from a prompt."""
    key: Optional[str] = None
    mode: Optional[str] = None
    genre: Optional[str] = None
    emotion: Optional[str] = None
    tempo: Optional[int] = None
    tempo_hint: Optional[str] = None
    raw_prompt: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "key": self.key,
            "mode": self.mode,
            "genre": self.genre,
            "emotion": self.emotion,
            "tempo": self.tempo,
            "tempo_hint": self.tempo_hint,
        }
    
    def __str__(self) -> str:
        parts = []
        if self.key:
            mode_str = self.mode or "major"
            parts.append(f"Key: {self.key} {mode_str}")
        if self.genre:
            parts.append(f"Genre: {self.genre}")
        if self.emotion:
            parts.append(f"Emotion: {self.emotion}")
        if self.tempo:
            parts.append(f"Tempo: {self.tempo} BPM")
        elif self.tempo_hint:
            parts.append(f"Tempo: {self.tempo_hint}")
        return " | ".join(parts) if parts else "(no features extracted)"


# =============================================================================
# KEYWORD DICTIONARIES
# =============================================================================

GENRE_KEYWORDS = {
    "pop": ["pop", "popular", "mainstream", "radio"],
    "rock": ["rock", "rocky", "hard rock", "classic rock"],
    "folk": ["folk", "folksy", "traditional", "acoustic folk"],
    "ballad": ["ballad", "slow song", "love song", "power ballad"],
    "country": ["country", "western", "nashville", "twangy"],
    "blues": ["blues", "bluesy", "12-bar", "twelve bar"],
    "jazz": ["jazz", "jazzy", "swing", "bebop"],
    "indie": ["indie", "independent", "alternative", "alt"],
    "acoustic": ["acoustic", "unplugged", "singer-songwriter"],
}

EMOTION_KEYWORDS = {
    "upbeat": ["upbeat", "happy", "cheerful", "joyful", "bright", "fun", "lively"],
    "melancholic": ["melancholic", "melancholy", "sad", "sorrowful", "tearful", "heartbroken"],
    "mellow": ["mellow", "chill", "relaxed", "laid-back", "easy", "smooth"],
    "energetic": ["energetic", "energized", "driving", "powerful", "intense", "high-energy"],
    "peaceful": ["peaceful", "calm", "serene", "tranquil", "gentle", "soft"],
    "dramatic": ["dramatic", "epic", "cinematic", "theatrical", "grandiose"],
    "hopeful": ["hopeful", "optimistic", "uplifting", "inspiring", "positive"],
    "nostalgic": ["nostalgic", "wistful", "reminiscent", "bittersweet", "longing"],
}

TEMPO_KEYWORDS = {
    "slow": ["slow", "slowly", "gentle", "leisurely", "relaxed", "ballad-tempo", "adagio"],
    "moderate": ["moderate", "medium", "mid-tempo", "steady", "andante", "moderato"],
    "fast": ["fast", "quick", "upbeat", "energetic", "driving", "allegro", "presto", "uptempo"],
}

KEY_PATTERNS = [
    r'\bin\s+([A-Ga-g][#b]?)\s*(major|minor|m)?\b',
    r'\bkey\s+of\s+([A-Ga-g][#b]?)\s*(major|minor|m)?\b',
    r'\b([A-Ga-g][#b]?)\s*(major|minor)\b',
    r'\b([A-Ga-g][#b]?)m\b(?!\s*[-–→])',
]


# =============================================================================
# PARSING FUNCTIONS
# =============================================================================

def extract_key_and_mode(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract musical key and mode from a prompt."""
    for pattern in KEY_PATTERNS:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            key = match.group(1).upper()
            
            if len(key) > 1:
                key = key[0].upper() + key[1].lower()
            
            mode_match = match.group(2) if len(match.groups()) > 1 else None
            
            if mode_match:
                mode_match = mode_match.lower()
                if mode_match in ["minor", "m"]:
                    mode = "minor"
                else:
                    mode = "major"
            else:
                full_match = match.group(0)
                if full_match.endswith('m') and not full_match.endswith('major'):
                    mode = "minor"
                else:
                    mode = "major"
            
            return (key, mode)
    
    return (None, None)


def extract_genre(prompt: str) -> Optional[str]:
    """Extract genre from a prompt using keyword matching."""
    prompt_lower = prompt.lower()
    
    for genre, keywords in GENRE_KEYWORDS.items():
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, prompt_lower):
                return genre
    
    return None


def extract_emotion(prompt: str) -> Optional[str]:
    """Extract emotion from a prompt using keyword matching."""
    prompt_lower = prompt.lower()
    
    for emotion, keywords in EMOTION_KEYWORDS.items():
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, prompt_lower):
                return emotion
    
    return None


def extract_tempo(prompt: str) -> Tuple[Optional[int], Optional[str]]:
    """Extract tempo from a prompt."""
    prompt_lower = prompt.lower()
    
    bpm_patterns = [
        r'(\d{2,3})\s*bpm\b',
        r'(\d{2,3})\s*beats?\s*per\s*min',
        r'\btempo\s*(?:of\s*)?(\d{2,3})\b',
        r'\bat\s+(\d{2,3})\s*bpm\b',
    ]
    
    for pattern in bpm_patterns:
        match = re.search(pattern, prompt_lower)
        if match:
            bpm = int(match.group(1))
            if 40 <= bpm <= 200:
                return (bpm, None)
    
    for tempo_category, keywords in TEMPO_KEYWORDS.items():
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, prompt_lower):
                return (None, tempo_category)
    
    return (None, None)


# =============================================================================
# MAIN PARSING FUNCTION
# =============================================================================

def parse_prompt(prompt: str) -> ParsedFeatures:
    """Parse a natural language prompt and extract all musical features."""
    features = ParsedFeatures(raw_prompt=prompt)
    
    features.key, features.mode = extract_key_and_mode(prompt)
    features.genre = extract_genre(prompt)
    features.emotion = extract_emotion(prompt)
    features.tempo, features.tempo_hint = extract_tempo(prompt)
    
    return features


# =============================================================================
# DEFAULTS AND COMPLETION
# =============================================================================

def apply_defaults(features: ParsedFeatures) -> ParsedFeatures:
    """Fill in default values for any missing features."""
    if features.key is None:
        features.key = "C"
    if features.mode is None:
        features.mode = "major"
    if features.genre is None:
        features.genre = "pop"
    if features.emotion is None:
        features.emotion = "mellow"
    if features.tempo is None:
        from src.rules.strumming import pick_tempo
        features.tempo = pick_tempo(
            emotion=features.emotion,
            tempo_hint=features.tempo_hint
        )
    
    return features


def parse_and_complete(prompt: str) -> ParsedFeatures:
    """Parse a prompt and apply defaults for any missing values."""
    features = parse_prompt(prompt)
    features = apply_defaults(features)
    return features


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing prompt_parser.py")
    print("=" * 60)
    
    test_prompts = [
        "upbeat folk song in G major at 110 BPM",
        "melancholic ballad in Am",
        "Give me a slow, sad rock progression",
        "energetic pop song, fast tempo",
    ]
    
    print("\n✓ Testing prompt parsing:")
    for prompt in test_prompts:
        features = parse_prompt(prompt)
        print(f"\n  Prompt: \"{prompt}\"")
        print(f"  Result: {features}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)