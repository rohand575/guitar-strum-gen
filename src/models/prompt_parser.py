"""
Prompt Parser - NLP Module
==========================
Guitar Strum Generator - Thesis Project
Author: Rohan Rajendra Dhanawade

This module extracts structured musical features from natural language prompts.
It's the "brain" that interprets what users want when they type things like
"Give me a sad folk song in E minor at a slow pace".

Architecture:
    User Prompt → [Key Extractor] → [Genre Detector] → [Emotion Analyzer] 
                → [Tempo Extractor] → [Chord Extractor] → PromptFeatures

Design Philosophy:
    1. Extract what IS there with high confidence
    2. Track what's MISSING honestly
    3. Fill gaps with PRINCIPLED defaults (not hallucinations)
    4. Provide CONFIDENCE scores for downstream decision-making

Usage:
    from src.models.prompt_parser import PromptParser
    
    parser = PromptParser()
    features = parser.parse("sad folk song in E minor, slow tempo")
    print(features.summary())

This is the RULE-BASED implementation. A neural enhancement (DistilBERT)
can be added later for ambiguous cases.
"""

import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass

from .prompt_features import (
    PromptFeatures, 
    ExtractionConfidence,
    DEFAULT_FEATURES,
    VALID_GENRES,
    VALID_EMOTIONS,
    VALID_KEY_ROOTS
)


# =============================================================================
# KNOWLEDGE BASES (Dictionaries for extraction)
# =============================================================================

# -----------------------------------------------------------------------------
# KEY/MODE PATTERNS
# -----------------------------------------------------------------------------

# Regex pattern for explicit key mentions
# Matches: "in C", "in Am", "in F# minor", "key of Bb", "E minor", etc.
KEY_PATTERNS = [
    # "in X minor/major" or "in Xm"
    r'\b(?:in|key\s+of|key:?)\s+([A-G][#b]?)\s*(m|min|minor|maj|major)?\b',
    # "X minor/major" without "in"
    r'\b([A-G][#b]?)\s+(minor|major|min|maj)\b',
    # Just "Xm" like "Am" or "Em" (common shorthand)
    r'\b([A-G][#b]?)(m)\b(?!\w)',
]

# Mode indicators (when mode is stated separately from key)
MINOR_INDICATORS = {
    'minor', 'min', 'm', 'sad', 'dark', 'melancholic', 'melancholy',
    'somber', 'moody', 'gloomy', 'tragic', 'haunting', 'ominous'
}

MAJOR_INDICATORS = {
    'major', 'maj', 'happy', 'bright', 'uplifting', 'cheerful',
    'joyful', 'sunny', 'optimistic', 'triumphant', 'celebratory'
}

# -----------------------------------------------------------------------------
# GENRE MAPPINGS (keyword → genre with synonyms)
# -----------------------------------------------------------------------------

GENRE_KEYWORDS: Dict[str, Set[str]] = {
    'pop': {
        'pop', 'popular', 'mainstream', 'radio', 'catchy', 'commercial',
        'top 40', 'chart', 'contemporary'
    },
    'rock': {
        'rock', 'rocky', 'hard rock', 'classic rock', 'alternative',
        'grunge', 'punk', 'metal', 'heavy', 'electric', 'distorted',
        'powerful', 'driving'
    },
    'folk': {
        'folk', 'folksy', 'acoustic folk', 'traditional', 'americana',
        'woody', 'earthy', 'campfire', 'singer-songwriter', 'fingerpick',
        'fingerstyle', 'storytelling'
    },
    'ballad': {
        'ballad', 'slow song', 'love song', 'romantic', 'slow dance',
        'emotional ballad', 'power ballad', 'heartfelt', 'tender',
        'sentimental', 'intimate'
    },
    'country': {
        'country', 'country western', 'nashville', 'twang', 'cowboy',
        'southern', 'bluegrass', 'honky tonk', 'outlaw', 'western'
    },
    'blues': {
        'blues', 'bluesy', 'blue', '12 bar', 'twelve bar', 'shuffle',
        'delta', 'chicago blues', 'soul', 'soulful', 'groovy', 'funky'
    },
    'jazz': {
        'jazz', 'jazzy', 'swing', 'bebop', 'smooth', 'cool jazz',
        'fusion', 'improvised', 'sophisticated', 'complex', 'chromatic'
    },
    'indie': {
        'indie', 'independent', 'alternative', 'lo-fi', 'lofi',
        'bedroom', 'dreamy', 'shoegaze', 'ethereal', 'atmospheric',
        'ambient', 'chill', 'chillout'
    },
    'acoustic': {
        'acoustic', 'unplugged', 'stripped', 'bare', 'simple',
        'organic', 'natural', 'raw', 'intimate', 'coffeehouse',
        'cafe', 'gentle'
    }
}

# -----------------------------------------------------------------------------
# EMOTION MAPPINGS (keyword → emotion with synonyms)
# -----------------------------------------------------------------------------

EMOTION_KEYWORDS: Dict[str, Set[str]] = {
    'upbeat': {
        'upbeat', 'happy', 'joyful', 'cheerful', 'fun', 'playful',
        'bouncy', 'lively', 'bright', 'sunny', 'positive', 'feel good',
        'feel-good', 'uplifting', 'celebratory', 'party', 'dancing'
    },
    'melancholic': {
        'melancholic', 'melancholy', 'sad', 'sorrowful', 'mournful',
        'tearful', 'heartbroken', 'lonely', 'lonesome', 'depressed',
        'somber', 'gloomy', 'dark', 'tragic', 'grief', 'loss',
        'bittersweet', 'wistful', 'regretful'
    },
    'mellow': {
        'mellow', 'chill', 'chilled', 'relaxed', 'laid back', 'laid-back',
        'easy', 'easygoing', 'smooth', 'soft', 'gentle', 'calm',
        'soothing', 'comfortable', 'cozy', 'warm', 'evening', 'night',
        'late night', 'lounge'
    },
    'energetic': {
        'energetic', 'energy', 'powerful', 'intense', 'driving',
        'fast', 'quick', 'rapid', 'aggressive', 'fierce', 'wild',
        'explosive', 'dynamic', 'exciting', 'thrilling', 'pumped',
        'workout', 'gym', 'running'
    },
    'peaceful': {
        'peaceful', 'peace', 'tranquil', 'serene', 'calm', 'quiet',
        'still', 'meditative', 'zen', 'spiritual', 'reflective',
        'contemplative', 'ambient', 'floating', 'dreamy', 'sleep',
        'relaxation', 'yoga', 'morning'
    },
    'dramatic': {
        'dramatic', 'drama', 'epic', 'cinematic', 'theatrical',
        'intense', 'powerful', 'building', 'climactic', 'tension',
        'suspense', 'emotional', 'moving', 'stirring', 'grand',
        'majestic', 'triumphant'
    },
    'hopeful': {
        'hopeful', 'hope', 'optimistic', 'inspiring', 'inspirational',
        'uplifting', 'encouraging', 'motivational', 'aspirational',
        'determined', 'resilient', 'perseverance', 'overcoming',
        'new beginning', 'sunrise', 'dawn'
    },
    'nostalgic': {
        'nostalgic', 'nostalgia', 'reminiscent', 'throwback', 'retro',
        'vintage', 'old school', 'classic', 'memories', 'remembering',
        'looking back', 'childhood', 'hometown', 'past', 'timeless',
        'sentimental'
    }
}

# -----------------------------------------------------------------------------
# TEMPO MAPPINGS (keyword → BPM range)
# -----------------------------------------------------------------------------

TEMPO_KEYWORDS: Dict[str, Tuple[int, int]] = {
    # Slow (40-79 BPM)
    'very slow': (40, 55),
    'extremely slow': (40, 50),
    'super slow': (40, 50),
    'slow': (55, 75),
    'slowly': (55, 75),
    'gentle': (55, 70),
    'soft': (55, 70),
    'relaxed': (60, 80),
    'laid back': (60, 80),
    'lazy': (50, 70),
    'ballad': (55, 75),
    'dreamy': (55, 75),
    'peaceful': (50, 70),
    'meditation': (45, 60),
    'sleep': (45, 60),
    
    # Moderate (80-119 BPM)
    'moderate': (90, 110),
    'medium': (85, 115),
    'mid tempo': (90, 110),
    'mid-tempo': (90, 110),
    'walking': (95, 115),
    'comfortable': (85, 105),
    'steady': (90, 110),
    'groove': (90, 115),
    'groovy': (95, 115),
    
    # Fast (120-200 BPM)
    'fast': (125, 150),
    'quick': (120, 145),
    'upbeat': (115, 140),
    'energetic': (120, 150),
    'lively': (115, 140),
    'driving': (125, 155),
    'uptempo': (120, 145),
    'up-tempo': (120, 145),
    'dance': (115, 135),
    'dancing': (115, 135),
    'party': (120, 140),
    
    # Very fast (150+ BPM)
    'very fast': (150, 180),
    'super fast': (160, 200),
    'blazing': (170, 200),
    'rapid': (150, 175),
    'punk': (160, 200),
    'thrash': (170, 200),
}

# -----------------------------------------------------------------------------
# CHORD PATTERN (for extracting explicitly mentioned chords)
# -----------------------------------------------------------------------------

# Matches chord symbols like: Am, C, F#m7, Bb, Dmaj7, G/B, etc.
CHORD_PATTERN = r'\b([A-G][#b]?(?:m|min|maj|dim|aug|sus[24]?|add[249]?|[245679]|maj7|min7|7)?(?:/[A-G][#b]?)?)\b'


# =============================================================================
# MAIN PARSER CLASS
# =============================================================================

class PromptParser:
    """
    Rule-based parser that extracts musical features from natural language.
    
    This parser uses:
    - Regex patterns for key/mode extraction
    - Keyword dictionaries with synonym expansion for genre/emotion
    - Tempo keyword → BPM range mapping
    - Confidence scoring based on match quality
    
    Attributes:
        verbose: If True, print debug information during parsing
    
    Example:
        >>> parser = PromptParser()
        >>> features = parser.parse("sad folk song in E minor, slow")
        >>> print(features.full_key)
        'Em'
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the parser.
        
        Args:
            verbose: Print debug information if True
        """
        self.verbose = verbose
        
        # Compile regex patterns for efficiency
        self._key_patterns = [re.compile(p, re.IGNORECASE) for p in KEY_PATTERNS]
        self._chord_pattern = re.compile(CHORD_PATTERN)
        
        # Build reverse lookup: word → (category, canonical_name)
        self._genre_lookup = self._build_reverse_lookup(GENRE_KEYWORDS)
        self._emotion_lookup = self._build_reverse_lookup(EMOTION_KEYWORDS)
    
    def _build_reverse_lookup(self, keyword_dict: Dict[str, Set[str]]) -> Dict[str, str]:
        """
        Build reverse lookup from keyword to canonical category.
        
        Args:
            keyword_dict: e.g., {'folk': {'folk', 'folksy', 'campfire', ...}}
        
        Returns:
            Reverse mapping: {'folksy': 'folk', 'campfire': 'folk', ...}
        """
        lookup = {}
        for canonical, keywords in keyword_dict.items():
            for keyword in keywords:
                lookup[keyword.lower()] = canonical
        return lookup
    
    def _log(self, message: str) -> None:
        """Print debug message if verbose mode is on."""
        if self.verbose:
            print(f"  [Parser] {message}")
    
    # =========================================================================
    # MAIN PARSING METHOD
    # =========================================================================
    
    def parse(self, prompt: str) -> PromptFeatures:
        """
        Parse a natural language prompt into structured musical features.
        
        This is the main entry point. It orchestrates all the extractors
        and assembles the final PromptFeatures object.
        
        Args:
            prompt: Natural language input like "sad folk in Am, slow"
        
        Returns:
            PromptFeatures with all extracted (or defaulted) values
        """
        self._log(f"Parsing: \"{prompt}\"")
        
        # Normalize input
        prompt_lower = prompt.lower().strip()
        prompt_clean = prompt.strip()
        
        # Initialize tracking
        confidence = ExtractionConfidence()
        explicitly_stated = {
            'key': False, 'mode': False, 'genre': False,
            'emotion': False, 'tempo': False
        }
        warnings = []
        
        # =====================================================================
        # EXTRACTION PHASE
        # =====================================================================
        
        # 1. Extract key and mode
        key, mode, key_conf, mode_conf = self._extract_key_mode(prompt_clean)
        confidence.key = key_conf
        confidence.mode = mode_conf
        explicitly_stated['key'] = key_conf > 0.5
        explicitly_stated['mode'] = mode_conf > 0.5
        
        if key_conf == 0.0:
            warnings.append(f"Key not specified, using default: {DEFAULT_FEATURES['key']}")
            key = DEFAULT_FEATURES['key']
        
        if mode_conf == 0.0:
            # Try to infer mode from emotion words
            inferred_mode = self._infer_mode_from_context(prompt_lower)
            if inferred_mode:
                mode = inferred_mode
                confidence.mode = 0.4  # Low confidence inference
                self._log(f"Inferred mode '{mode}' from emotional context")
            else:
                warnings.append(f"Mode not specified, using default: {DEFAULT_FEATURES['mode']}")
                mode = DEFAULT_FEATURES['mode']
        
        # 2. Extract genre
        genre, genre_conf = self._extract_genre(prompt_lower)
        confidence.genre = genre_conf
        explicitly_stated['genre'] = genre_conf > 0.5
        
        if genre_conf == 0.0:
            warnings.append(f"Genre not specified, using default: {DEFAULT_FEATURES['genre']}")
            genre = DEFAULT_FEATURES['genre']
        
        # 3. Extract emotion
        emotion, emotion_conf = self._extract_emotion(prompt_lower)
        confidence.emotion = emotion_conf
        explicitly_stated['emotion'] = emotion_conf > 0.5
        
        if emotion_conf == 0.0:
            warnings.append(f"Emotion not specified, using default: {DEFAULT_FEATURES['emotion']}")
            emotion = DEFAULT_FEATURES['emotion']
        
        # 4. Extract tempo
        tempo, tempo_conf = self._extract_tempo(prompt_lower)
        confidence.tempo = tempo_conf
        explicitly_stated['tempo'] = tempo_conf > 0.5
        
        if tempo_conf == 0.0:
            # Try to infer tempo from genre/emotion
            inferred_tempo = self._infer_tempo_from_context(genre, emotion)
            tempo = inferred_tempo
            confidence.tempo = 0.3  # Low confidence inference
            self._log(f"Inferred tempo {tempo} BPM from genre/emotion context")
        
        # 5. Extract any explicit chords
        extracted_chords = self._extract_chords(prompt_clean)
        
        # =====================================================================
        # ASSEMBLE RESULT
        # =====================================================================
        
        features = PromptFeatures(
            key=key,
            mode=mode,
            genre=genre,
            emotion=emotion,
            tempo=tempo,
            time_signature=DEFAULT_FEATURES['time_signature'],
            original_prompt=prompt_clean,
            confidence=confidence,
            extracted_chords=extracted_chords,
            warnings=warnings,
            explicitly_stated=explicitly_stated
        )
        
        self._log(f"Result: {features}")
        return features
    
    # =========================================================================
    # INDIVIDUAL EXTRACTORS
    # =========================================================================
    
    def _extract_key_mode(self, prompt: str) -> Tuple[str, str, float, float]:
        """
        Extract musical key and mode from prompt.
        
        Looks for patterns like:
        - "in E minor" → E, minor, high confidence
        - "key of C" → C, (need to infer mode), high key confidence
        - "Am" → A, minor, medium confidence
        
        Args:
            prompt: Input text
        
        Returns:
            Tuple of (key, mode, key_confidence, mode_confidence)
        """
        self._log("Extracting key/mode...")
        
        # Try each pattern in order of specificity
        for i, pattern in enumerate(self._key_patterns):
            match = pattern.search(prompt)
            if match:
                key = match.group(1).upper()
                mode_indicator = match.group(2) if len(match.groups()) > 1 else None
                
                # Determine mode from match
                if mode_indicator:
                    mode_lower = mode_indicator.lower()
                    if mode_lower in ('m', 'min', 'minor'):
                        mode = 'minor'
                    elif mode_lower in ('maj', 'major'):
                        mode = 'major'
                    else:
                        mode = DEFAULT_FEATURES['mode']
                    
                    self._log(f"Found key={key}, mode={mode} (pattern {i+1})")
                    return key, mode, 0.95, 0.95
                else:
                    # Key found but mode not explicit
                    self._log(f"Found key={key}, mode not specified")
                    return key, DEFAULT_FEATURES['mode'], 0.90, 0.0
        
        # No key found
        self._log("No key pattern matched")
        return DEFAULT_FEATURES['key'], DEFAULT_FEATURES['mode'], 0.0, 0.0
    
    def _infer_mode_from_context(self, prompt_lower: str) -> Optional[str]:
        """
        Try to infer mode from emotional/descriptive words.
        
        "Sad" typically implies minor, "happy" implies major.
        This is a low-confidence inference used as fallback.
        
        Args:
            prompt_lower: Lowercased prompt text
        
        Returns:
            'minor' or 'major' if inferred, None otherwise
        """
        words = set(prompt_lower.split())
        
        minor_score = len(words.intersection(MINOR_INDICATORS))
        major_score = len(words.intersection(MAJOR_INDICATORS))
        
        if minor_score > major_score:
            return 'minor'
        elif major_score > minor_score:
            return 'major'
        
        return None
    
    def _extract_genre(self, prompt_lower: str) -> Tuple[str, float]:
        """
        Extract genre from prompt using keyword matching.
        
        Uses synonym expansion: "campfire" → folk, "groovy" → blues
        
        Args:
            prompt_lower: Lowercased prompt text
        
        Returns:
            Tuple of (genre, confidence)
        """
        self._log("Extracting genre...")
        
        # Check for multi-word phrases first
        for canonical, keywords in GENRE_KEYWORDS.items():
            for keyword in keywords:
                if ' ' in keyword and keyword in prompt_lower:
                    self._log(f"Found genre phrase: '{keyword}' → {canonical}")
                    return canonical, 0.90
        
        # Then check individual words
        words = prompt_lower.split()
        genre_matches = []
        
        for word in words:
            # Clean punctuation
            clean_word = re.sub(r'[^\w\s-]', '', word)
            if clean_word in self._genre_lookup:
                genre_matches.append(self._genre_lookup[clean_word])
        
        if genre_matches:
            # Return most common match (in case of conflicts)
            from collections import Counter
            most_common = Counter(genre_matches).most_common(1)[0][0]
            conf = 0.85 if genre_matches.count(most_common) > 1 else 0.75
            self._log(f"Found genre: {most_common} (confidence: {conf})")
            return most_common, conf
        
        self._log("No genre found")
        return DEFAULT_FEATURES['genre'], 0.0
    
    def _extract_emotion(self, prompt_lower: str) -> Tuple[str, float]:
        """
        Extract emotion/mood from prompt using keyword matching.
        
        Uses extensive synonym expansion for each emotion category.
        
        Args:
            prompt_lower: Lowercased prompt text
        
        Returns:
            Tuple of (emotion, confidence)
        """
        self._log("Extracting emotion...")
        
        # Check multi-word phrases first
        for canonical, keywords in EMOTION_KEYWORDS.items():
            for keyword in keywords:
                if ' ' in keyword and keyword in prompt_lower:
                    self._log(f"Found emotion phrase: '{keyword}' → {canonical}")
                    return canonical, 0.90
        
        # Then check individual words
        words = prompt_lower.split()
        emotion_matches = []
        
        for word in words:
            clean_word = re.sub(r'[^\w\s-]', '', word)
            if clean_word in self._emotion_lookup:
                emotion_matches.append(self._emotion_lookup[clean_word])
        
        if emotion_matches:
            from collections import Counter
            most_common = Counter(emotion_matches).most_common(1)[0][0]
            conf = 0.85 if emotion_matches.count(most_common) > 1 else 0.75
            self._log(f"Found emotion: {most_common} (confidence: {conf})")
            return most_common, conf
        
        self._log("No emotion found")
        return DEFAULT_FEATURES['emotion'], 0.0
    
    def _extract_tempo(self, prompt_lower: str) -> Tuple[int, float]:
        """
        Extract tempo from prompt.
        
        Handles:
        - Explicit BPM: "at 120 bpm" → 120
        - Keywords: "slow" → 55-75 BPM range
        
        Args:
            prompt_lower: Lowercased prompt text
        
        Returns:
            Tuple of (tempo_bpm, confidence)
        """
        self._log("Extracting tempo...")
        
        # First, check for explicit BPM
        bpm_pattern = r'(\d{2,3})\s*(?:bpm|beats?\s*per\s*min)'
        bpm_match = re.search(bpm_pattern, prompt_lower)
        if bpm_match:
            bpm = int(bpm_match.group(1))
            # Clamp to valid range
            bpm = max(40, min(200, bpm))
            self._log(f"Found explicit BPM: {bpm}")
            return bpm, 0.95
        
        # Check for tempo keywords (longer phrases first)
        sorted_keywords = sorted(TEMPO_KEYWORDS.keys(), key=len, reverse=True)
        for keyword in sorted_keywords:
            if keyword in prompt_lower:
                low, high = TEMPO_KEYWORDS[keyword]
                # Return middle of range
                tempo = (low + high) // 2
                self._log(f"Found tempo keyword '{keyword}' → {tempo} BPM")
                return tempo, 0.80
        
        self._log("No tempo indicator found")
        return DEFAULT_FEATURES['tempo'], 0.0
    
    def _infer_tempo_from_context(self, genre: str, emotion: str) -> int:
        """
        Infer tempo from genre and emotion when not explicitly stated.
        
        This is a low-confidence fallback that uses genre/emotion
        conventions to make reasonable guesses.
        
        Args:
            genre: Extracted genre
            emotion: Extracted emotion
        
        Returns:
            Inferred BPM
        """
        # Genre-based tempo tendencies
        genre_tempos = {
            'ballad': 65,
            'blues': 85,
            'jazz': 95,
            'folk': 95,
            'acoustic': 90,
            'indie': 100,
            'pop': 110,
            'country': 115,
            'rock': 125
        }
        
        # Emotion-based tempo adjustments
        emotion_adjustments = {
            'peaceful': -20,
            'melancholic': -15,
            'mellow': -10,
            'nostalgic': -5,
            'hopeful': 0,
            'dramatic': +5,
            'upbeat': +15,
            'energetic': +25
        }
        
        base = genre_tempos.get(genre, DEFAULT_FEATURES['tempo'])
        adjustment = emotion_adjustments.get(emotion, 0)
        
        tempo = base + adjustment
        # Clamp to valid range
        tempo = max(40, min(200, tempo))
        
        return tempo
    
    def _extract_chords(self, prompt: str) -> List[str]:
        """
        Extract any explicitly mentioned chord symbols.
        
        Handles patterns like:
        - "use Am, G, C, F"
        - "with chords: Em - D - C"
        - "Am to G progression"
        
        Args:
            prompt: Original prompt text
        
        Returns:
            List of chord symbols found (may be empty)
        """
        self._log("Extracting explicit chords...")
        
        # Find all chord-like patterns
        matches = self._chord_pattern.findall(prompt)
        
        # Filter out false positives (single letters that aren't chords)
        valid_chords = []
        for match in matches:
            # Must be at least a root note
            if len(match) >= 1 and match[0].upper() in 'ABCDEFG':
                # Avoid matching common words that look like chords
                # (e.g., "A" as article, "Am" as "am/are")
                if match.lower() not in ('a', 'am', 'be', 'ad'):
                    valid_chords.append(match)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_chords = []
        for chord in valid_chords:
            if chord not in seen:
                seen.add(chord)
                unique_chords.append(chord)
        
        if unique_chords:
            self._log(f"Found explicit chords: {unique_chords}")
        
        return unique_chords


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def parse_prompt(prompt: str, verbose: bool = False) -> PromptFeatures:
    """
    Convenience function to parse a prompt without creating parser instance.
    
    Args:
        prompt: Natural language input
        verbose: Print debug info if True
    
    Returns:
        PromptFeatures with extracted values
    
    Example:
        >>> features = parse_prompt("sad folk in Am")
        >>> print(features.full_key)
        'Am'
    """
    parser = PromptParser(verbose=verbose)
    return parser.parse(prompt)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Prompt Parser")
    print("=" * 70)
    
    # Create parser in verbose mode for testing
    parser = PromptParser(verbose=True)
    
    # Test cases covering various scenarios
    test_prompts = [
        # Full specification
        "Give me a sad folk song in E minor at a slow pace",
        
        # Explicit BPM
        "upbeat pop progression in G major at 128 bpm",
        
        # Minimal input
        "something chill for the evening",
        
        # Key only
        "in C major",
        
        # Genre/emotion rich, no key
        "dark melancholic ballad, very slow and emotional",
        
        # With explicit chords
        "use Am, G, C, F for a nostalgic indie track",
        
        # Shorthand key (Am = A minor)
        "soft acoustic Am progression",
        
        # Complex/conflicting
        "happy but dark rock in Dm",
        
        # Near-empty
        "guitar",
        
        # Tempo keywords
        "energetic driving rock anthem",
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}: \"{prompt}\"")
        print("=" * 70)
        
        features = parser.parse(prompt)
        print(f"\n{features.summary()}")
        print(f"\nDictionary export keys: {list(features.to_dict().keys())[:5]}...")
    
    print("\n" + "=" * 70)
    print("All parsing tests complete!")
    print("=" * 70)
