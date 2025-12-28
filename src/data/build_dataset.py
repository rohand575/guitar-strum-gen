"""
Dataset Builder - Generate Training Data for Guitar Strum Generator

This module creates the training dataset by combining:
    1. SYNTHETIC samples (70%) - Generated using rule-based system
    2. REAL progressions (30%) - Sampled from Chordonomicon dataset

The generated dataset follows the GuitarSample schema and is saved as JSONL.

Author: Rohan Rajendra Dhanawade
Thesis: A Conversational AI System for Symbolic Guitar Strumming Pattern 
        and Chord Progression Generation
"""

import json
import random
import uuid
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

# Import our modules
from src.data.schema import GuitarSample, VALID_GENRES, VALID_EMOTIONS
from src.rules.harmony import select_progression, get_diatonic_chords
from src.rules.strumming import select_strumming_pattern, pick_tempo, TEMPO_RANGES


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    total_samples: int = 150
    synthetic_ratio: float = 0.70  # 70% synthetic, 30% real
    train_ratio: float = 0.70     # 70% train, 15% val, 15% test
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    
    @property
    def num_synthetic(self) -> int:
        return int(self.total_samples * self.synthetic_ratio)
    
    @property
    def num_real(self) -> int:
        return self.total_samples - self.num_synthetic


# =============================================================================
# PROMPT TEMPLATES - The Key to Natural Language Variety
# =============================================================================

# Template categories for diverse prompt generation
# {genre}, {emotion}, {key}, {mode}, {tempo}, {tempo_word} are placeholders

PROMPT_TEMPLATES = [
    # =========== Category 1: Direct/Formal Requests (8) ===========
    "Give me a {emotion} {genre} progression in {key} {mode}",
    "I need a {emotion} {genre} song in {key} {mode}, {tempo_word} tempo",
    "Create a {emotion} {genre} chord progression in the key of {key} {mode}",
    "Generate a {genre} progression in {key} {mode} with a {emotion} feel",
    "Produce a {emotion} {genre} sequence in {key} {mode}",
    "I'd like a {tempo_word} {emotion} {genre} progression in {key}",
    "Make me a {emotion} {genre} in {key} {mode}",
    "Show me a {genre} chord progression, {emotion} style, {key} {mode}",
    
    # =========== Category 2: Casual/Conversational (10) ===========
    "{emotion} {genre} vibes in {key} {mode}",
    "{emotion} {genre} in {key}, {tempo_word}",
    "something {emotion} and {genre}-ish in {key} {mode}",
    "give me something {emotion} in {key}",
    "play me a {genre} thing, {emotion} vibes",
    "I want {genre} chords that feel {emotion}",
    "how about a {emotion} {genre} in {key}?",
    "let's do a {genre} progression, {emotion} style",
    "hit me with some {emotion} {genre} in {key}",
    "throw me a {emotion} {genre} progression",
    
    # =========== Category 3: Tempo-Focused (8) ===========
    "{emotion} {genre} progression in {key} {mode} at {tempo} BPM",
    "{genre} song in {key} {mode}, around {tempo} BPM, {emotion} feel",
    "I want a {tempo_word} {emotion} {genre} in {key} {mode}",
    "{tempo} BPM {genre} with {emotion} energy in {key}",
    "{genre} at {tempo} BPM, {emotion} mood",
    "a {tempo_word} {genre} progression in {key} {mode}",
    "{tempo_word} and {emotion} {genre} in {key}",
    "{genre} progression, {tempo_word} tempo, {emotion}, {key} {mode}",
    
    # =========== Category 4: Mood-First (8) ===========
    "a {tempo_word}, {emotion} {genre} piece in {key} {mode}",
    "{tempo_word} {genre} with {emotion} mood, key of {key} {mode}",
    "looking for {emotion} {genre} chords in {key} {mode}",
    "{emotion} feeling, {genre} style, key of {key}",
    "capture a {emotion} mood with {genre} chords in {key}",
    "I'm feeling {emotion}, give me {genre} in {key}",
    "need something {emotion} - {genre} in {key} {mode}",
    "{emotion} {genre} sound in {key} {mode}",
    
    # =========== Category 5: Action-Oriented (6) ===========
    "write me a {emotion} {genre} progression in {key} {mode}",
    "generate a {emotion} {genre} in {key} {mode}, {tempo_word} tempo",
    "make a {tempo_word} {emotion} {genre} song in {key} {mode}",
    "compose a {emotion} {genre} progression in {key}",
    "build a {genre} progression with {emotion} feel in {key} {mode}",
    "craft a {tempo_word} {genre} in {key}, {emotion} mood",
    
    # =========== Category 6: Simple/Minimal (8) ===========
    "{emotion} {genre} in {key}",
    "{genre} {key} {mode} {emotion}",
    "{tempo_word} {genre} in {key} {mode}",
    "{genre} in {key}",
    "{emotion} {genre}",
    "{key} {mode} {genre}",
    "{genre} {emotion} {tempo}bpm",
    "some {genre} in {key}",
    
    # =========== Category 7: Question Format (6) ===========
    "can you give me a {emotion} {genre} in {key} {mode}?",
    "what's a good {emotion} {genre} progression in {key} {mode}?",
    "what chords work for a {emotion} {genre} song in {key}?",
    "any ideas for {genre} in {key} with {emotion} vibe?",
    "got any {emotion} {genre} progressions in {key}?",
    "know any {tempo_word} {genre} patterns in {key} {mode}?",
    
    # =========== Category 8: Context/Story (8) ===========
    "{emotion} {genre} with strumming pattern, {key} {mode}",
    "need chords and strum for {emotion} {genre} in {key} {mode}",
    "{genre} progression with strum pattern, {emotion} feel, {key} {mode}",
    "I'm writing a {emotion} {genre} song in {key}",
    "working on a {tempo_word} {genre} track in {key} {mode}",
    "practicing {genre} and want a {emotion} progression in {key}",
    "jamming in {key} {mode}, need {emotion} {genre} chords",
    "recording a {emotion} {genre} piece in {key}",
    
    # =========== Category 9: Descriptive/Poetic (6) ===========
    "a {genre} progression that sounds {emotion} in {key}",
    "{emotion} chords in {genre} style, {key} {mode}",
    "the kind of {genre} progression that feels {emotion}",
    "{genre} with a {emotion}, {tempo_word} feel in {key}",
    "dreamy {emotion} {genre} in {key} {mode}",
    "smooth {emotion} {genre} progression in {key}",
]

# Tempo word mappings for natural language
TEMPO_WORDS = {
    "slow": ["slow", "gentle", "relaxed", "laid-back", "easy"],
    "moderate": ["moderate", "medium", "mid-tempo", "steady"],
    "fast": ["fast", "upbeat", "energetic", "driving", "quick"],
}

# Emotion synonyms for variety
EMOTION_SYNONYMS = {
    "upbeat": ["upbeat", "happy", "cheerful", "bright", "joyful"],
    "melancholic": ["melancholic", "sad", "sorrowful", "melancholy"],
    "mellow": ["mellow", "chill", "relaxed", "smooth", "easy-going"],
    "energetic": ["energetic", "powerful", "intense", "driving"],
    "peaceful": ["peaceful", "calm", "serene", "gentle", "soft"],
    "dramatic": ["dramatic", "epic", "cinematic", "intense"],
    "hopeful": ["hopeful", "optimistic", "uplifting", "inspiring"],
    "nostalgic": ["nostalgic", "wistful", "bittersweet", "reminiscent"],
}

# Common keys for guitar (weighted by frequency in pop music)
COMMON_KEYS = {
    "major": ["G", "C", "D", "A", "E", "F"],
    "minor": ["Am", "Em", "Dm", "Bm", "Fm"],
}

# Simplified key roots for generation
KEY_ROOTS = ["C", "D", "E", "F", "G", "A", "B"]
SHARP_KEYS = ["C#", "F#", "G#"]  # Less common but valid


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_tempo_word(tempo: int) -> str:
    """Convert numeric tempo to a descriptive word."""
    if tempo <= 80:
        category = "slow"
    elif tempo <= 120:
        category = "moderate"
    else:
        category = "fast"
    return random.choice(TEMPO_WORDS[category])


def get_emotion_variant(emotion: str) -> str:
    """Get a synonym for an emotion to add variety."""
    if emotion in EMOTION_SYNONYMS:
        return random.choice(EMOTION_SYNONYMS[emotion])
    return emotion


def generate_random_parameters() -> Dict:
    """Generate random musical parameters for a synthetic sample."""
    # Pick genre
    genre = random.choice(VALID_GENRES)
    
    # Pick emotion
    emotion = random.choice(VALID_EMOTIONS)
    
    # Pick mode (weighted: 70% major, 30% minor for guitar songs)
    mode = random.choices(["major", "minor"], weights=[0.7, 0.3])[0]
    
    # Pick key based on mode
    if mode == "major":
        key = random.choice(COMMON_KEYS["major"])
    else:
        # For minor, we need just the root (e.g., "A" not "Am")
        minor_key = random.choice(COMMON_KEYS["minor"])
        key = minor_key.replace("m", "")  # "Am" -> "A"
    
    # Pick tempo based on emotion (more realistic)
    tempo = pick_tempo(emotion=emotion)
    
    return {
        "genre": genre,
        "emotion": emotion,
        "mode": mode,
        "key": key,
        "tempo": tempo,
    }


def build_prompt(params: Dict) -> str:
    """Build a natural language prompt from parameters."""
    template = random.choice(PROMPT_TEMPLATES)
    
    # Get variations
    emotion_word = get_emotion_variant(params["emotion"])
    tempo_word = get_tempo_word(params["tempo"])
    
    # Format the key display (e.g., "G major" or "A minor")
    key_display = params["key"]
    mode_display = params["mode"]
    
    # Build the prompt
    prompt = template.format(
        genre=params["genre"],
        emotion=emotion_word,
        key=key_display,
        mode=mode_display,
        tempo=params["tempo"],
        tempo_word=tempo_word,
    )
    
    # Random capitalization (sometimes capitalize, sometimes not)
    if random.random() < 0.3:
        prompt = prompt.capitalize()
    
    return prompt


# =============================================================================
# SYNTHETIC SAMPLE GENERATION
# =============================================================================

def generate_synthetic_sample(sample_id: str) -> GuitarSample:
    """
    Generate a single synthetic training sample.
    
    Process:
        1. Generate random parameters (genre, emotion, key, mode, tempo)
        2. Build natural language prompt from template
        3. Generate chord progression using harmony rules
        4. Select strumming pattern using strumming rules
        5. Package as GuitarSample
    
    Args:
        sample_id: Unique identifier for this sample
        
    Returns:
        Validated GuitarSample instance
    """
    # Step 1: Random parameters
    params = generate_random_parameters()
    
    # Step 2: Build prompt
    prompt = build_prompt(params)
    
    # Step 3: Generate chords using our rule-based system
    chords = select_progression(
        key=params["key"],
        mode=params["mode"],
        genre=params["genre"],
        emotion=params["emotion"]
    )
    
    # Step 4: Select strumming pattern
    strum_pattern = select_strumming_pattern(
        genre=params["genre"],
        tempo=params["tempo"],
        emotion=params["emotion"]
    )
    
    # Step 5: Build the key string for schema
    # Schema expects "G" for G major or "Am" for A minor
    if params["mode"] == "minor":
        key_str = params["key"] + "m"
    else:
        key_str = params["key"]
    
    # Create and return the sample
    return GuitarSample(
        id=sample_id,
        prompt=prompt,
        chords=chords,
        strum_pattern=strum_pattern,
        tempo=params["tempo"],
        time_signature="4/4",
        genre=params["genre"],
        emotion=params["emotion"],
        key=key_str,
        mode=params["mode"]
    )


def generate_synthetic_samples(
    num_samples: int,
    id_prefix: str = "syn"
) -> List[GuitarSample]:
    """
    Generate multiple synthetic samples.
    
    Args:
        num_samples: Number of samples to generate
        id_prefix: Prefix for sample IDs
        
    Returns:
        List of GuitarSample instances
    """
    samples = []
    for i in range(num_samples):
        sample_id = f"{id_prefix}_{i+1:04d}"
        sample = generate_synthetic_sample(sample_id)
        samples.append(sample)
    
    return samples


# =============================================================================
# CHORDONOMICON INTEGRATION
# =============================================================================

def load_chordonomicon(filepath: str) -> List[Dict]:
    """
    Load chord progressions from Chordonomicon dataset (local file).
    
    Expected format (CSV or JSON):
        - chord_sequence: List of chord symbols
        - key: Musical key
        - mode: major/minor
        
    Args:
        filepath: Path to Chordonomicon file
        
    Returns:
        List of progression dictionaries
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Chordonomicon file not found: {filepath}")
    
    progressions = []
    
    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                progressions = data
            else:
                progressions = data.get('progressions', [])
                
    elif filepath.suffix == '.csv':
        import csv
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse chord sequence from string
                chords_str = row.get('chords', row.get('chord_sequence', ''))
                if isinstance(chords_str, str):
                    # Handle different formats: "C,G,Am,F" or "['C','G','Am','F']"
                    chords_str = chords_str.strip('[]').replace("'", "").replace('"', '')
                    chords = [c.strip() for c in chords_str.split(',')]
                else:
                    chords = chords_str
                
                progressions.append({
                    'chords': chords,
                    'key': row.get('key', 'C'),
                    'mode': row.get('mode', 'major'),
                    'genre': row.get('genre', None),
                })
    
    elif filepath.suffix == '.jsonl':
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    progressions.append(json.loads(line))
    
    return progressions


# =============================================================================
# HUGGING FACE CHORDONOMICON INTEGRATION (NEW!)
# =============================================================================

# Mapping from Chordonomicon genre strings to our vocabulary
CHORDONOMICON_GENRE_MAP = {
    "rock": "rock",
    "pop": "pop", 
    "country": "country",
    "folk": "folk",
    "blues": "blues",
    "jazz": "jazz",
    "indie": "indie",
    "alternative": "indie",
    "acoustic": "acoustic",
    "singer-songwriter": "acoustic",
    "classic rock": "rock",
    "hard rock": "rock",
    "soft rock": "rock",
    "indie rock": "indie",
    "folk rock": "folk",
    "country rock": "country",
    "americana": "folk",
    "r&b": "pop",
    "soul": "blues",
    "metal": "rock",
}


def parse_chordonomicon_chord_string(chord_string: str) -> Tuple[List[str], Optional[str]]:
    """
    Parse chord progression from Chordonomicon format.
    
    Chordonomicon uses structural tags: "<intro_1> C <verse_1> F C E7 Am G..."
    We extract just the chord symbols.
    
    Args:
        chord_string: Raw chord string from Chordonomicon
        
    Returns:
        Tuple of (list of chords, detected section)
    """
    import re
    
    # Find section tags like <intro_1>, <verse_1>, <chorus_1>
    section_pattern = r'<(\w+)_?\d*>'
    sections_found = re.findall(section_pattern, chord_string)
    
    # Remove all section tags
    clean_string = re.sub(section_pattern, ' ', chord_string)
    
    # Split into tokens
    tokens = clean_string.split()
    
    # Filter to valid chord symbols
    # Valid chord pattern: root note + optional accidental + optional quality
    chord_pattern = r'^[A-Ga-g][#b]?(m|maj|min|dim|aug|sus|add|7|9|11|13|M)*\d*$'
    
    chords = []
    for token in tokens:
        token = token.strip()
        if token and re.match(chord_pattern, token, re.IGNORECASE):
            # Normalize: capitalize root, keep rest as-is
            normalized = normalize_chord(token)
            if normalized:
                chords.append(normalized)
    
    # Return first section found (or None)
    section = sections_found[0] if sections_found else None
    
    return chords, section


def normalize_chord(chord: str) -> Optional[str]:
    """
    Normalize a chord symbol to our schema format.
    
    Examples:
        "am" -> "Am"
        "Cmaj" -> "C"
        "f#m" -> "F#m"
    """
    if not chord:
        return None
    
    # Uppercase the root note
    if len(chord) >= 1:
        chord = chord[0].upper() + chord[1:]
    
    # Handle accidental (second character)
    if len(chord) >= 2 and chord[1] in ['#', 'b']:
        chord = chord[0] + chord[1] + chord[2:]
    
    # Normalize common variations
    replacements = [
        ("maj7", "maj7"),  # Keep maj7
        ("Maj7", "maj7"),
        ("maj", ""),       # Cmaj -> C
        ("Maj", ""),
        ("min", "m"),      # Cmin -> Cm
        ("Min", "m"),
        ("M7", "maj7"),    # CM7 -> Cmaj7
    ]
    
    for old, new in replacements:
        if old in chord:
            chord = chord.replace(old, new)
    
    # Validate against our schema
    from src.data.schema import is_valid_chord
    if is_valid_chord(chord):
        return chord
    
    # Try simplifying (just root + m/nothing)
    import re
    simple = re.match(r'^([A-G][#b]?)(m)?', chord)
    if simple:
        root = simple.group(1)
        minor = simple.group(2) or ""
        simplified = root + minor
        if is_valid_chord(simplified):
            return simplified
    
    return None


def map_chordonomicon_genre(genre_string: str) -> str:
    """Map Chordonomicon genre to our vocabulary."""
    if not genre_string:
        return random.choice(VALID_GENRES)
    
    genre_lower = genre_string.lower()
    
    for pattern, our_genre in CHORDONOMICON_GENRE_MAP.items():
        if pattern in genre_lower:
            return our_genre
    
    return random.choice(VALID_GENRES)


def detect_mode_from_chords(chords: List[str]) -> str:
    """Detect major/minor mode from chord progression."""
    if not chords:
        return "major"
    
    # Check first and last chord
    first = chords[0]
    last = chords[-1]
    
    # Minor if first/last chord is minor (ends with 'm' but not 'maj')
    def is_minor(c):
        return c.endswith('m') and not 'maj' in c.lower()
    
    if is_minor(first) or is_minor(last):
        return "minor"
    
    # Count minor vs major
    minor_count = sum(1 for c in chords if is_minor(c))
    if minor_count > len(chords) // 2:
        return "minor"
    
    return "major"


def detect_key_from_chords(chords: List[str], mode: str) -> str:
    """Detect key from first chord."""
    if not chords:
        return "C" if mode == "major" else "Am"
    
    import re
    first = chords[0]
    
    # Extract root
    match = re.match(r'^([A-G][#b]?)', first)
    if match:
        root = match.group(1)
        if mode == "minor":
            return root + "m"
        return root
    
    return "C" if mode == "major" else "Am"


def load_chordonomicon_huggingface(num_samples: int = 75) -> List[Dict]:
    """
    Load chord progressions directly from Hugging Face.
    
    Uses: https://huggingface.co/datasets/ailsntua/Chordonomicon
    
    Args:
        num_samples: Number of samples to load
        
    Returns:
        List of progression dictionaries ready for our pipeline
    """
    print("  📥 Loading from Hugging Face (ailsntua/Chordonomicon)...")
    
    try:
        # Try using datasets library
        from datasets import load_dataset
        
        dataset = load_dataset("ailsntua/Chordonomicon", split="train")
        df = dataset.to_pandas()
        
        print(f"  ✓ Loaded {len(df)} total entries from Chordonomicon")
        
    except ImportError:
        print("  ⚠️  'datasets' library not installed. Trying pandas direct load...")
        try:
            import pandas as pd
            df = pd.read_csv("hf://datasets/ailsntua/Chordonomicon/chordonomicon_v2.csv")
        except Exception as e:
            print(f"  ❌ Could not load from Hugging Face: {e}")
            return []
    
    # Filter to entries with chord data
    df = df[df['chords'].notna()]
    df = df[df['chords'].str.len() > 10]
    
    print(f"  ✓ {len(df)} entries have valid chord data")
    
    # Sample more than needed (some will fail parsing)
    sample_size = min(num_samples * 3, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    progressions = []
    successful = 0
    
    for _, row in df_sample.iterrows():
        if successful >= num_samples:
            break
            
        try:
            # Parse the chord string
            chords, section = parse_chordonomicon_chord_string(str(row['chords']))
            
            # Skip if too few or too many
            if len(chords) < 2 or len(chords) > 12:
                continue
            
            # Remove consecutive duplicates and limit to 8
            unique_chords = []
            for c in chords:
                if not unique_chords or c != unique_chords[-1]:
                    unique_chords.append(c)
            chords = unique_chords[:8]
            
            if len(chords) < 2:
                continue
            
            # Detect mode and key
            mode = detect_mode_from_chords(chords)
            key = detect_key_from_chords(chords, mode)
            
            # Map genre
            genre_str = str(row.get('genres', '')) or str(row.get('main_genre', ''))
            genre = map_chordonomicon_genre(genre_str)
            
            progressions.append({
                'chords': chords,
                'key': key,
                'mode': mode,
                'genre': genre,
                'section': section,
            })
            successful += 1
            
        except Exception:
            continue
    
    print(f"  ✓ Successfully parsed {len(progressions)} progressions")
    return progressions


def create_sample_from_chordonomicon(
    progression: Dict,
    sample_id: str
) -> GuitarSample:
    """
    Create a GuitarSample from a Chordonomicon progression.
    
    We take the real chords and ADD:
        - Natural language prompt (generated from template)
        - Strumming pattern (selected using our rules)
        - Tempo (picked based on assumed genre/emotion)
    
    Args:
        progression: Dictionary with 'chords', 'key', 'mode' from Chordonomicon
        sample_id: Unique identifier
        
    Returns:
        GuitarSample instance
    """
    # Extract real chords
    chords = progression.get('chords', ['C', 'G', 'Am', 'F'])
    
    # Ensure we have 2-8 chords
    if len(chords) < 2:
        chords = chords * 2
    if len(chords) > 8:
        chords = chords[:8]
    
    # Get key and mode
    key_raw = progression.get('key', 'C')
    mode = progression.get('mode', 'major').lower()
    
    # Clean up key (remove 'm' suffix if present, we track mode separately)
    if key_raw.endswith('m') and len(key_raw) <= 3:
        key = key_raw[:-1]
        mode = 'minor'
    else:
        key = key_raw
    
    # Assign genre and emotion (random if not provided)
    genre = progression.get('genre')
    if not genre or genre not in VALID_GENRES:
        genre = random.choice(VALID_GENRES)
    
    emotion = random.choice(VALID_EMOTIONS)
    
    # Pick tempo based on emotion
    tempo = pick_tempo(emotion=emotion)
    
    # Select strumming pattern
    strum_pattern = select_strumming_pattern(
        genre=genre,
        tempo=tempo,
        emotion=emotion
    )
    
    # Build prompt
    params = {
        'genre': genre,
        'emotion': emotion,
        'key': key,
        'mode': mode,
        'tempo': tempo,
    }
    prompt = build_prompt(params)
    
    # Build key string for schema
    if mode == 'minor':
        key_str = key + 'm'
    else:
        key_str = key
    
    return GuitarSample(
        id=sample_id,
        prompt=prompt,
        chords=chords,
        strum_pattern=strum_pattern,
        tempo=tempo,
        time_signature="4/4",
        genre=genre,
        emotion=emotion,
        key=key_str,
        mode=mode
    )


def generate_samples_from_chordonomicon(
    filepath: str,
    num_samples: int,
    id_prefix: str = "real"
) -> List[GuitarSample]:
    """
    Generate samples using real progressions from Chordonomicon.
    
    Args:
        filepath: Path to Chordonomicon file
        num_samples: Number of samples to generate
        id_prefix: Prefix for sample IDs
        
    Returns:
        List of GuitarSample instances
    """
    # Load all progressions
    all_progressions = load_chordonomicon(filepath)
    
    if len(all_progressions) == 0:
        raise ValueError("No progressions found in Chordonomicon file")
    
    # Sample randomly (with replacement if needed)
    if len(all_progressions) >= num_samples:
        selected = random.sample(all_progressions, num_samples)
    else:
        # Sample with replacement
        selected = random.choices(all_progressions, k=num_samples)
    
    # Create samples
    samples = []
    for i, prog in enumerate(selected):
        sample_id = f"{id_prefix}_{i+1:04d}"
        try:
            sample = create_sample_from_chordonomicon(prog, sample_id)
            samples.append(sample)
        except Exception as e:
            print(f"Warning: Skipping progression {i}: {e}")
            # Generate synthetic as fallback
            sample = generate_synthetic_sample(f"{id_prefix}_fallback_{i+1:04d}")
            samples.append(sample)
    
    return samples


# =============================================================================
# DATASET BUILDING
# =============================================================================

def build_dataset(
    config: DatasetConfig,
    chordonomicon_path: Optional[str] = None,
    use_huggingface: bool = True,
    output_dir: str = "data/processed"
) -> Dict[str, List[GuitarSample]]:
    """
    Build the complete training dataset.
    
    Args:
        config: Dataset configuration
        chordonomicon_path: Path to local Chordonomicon file (optional)
        use_huggingface: If True, load from Hugging Face (recommended)
        output_dir: Directory to save output files
        
    Returns:
        Dictionary with 'train', 'val', 'test' splits
    """
    random.seed(config.random_seed)
    
    print("=" * 60)
    print("🎸 GUITAR STRUM GENERATOR - DATASET BUILDER")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Total samples: {config.total_samples}")
    print(f"  Synthetic: {config.num_synthetic} ({config.synthetic_ratio*100:.0f}%)")
    print(f"  Real (Chordonomicon): {config.num_real} ({(1-config.synthetic_ratio)*100:.0f}%)")
    
    all_samples = []
    
    # Generate synthetic samples
    print(f"\n--- Generating {config.num_synthetic} synthetic samples ---")
    synthetic_samples = generate_synthetic_samples(config.num_synthetic, "syn")
    all_samples.extend(synthetic_samples)
    print(f"  ✅ Generated {len(synthetic_samples)} synthetic samples")
    
    # Generate real samples from Chordonomicon
    chordonomicon_loaded = False
    
    # Option 1: Try Hugging Face first (recommended)
    if use_huggingface and config.num_real > 0:
        print(f"\n--- Loading {config.num_real} real progressions from Hugging Face ---")
        try:
            real_progressions = load_chordonomicon_huggingface(config.num_real)
            if real_progressions:
                real_samples = []
                for i, prog in enumerate(real_progressions):
                    sample = create_sample_from_chordonomicon(prog, f"real_{i+1:04d}")
                    real_samples.append(sample)
                all_samples.extend(real_samples)
                print(f"  ✅ Generated {len(real_samples)} samples from Chordonomicon")
                chordonomicon_loaded = True
        except Exception as e:
            print(f"  ⚠️  Hugging Face loading failed: {e}")
    
    # Option 2: Try local file
    if not chordonomicon_loaded and chordonomicon_path and Path(chordonomicon_path).exists():
        print(f"\n--- Loading from local file: {chordonomicon_path} ---")
        try:
            real_samples = generate_samples_from_chordonomicon(
                chordonomicon_path,
                config.num_real,
                "real"
            )
            all_samples.extend(real_samples)
            print(f"  ✅ Generated {len(real_samples)} samples from local Chordonomicon")
            chordonomicon_loaded = True
        except Exception as e:
            print(f"  ⚠️  Local file loading failed: {e}")
    
    # Fallback: Generate more synthetic if Chordonomicon failed
    if not chordonomicon_loaded and config.num_real > 0:
        print(f"\n--- Chordonomicon unavailable, generating more synthetic ---")
        extra_synthetic = generate_synthetic_samples(config.num_real, "syn_extra")
        all_samples.extend(extra_synthetic)
        print(f"  ✅ Generated {len(extra_synthetic)} additional synthetic samples")
    
    # Shuffle
    random.shuffle(all_samples)
    
    # Split into train/val/test
    total = len(all_samples)
    train_end = int(total * config.train_ratio)
    val_end = train_end + int(total * config.val_ratio)
    
    splits = {
        'train': all_samples[:train_end],
        'val': all_samples[train_end:val_end],
        'test': all_samples[val_end:],
    }
    
    print(f"\n--- Dataset Splits ---")
    print(f"  Train: {len(splits['train'])} samples")
    print(f"  Val:   {len(splits['val'])} samples")
    print(f"  Test:  {len(splits['test'])} samples")
    
    # Save to files
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save combined dataset
    all_path = output_path / "dataset.jsonl"
    with open(all_path, 'w') as f:
        for sample in all_samples:
            f.write(sample.model_dump_json() + '\n')
    print(f"\n  ✅ Saved: {all_path}")
    
    # Save splits
    for split_name, split_samples in splits.items():
        split_path = output_path / f"{split_name}.jsonl"
        with open(split_path, 'w') as f:
            for sample in split_samples:
                f.write(sample.model_dump_json() + '\n')
        print(f"  ✅ Saved: {split_path}")
    
    # Save statistics
    stats = dataset_statistics(all_samples)
    stats_path = output_path / "stats.json"
    with open(stats_path, 'w') as f:
        # Convert Counter objects to dicts for JSON serialization
        stats_json = {
            'total': stats['total'],
            'genres': dict(stats['genres']),
            'emotions': dict(stats['emotions']),
            'modes': dict(stats['modes']),
            'keys': dict(stats['keys']),
            'tempo_range': list(stats['tempo_range']),
            'avg_tempo': stats['avg_tempo'],
            'unique_patterns': stats['unique_patterns'],
            'unique_progressions': stats['unique_progressions'],
            'splits': {
                'train': len(splits['train']),
                'val': len(splits['val']),
                'test': len(splits['test']),
            }
        }
        json.dump(stats_json, f, indent=2)
    print(f"  ✅ Saved: {stats_path}")
    
    print(f"\n{'='*60}")
    print("✅ Dataset generation complete!")
    print(f"{'='*60}")
    
    return splits


# =============================================================================
# DATASET LOADING
# =============================================================================

def load_dataset(filepath: str) -> List[GuitarSample]:
    """
    Load a dataset from JSONL file.
    
    Args:
        filepath: Path to JSONL file
        
    Returns:
        List of GuitarSample instances
    """
    samples = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                samples.append(GuitarSample(**data))
    return samples


def dataset_statistics(samples: List[GuitarSample]) -> Dict:
    """
    Compute statistics about a dataset.
    
    Args:
        samples: List of GuitarSample instances
        
    Returns:
        Dictionary of statistics
    """
    from collections import Counter
    
    stats = {
        'total': len(samples),
        'genres': Counter(s.genre for s in samples),
        'emotions': Counter(s.emotion for s in samples),
        'modes': Counter(s.mode for s in samples),
        'keys': Counter(s.key for s in samples),
        'tempo_range': (
            min(s.tempo for s in samples),
            max(s.tempo for s in samples)
        ),
        'avg_tempo': sum(s.tempo for s in samples) / len(samples),
        'unique_patterns': len(set(s.strum_pattern for s in samples)),
        'unique_progressions': len(set(tuple(s.chords) for s in samples)),
    }
    
    return stats


def print_statistics(stats: Dict) -> None:
    """Pretty print dataset statistics."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    
    print(f"\nTotal samples: {stats['total']}")
    print(f"Tempo range: {stats['tempo_range'][0]} - {stats['tempo_range'][1]} BPM")
    print(f"Average tempo: {stats['avg_tempo']:.1f} BPM")
    print(f"Unique strumming patterns: {stats['unique_patterns']}")
    print(f"Unique chord progressions: {stats['unique_progressions']}")
    
    print(f"\nGenre distribution:")
    for genre, count in stats['genres'].most_common():
        print(f"  {genre:12} {count:3} ({count/stats['total']*100:5.1f}%)")
    
    print(f"\nEmotion distribution:")
    for emotion, count in stats['emotions'].most_common():
        print(f"  {emotion:12} {count:3} ({count/stats['total']*100:5.1f}%)")
    
    print(f"\nMode distribution:")
    for mode, count in stats['modes'].most_common():
        print(f"  {mode:12} {count:3} ({count/stats['total']*100:5.1f}%)")


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command-line interface for dataset building."""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build training dataset for Guitar Strum Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.data.build_dataset
  python -m src.data.build_dataset --samples 250
  python -m src.data.build_dataset --samples 300 --no-huggingface
  python -m src.data.build_dataset --local-file path/to/chordonomicon.csv
        """
    )
    parser.add_argument(
        '--samples', '-n',
        type=int,
        default=250,
        help='Total number of samples to generate (default: 250)'
    )
    parser.add_argument(
        '--synthetic-ratio',
        type=float,
        default=0.70,
        help='Ratio of synthetic samples (default: 0.70)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/processed',
        help='Output directory (default: data/processed)'
    )
    parser.add_argument(
        '--no-huggingface',
        action='store_true',
        help='Disable Hugging Face loading (synthetic only unless local file provided)'
    )
    parser.add_argument(
        '--local-file',
        type=str,
        default=None,
        help='Path to local Chordonomicon CSV file'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎸 GUITAR STRUM GENERATOR - DATASET BUILDER")
    print("=" * 60)
    
    # Configuration
    config = DatasetConfig(
        total_samples=args.samples,
        synthetic_ratio=args.synthetic_ratio,
        random_seed=args.seed
    )
    
    print(f"\n📊 Target: {args.samples} samples")
    print(f"   Synthetic ratio: {args.synthetic_ratio*100:.0f}%")
    print(f"   Hugging Face: {'Disabled' if args.no_huggingface else 'Enabled'}")
    if args.local_file:
        print(f"   Local file: {args.local_file}")
    
    # Build dataset
    splits = build_dataset(
        config=config,
        chordonomicon_path=args.local_file,
        use_huggingface=not args.no_huggingface,
        output_dir=args.output
    )
    
    # Show statistics
    all_samples = splits['train'] + splits['val'] + splits['test']
    stats = dataset_statistics(all_samples)
    print_statistics(stats)
    
    # Show a few examples
    print("\n" + "=" * 60)
    print("📝 SAMPLE EXAMPLES")
    print("=" * 60)
    
    for i, sample in enumerate(random.sample(all_samples, min(3, len(all_samples)))):
        print(f"\n--- Example {i+1} ---")
        print(f"ID: {sample.id}")
        print(f"Prompt: \"{sample.prompt}\"")
        print(f"Chords: {sample.chords}")
        print(f"Pattern: {sample.strum_pattern}")
        print(f"Tempo: {sample.tempo} BPM")
        print(f"Genre: {sample.genre} | Emotion: {sample.emotion}")
        print(f"Key: {sample.key} {sample.mode}")
    
    print("\n" + "=" * 60)
    print("✅ COMPLETE! Dataset ready in:", args.output)
    print("=" * 60)


if __name__ == "__main__":
    main()
