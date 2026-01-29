"""
Hybrid Generator - Neural Model + Rule-Based Fallback
======================================================

This is the MAIN entry point for generating guitar chord progressions
and strumming patterns from natural language prompts.

The Hybrid Approach:
    1. Parse the user's prompt to extract musical features
    2. Try neural model generation (LSTM trained in Chat 7)
    3. Validate the neural output for musical correctness
    4. If invalid → fallback to rule-based generation
    5. Return a validated GuitarSample

Why Hybrid?
    - Neural models are creative but can produce invalid outputs
    - Rule-based systems are reliable but repetitive
    - Combining both gives us creativity WITH safety

Usage:
    from src.app.generate import generate_guitar_part
    
    # Simple usage
    result = generate_guitar_part("mellow acoustic in D major")
    print(result.chords)        # ['D', 'A', 'Bm', 'G']
    print(result.strum_pattern) # 'D_DU_DU_'
    
    # With verbose output
    result = generate_guitar_part("sad folk ballad", verbose=True)

Author: Rohan Rajendra Dhanawade
Project: Master's Thesis - SRH Berlin University of Applied Sciences
Chat: 8 - Hybrid Generator Implementation
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import re

# =============================================================================
# OPTIONAL IMPORTS (PyTorch may not be installed)
# =============================================================================
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not installed. Neural generation disabled.")
    print("   Only rule-based generation will be available.")
    print("   To enable neural generation: pip install torch")


# =============================================================================
# PART 1: VALIDATION CONSTANTS
# =============================================================================

# All valid chords that our system recognizes
# This matches the vocabulary from tokenizer.py (29 chords)
VALID_CHORDS = {
    # Natural major chords
    "C", "D", "E", "F", "G", "A", "B",
    # Sharp major chords
    "A#", "C#", "D#", "G#",
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

# Required pattern length (8 slots for 4/4 time)
REQUIRED_PATTERN_LENGTH = 8

# Chromatic scale for key validation
CHROMATIC_SCALE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Enharmonic equivalents (flats → sharps)
FLAT_TO_SHARP = {
    "Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"
}


# =============================================================================
# PART 2: VALIDATION RESULT DATA CLASS
# =============================================================================

@dataclass
class ValidationResult:
    """
    Container for validation results with detailed error information.
    
    This helps us understand exactly WHY something failed validation,
    which is useful for debugging and logging.
    
    Attributes:
        is_valid: True if all checks passed, False otherwise
        errors: List of specific validation error messages
        warnings: List of non-critical issues (output still usable)
        
    Example:
        result = validate_output(sample)
        if not result.is_valid:
            print(f"Validation failed: {result.errors}")
    """
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    def __str__(self) -> str:
        if self.is_valid:
            status = "✅ VALID"
        else:
            status = "❌ INVALID"
        
        lines = [status]
        
        if self.errors:
            lines.append("  Errors:")
            for err in self.errors:
                lines.append(f"    - {err}")
        
        if self.warnings:
            lines.append("  Warnings:")
            for warn in self.warnings:
                lines.append(f"    - {warn}")
        
        return "\n".join(lines)


# =============================================================================
# PART 3: VALIDATION FUNCTIONS
# =============================================================================

def validate_chord(chord: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a single chord is valid.
    
    Args:
        chord: A chord string like "G", "Am", "D7"
        
    Returns:
        Tuple of (is_valid, error_message)
        If valid, error_message is None
        
    Examples:
        >>> validate_chord("G")
        (True, None)
        >>> validate_chord("Xm")
        (False, "Unknown chord: 'Xm'")
        >>> validate_chord("")
        (False, "Empty chord string")
    """
    # Check for empty string
    if not chord or not chord.strip():
        return (False, "Empty chord string")
    
    # Check if chord is in our valid set
    if chord in VALID_CHORDS:
        return (True, None)
    
    # Not found - provide helpful error
    return (False, f"Unknown chord: '{chord}'")


def validate_chord_list(chords: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate a list of chords.
    
    Args:
        chords: List of chord strings
        
    Returns:
        Tuple of (all_valid, list_of_errors)
        
    Example:
        >>> validate_chord_list(["G", "D", "Em", "C"])
        (True, [])
        >>> validate_chord_list(["G", "Xm", "Z7"])
        (False, ["Unknown chord: 'Xm'", "Unknown chord: 'Z7'"])
    """
    errors = []
    
    # Check if list is empty
    if not chords:
        return (False, ["Empty chord list - need at least 1 chord"])
    
    # Check each chord
    for chord in chords:
        is_valid, error = validate_chord(chord)
        if not is_valid:
            errors.append(error)
    
    return (len(errors) == 0, errors)


def validate_strum_pattern(pattern: str) -> Tuple[bool, List[str]]:
    """
    Validate a strumming pattern string.
    
    Rules:
        1. Must be exactly 8 characters long
        2. Each character must be D, U, or _
        3. Cannot be all rests (________)
    
    Args:
        pattern: Strumming pattern like "D_DU_DU_"
        
    Returns:
        Tuple of (is_valid, list_of_errors)
        
    Examples:
        >>> validate_strum_pattern("D_DU_DU_")
        (True, [])
        >>> validate_strum_pattern("D_DU_U")
        (False, ["Pattern length is 6, expected 8"])
        >>> validate_strum_pattern("D_DX_DU_")
        (False, ["Invalid character 'X' at position 3"])
    """
    errors = []
    
    # Check if empty
    if not pattern:
        return (False, ["Empty strumming pattern"])
    
    # Check length
    if len(pattern) != REQUIRED_PATTERN_LENGTH:
        errors.append(
            f"Pattern length is {len(pattern)}, expected {REQUIRED_PATTERN_LENGTH}"
        )
    
    # Check each character
    invalid_chars = []
    for i, char in enumerate(pattern):
        if char not in VALID_STRUM_CHARS:
            invalid_chars.append((i, char))
    
    if invalid_chars:
        for pos, char in invalid_chars:
            errors.append(f"Invalid character '{char}' at position {pos}")
    
    # Check if all rests (no actual strumming)
    if pattern and all(c == "_" for c in pattern):
        errors.append("Pattern is all rests - need at least one D or U")
    
    return (len(errors) == 0, errors)


def fix_strum_pattern(pattern: str) -> Tuple[str, bool, str]:
    """
    Attempt to fix a strumming pattern that has minor issues.
    
    This function tries to salvage neural output that is close to valid:
    - Too short → Pad with rests at the end
    - Too long → Truncate to 8 characters
    - Invalid characters → Replace with rest (_)
    
    Args:
        pattern: The strumming pattern to fix
        
    Returns:
        Tuple of (fixed_pattern, was_fixed, fix_description)
        - fixed_pattern: The corrected pattern (or original if unfixable)
        - was_fixed: True if any changes were made
        - fix_description: Human-readable description of what was fixed
        
    Examples:
        >>> fix_strum_pattern("D___DU_")
        ('D___DU__', True, 'Padded from 7 to 8 characters')
        
        >>> fix_strum_pattern("D_D_D_")
        ('D_D_D___', True, 'Padded from 6 to 8 characters')
        
        >>> fix_strum_pattern("D_DU_DU_")
        ('D_DU_DU_', False, 'No fix needed')
        
        >>> fix_strum_pattern("D_DX_DU_")
        ('D_D__DU_', True, "Replaced invalid character 'X' with '_'")
    """
    if not pattern:
        return ("________", True, "Empty pattern replaced with all rests")
    
    fixes_applied = []
    fixed = pattern
    
    # ─────────────────────────────────────────────────────────────────────────
    # Fix 1: Replace invalid characters with rests
    # ─────────────────────────────────────────────────────────────────────────
    cleaned_chars = []
    invalid_found = []
    
    for char in fixed:
        if char in VALID_STRUM_CHARS:
            cleaned_chars.append(char)
        else:
            cleaned_chars.append("_")  # Replace invalid with rest
            invalid_found.append(char)
    
    fixed = "".join(cleaned_chars)
    
    if invalid_found:
        unique_invalid = list(set(invalid_found))
        fixes_applied.append(f"Replaced invalid characters {unique_invalid} with '_'")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Fix 2: Handle length issues
    # ─────────────────────────────────────────────────────────────────────────
    original_len = len(fixed)
    
    if original_len < REQUIRED_PATTERN_LENGTH:
        # Too short - pad with rests at the end
        padding_needed = REQUIRED_PATTERN_LENGTH - original_len
        fixed = fixed + ("_" * padding_needed)
        fixes_applied.append(f"Padded from {original_len} to {REQUIRED_PATTERN_LENGTH} characters")
    
    elif original_len > REQUIRED_PATTERN_LENGTH:
        # Too long - truncate (keep the beginning)
        fixed = fixed[:REQUIRED_PATTERN_LENGTH]
        fixes_applied.append(f"Truncated from {original_len} to {REQUIRED_PATTERN_LENGTH} characters")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Check if result is all rests (not useful)
    # ─────────────────────────────────────────────────────────────────────────
    if all(c == "_" for c in fixed):
        # Try to add at least one downstroke on beat 1
        fixed = "D" + fixed[1:]
        fixes_applied.append("Added downstroke on beat 1 (was all rests)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Build result
    # ─────────────────────────────────────────────────────────────────────────
    was_fixed = len(fixes_applied) > 0
    
    if was_fixed:
        fix_description = "; ".join(fixes_applied)
    else:
        fix_description = "No fix needed"
    
    return (fixed, was_fixed, fix_description)


def validate_chords_in_key(
    chords: List[str], 
    key: str, 
    mode: str
) -> Tuple[bool, List[str]]:
    """
    Check if chords belong to the specified key (optional, for warnings).
    
    This is a "soft" validation - we warn but don't reject, because:
    1. Musicians often use borrowed chords (from parallel keys)
    2. Jazz and blues use chromatic approach chords
    3. Pop music commonly uses "wrong" chords that sound good
    
    Args:
        chords: List of chord strings
        key: Root note (e.g., "G", "A")
        mode: "major" or "minor"
        
    Returns:
        Tuple of (all_in_key, list_of_out_of_key_chords)
        
    Example:
        >>> validate_chords_in_key(["G", "D", "Em", "C"], "G", "major")
        (True, [])
        >>> validate_chords_in_key(["G", "F#", "Em", "C"], "G", "major")
        (False, ["F# is not diatonic to G major"])
    """
    warnings = []
    
    # Get diatonic chords for this key
    diatonic = _get_diatonic_chords(key, mode)
    
    # Check each chord
    for chord in chords:
        if chord not in diatonic:
            warnings.append(f"'{chord}' is not diatonic to {key} {mode}")
    
    return (len(warnings) == 0, warnings)


def _get_diatonic_chords(key: str, mode: str) -> List[str]:
    """
    Get the 7 diatonic chords for a given key.
    
    This is a simplified version - see harmony.py for full implementation.
    We include this here to avoid import cycles.
    
    Args:
        key: Root note (e.g., "G", "C", "Am")
        mode: "major" or "minor"
        
    Returns:
        List of 7 diatonic chord symbols
    """
    # Handle key notation (e.g., "Am" should extract "A" and set mode to minor)
    if key.endswith("m") and len(key) > 1 and key[-2] != "#":
        # Key like "Am", "Em" - extract root
        root = key[:-1]
        mode = "minor"
    else:
        root = key
    
    # Normalize root note (handle flats)
    if root in FLAT_TO_SHARP:
        root = FLAT_TO_SHARP[root]
    
    # Find root index in chromatic scale
    if root not in CHROMATIC_SCALE:
        # Unknown root - return empty to skip validation
        return []
    
    root_idx = CHROMATIC_SCALE.index(root)
    
    # Scale intervals
    if mode == "minor":
        intervals = [0, 2, 3, 5, 7, 8, 10]  # Natural minor
        qualities = ["m", "dim", "", "m", "m", "", ""]
    else:  # major
        intervals = [0, 2, 4, 5, 7, 9, 11]  # Major scale
        qualities = ["", "m", "m", "", "", "m", "dim"]
    
    # Build diatonic chords
    chords = []
    for interval, quality in zip(intervals, qualities):
        note_idx = (root_idx + interval) % 12
        note = CHROMATIC_SCALE[note_idx]
        chords.append(note + quality)
    
    return chords


def validate_output(
    chords: List[str],
    strum_pattern: str,
    key: Optional[str] = None,
    mode: Optional[str] = None
) -> ValidationResult:
    """
    Perform complete validation of generated output.
    
    This is the main validation function that checks everything:
    1. Are all chords valid?
    2. Is the strumming pattern valid?
    3. (Optional) Are chords in key? (generates warnings only)
    
    Args:
        chords: List of chord strings
        strum_pattern: 8-character strumming pattern
        key: Optional key for in-key validation
        mode: Optional mode (major/minor) for in-key validation
        
    Returns:
        ValidationResult with is_valid, errors, and warnings
        
    Example:
        >>> result = validate_output(["G", "D", "Em"], "D_DU_DU_")
        >>> result.is_valid
        True
        
        >>> result = validate_output(["G", "Xm"], "D_DU_U")
        >>> result.is_valid
        False
        >>> result.errors
        ["Unknown chord: 'Xm'", "Pattern length is 7, expected 8"]
    """
    all_errors = []
    all_warnings = []
    
    # ─────────────────────────────────────────────────────────────────────────
    # Validation 1: Check chord list
    # ─────────────────────────────────────────────────────────────────────────
    chords_valid, chord_errors = validate_chord_list(chords)
    all_errors.extend(chord_errors)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Validation 2: Check strumming pattern
    # ─────────────────────────────────────────────────────────────────────────
    pattern_valid, pattern_errors = validate_strum_pattern(strum_pattern)
    all_errors.extend(pattern_errors)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Validation 3: Check chords in key (warnings only)
    # ─────────────────────────────────────────────────────────────────────────
    if key and mode and chords_valid:
        in_key, key_warnings = validate_chords_in_key(chords, key, mode)
        all_warnings.extend(key_warnings)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Build result
    # ─────────────────────────────────────────────────────────────────────────
    is_valid = len(all_errors) == 0
    
    return ValidationResult(
        is_valid=is_valid,
        errors=all_errors,
        warnings=all_warnings
    )


# =============================================================================
# PART 4: NEURAL GENERATOR
# =============================================================================

class NeuralGenerator:
    """
    Wrapper for loading and using the trained LSTM model.
    
    This class handles:
    1. Loading the checkpoint file
    2. Reconstructing the model architecture
    3. Running inference to generate chords and strumming patterns
    
    Usage:
        generator = NeuralGenerator("checkpoints/guitar_lstm_final.pt")
        
        features = {
            "key": "G",
            "mode": "major",
            "genre": "folk",
            "emotion": "upbeat",
            "tempo": 110
        }
        
        result = generator.generate(features)
        print(result["chords"])        # ['G', 'D', 'Em', 'C']
        print(result["strum_pattern"]) # 'D_DU_DU_'
    """
    
    def __init__(
        self, 
        checkpoint_path: str,
        device: Optional[str] = None
    ):
        """
        Load the trained LSTM model from a checkpoint.
        
        Args:
            checkpoint_path: Path to .pt checkpoint file
            device: 'cuda' or 'cpu' (auto-detected if None)
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ImportError: If PyTorch is not installed
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for neural generation. "
                "Install with: pip install torch"
            )
        
        # Check file exists
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                f"Make sure you've downloaded the trained model from Google Colab."
            )
        
        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load model architecture and weights from checkpoint."""
        print(f"Loading neural model from: {self.checkpoint_path}")
        print(f"Device: {self.device}")
        
        # Import model class (delayed import to avoid circular dependencies)
        from src.models.lstm_model import GuitarLSTM
        
        # Load checkpoint
        checkpoint = torch.load(
            self.checkpoint_path, 
            map_location=self.device
        )
        
        # Create model instance
        self.model = GuitarLSTM()
        
        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()
        
        # Store any config info
        self.config = checkpoint.get('config', {})
        
        print(f"✅ Model loaded successfully!")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def generate(
        self,
        features: Dict[str, Union[str, int]],
        temperature: float = 0.8,
        top_k: Optional[int] = 10,
        top_p: Optional[float] = None,
        max_attempts: int = 3
    ) -> Optional[Dict]:
        """
        Generate chords and strumming pattern from features.
        
        The model generates autoregressively, one token at a time,
        until it produces <EOS> or reaches max length.
        
        Args:
            features: Dictionary with musical features
                - key: "G", "Am", "C", etc.
                - mode: "major" or "minor"
                - genre: "folk", "rock", "pop", etc.
                - emotion: "upbeat", "melancholic", etc.
                - tempo: BPM (40-200)
            temperature: Sampling randomness (0.1=conservative, 1.5=wild)
            top_k: Only sample from top K most likely tokens
            top_p: Nucleus sampling threshold (alternative to top_k)
            max_attempts: Retry generation if output is too short
            
        Returns:
            Dictionary with:
                - chords: List of chord strings
                - strum_pattern: 8-character strumming pattern
                - token_ids: Raw token IDs (for debugging)
                - raw_sequence: Human-readable tokens (for debugging)
            Returns None if generation fails
        """
        for attempt in range(max_attempts):
            try:
                # Call the model's generate method
                result = self.model.generate(
                    features=features,
                    max_length=20,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=True
                )
                
                # Check if we got reasonable output
                if result["chords"] and result["strum_pattern"]:
                    return result
                    
            except Exception as e:
                print(f"  Generation attempt {attempt + 1} failed: {e}")
                continue
        
        # All attempts failed
        return None
    
    def generate_with_validation(
        self,
        features: Dict[str, Union[str, int]],
        temperature: float = 0.8,
        top_k: Optional[int] = 10,
        max_attempts: int = 3,
        auto_fix_pattern: bool = True
    ) -> Tuple[Optional[Dict], ValidationResult]:
        """
        Generate and validate output in one step.
        
        This is the method used by the hybrid generator.
        
        Args:
            features: Musical features dictionary
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            max_attempts: Number of generation attempts
            auto_fix_pattern: If True, attempt to fix invalid patterns
            
        Returns:
            Tuple of (generation_result, validation_result)
            generation_result is None if generation failed
        """
        # Try to generate
        result = self.generate(
            features=features,
            temperature=temperature,
            top_k=top_k,
            max_attempts=max_attempts
        )
        
        if result is None:
            # Generation completely failed
            return None, ValidationResult(
                is_valid=False,
                errors=["Neural generation failed after all attempts"],
                warnings=[]
            )
        
        # First validation attempt
        validation = validate_output(
            chords=result["chords"],
            strum_pattern=result["strum_pattern"],
            key=features.get("key"),
            mode=features.get("mode")
        )
        
        # ─────────────────────────────────────────────────────────────────────
        # If invalid and auto_fix is enabled, try to fix the pattern
        # ─────────────────────────────────────────────────────────────────────
        if not validation.is_valid and auto_fix_pattern:
            # Check if the issue is fixable (pattern-related errors)
            pattern_errors = [e for e in validation.errors if "Pattern" in e or "character" in e]
            chord_errors = [e for e in validation.errors if "chord" in e.lower()]
            
            # Only try to fix if there are pattern errors but no chord errors
            if pattern_errors and not chord_errors:
                # Attempt to fix the pattern
                fixed_pattern, was_fixed, fix_description = fix_strum_pattern(
                    result["strum_pattern"]
                )
                
                if was_fixed:
                    # Update result with fixed pattern
                    original_pattern = result["strum_pattern"]
                    result["strum_pattern"] = fixed_pattern
                    result["pattern_was_fixed"] = True
                    result["original_pattern"] = original_pattern
                    result["fix_description"] = fix_description
                    
                    # Re-validate with fixed pattern
                    validation = validate_output(
                        chords=result["chords"],
                        strum_pattern=fixed_pattern,
                        key=features.get("key"),
                        mode=features.get("mode")
                    )
                    
                    # Add fix info to warnings
                    if validation.is_valid:
                        validation.warnings.append(
                            f"Pattern auto-fixed: {fix_description} "
                            f"('{original_pattern}' → '{fixed_pattern}')"
                        )
        
        return result, validation


# =============================================================================
# PART 5: PROMPT FEATURE EXTRACTION
# =============================================================================

def extract_features_from_prompt(prompt: str) -> Dict[str, Union[str, int]]:
    """
    Extract musical features from a natural language prompt.
    
    This uses the rule-based parser from Chat 3. In a full system,
    you might also use the neural parser (DistilBERT) for better
    emotion/genre detection.
    
    Args:
        prompt: Natural language input like "mellow acoustic in D major"
        
    Returns:
        Dictionary with:
            - key: Root note (e.g., "D", "G")
            - mode: "major" or "minor"
            - genre: Detected genre
            - emotion: Detected emotion
            - tempo: BPM value
    """
    # Import the rule-based parser
    from src.rules.prompt_parser import parse_and_complete
    
    # Parse the prompt
    parsed = parse_and_complete(prompt)
    
    # Convert to the format expected by neural model
    features = {
        "key": parsed.key,
        "mode": parsed.mode,
        "genre": parsed.genre,
        "emotion": parsed.emotion,
        "tempo": parsed.tempo
    }
    
    return features


# =============================================================================
# PART 6: THE HYBRID GENERATOR (Main Entry Point!)
# =============================================================================

# Global neural generator instance (loaded once, reused)
_neural_generator: Optional[NeuralGenerator] = None


def get_neural_generator(
    checkpoint_path: str = "checkpoints/guitar_lstm_final.pt"
) -> Optional[NeuralGenerator]:
    """
    Get or create the neural generator (singleton pattern).
    
    This avoids reloading the model for every request.
    
    Args:
        checkpoint_path: Path to model checkpoint
        
    Returns:
        NeuralGenerator instance, or None if unavailable
    """
    global _neural_generator
    
    if _neural_generator is not None:
        return _neural_generator
    
    # Try to load
    try:
        _neural_generator = NeuralGenerator(checkpoint_path)
        return _neural_generator
    except (FileNotFoundError, ImportError) as e:
        print(f"⚠️  Neural generator unavailable: {e}")
        return None


def generate_guitar_part(
    prompt: str,
    checkpoint_path: str = "checkpoints/guitar_lstm_final.pt",
    prefer_neural: bool = True,
    temperature: float = 0.8,
    verbose: bool = False
) -> Dict:
    """
    Generate a guitar part from a natural language prompt.
    
    THIS IS THE MAIN FUNCTION YOU'LL USE!
    
    The hybrid approach:
    1. Parse the prompt to extract features
    2. If neural model available and prefer_neural=True:
       a. Generate with neural model
       b. Validate output
       c. If valid → return neural output
    3. Fall back to rule-based generation
    
    Args:
        prompt: Natural language description
            Examples:
            - "mellow acoustic song in D major"
            - "upbeat folk in G, fast tempo"
            - "sad ballad in Am"
        checkpoint_path: Path to trained LSTM checkpoint
        prefer_neural: If True, try neural first; if False, use rules only
        temperature: Neural model sampling temperature
        verbose: If True, print detailed progress
        
    Returns:
        Dictionary containing:
            - chords: List of chord strings
            - strum_pattern: 8-character pattern
            - tempo: BPM
            - key: Musical key
            - mode: "major" or "minor"
            - genre: Detected/assigned genre
            - emotion: Detected/assigned emotion
            - source: "neural" or "rule_based"
            - validation: ValidationResult object
            
    Example:
        >>> result = generate_guitar_part("mellow acoustic in D major")
        >>> result["chords"]
        ['D', 'A', 'Bm', 'G']
        >>> result["strum_pattern"]
        'D_DU_DU_'
        >>> result["source"]
        'neural'
    """
    if verbose:
        print("=" * 60)
        print("HYBRID GUITAR GENERATOR")
        print("=" * 60)
        print(f"\n📝 Input prompt: \"{prompt}\"")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Extract features from prompt
    # ─────────────────────────────────────────────────────────────────────────
    if verbose:
        print("\n--- Step 1: Parsing prompt ---")
    
    features = extract_features_from_prompt(prompt)
    
    if verbose:
        print(f"  Key: {features['key']} {features['mode']}")
        print(f"  Genre: {features['genre']}")
        print(f"  Emotion: {features['emotion']}")
        print(f"  Tempo: {features['tempo']} BPM")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Try neural generation (if available and preferred)
    # ─────────────────────────────────────────────────────────────────────────
    neural_result = None
    neural_validation = None
    
    if prefer_neural:
        if verbose:
            print("\n--- Step 2: Attempting neural generation ---")
        
        generator = get_neural_generator(checkpoint_path)
        
        if generator is not None:
            neural_result, neural_validation = generator.generate_with_validation(
                features=features,
                temperature=temperature
            )
            
            if verbose:
                if neural_result:
                    print(f"  Generated chords: {neural_result['chords']}")
                    
                    # Show original pattern if it was fixed
                    if neural_result.get("pattern_was_fixed"):
                        print(f"  Original pattern: {neural_result.get('original_pattern')} (invalid)")
                        print(f"  Fixed pattern:    {neural_result['strum_pattern']} ✨")
                        print(f"  Fix applied: {neural_result.get('fix_description')}")
                    else:
                        print(f"  Generated pattern: {neural_result['strum_pattern']}")
                    
                    print(f"  Validation: {neural_validation}")
                else:
                    print("  ❌ Neural generation failed")
            
            # Check if neural output is valid
            if neural_result and neural_validation.is_valid:
                if verbose:
                    if neural_result.get("pattern_was_fixed"):
                        print("\n✅ Using NEURAL output (auto-fixed pattern)")
                    else:
                        print("\n✅ Using NEURAL output (valid)")
                
                return {
                    "prompt": prompt,
                    "chords": neural_result["chords"],
                    "strum_pattern": neural_result["strum_pattern"],
                    "tempo": features["tempo"],
                    "key": features["key"],
                    "mode": features["mode"],
                    "genre": features["genre"],
                    "emotion": features["emotion"],
                    "source": "neural",
                    "validation": neural_validation,
                    "raw_tokens": neural_result.get("raw_sequence", []),
                    "pattern_was_fixed": neural_result.get("pattern_was_fixed", False),
                    "original_pattern": neural_result.get("original_pattern"),
                    "fix_description": neural_result.get("fix_description")
                }
            else:
                if verbose:
                    print("\n⚠️  Neural output invalid, falling back to rules...")
        else:
            if verbose:
                print("  ⚠️  Neural model not available")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Fall back to rule-based generation
    # ─────────────────────────────────────────────────────────────────────────
    if verbose:
        print("\n--- Step 3: Rule-based generation ---")
    
    # Import and use rule-based generator
    from src.rules.generate_rule_based import generate_rule_based
    
    rule_sample = generate_rule_based(prompt, verbose=False)
    
    # Validate rule-based output (should always be valid, but let's check)
    rule_validation = validate_output(
        chords=rule_sample.chords,
        strum_pattern=rule_sample.strum_pattern,
        key=rule_sample.key,
        mode=rule_sample.mode
    )
    
    if verbose:
        print(f"  Generated chords: {rule_sample.chords}")
        print(f"  Generated pattern: {rule_sample.strum_pattern}")
        print(f"  Validation: {rule_validation}")
        print("\n✅ Using RULE-BASED output")
    
    return {
        "prompt": prompt,
        "chords": rule_sample.chords,
        "strum_pattern": rule_sample.strum_pattern,
        "tempo": rule_sample.tempo,
        "key": rule_sample.key,
        "mode": rule_sample.mode,
        "genre": rule_sample.genre,
        "emotion": rule_sample.emotion,
        "source": "rule_based",
        "validation": rule_validation,
        "fallback_reason": (
            "neural_invalid" if neural_result else
            "neural_unavailable" if prefer_neural else
            "rules_preferred"
        )
    }


# =============================================================================
# PART 7: OUTPUT FORMATTING
# =============================================================================

def format_result_as_chord_sheet(result: Dict) -> str:
    """
    Format generation result as a nice ASCII chord sheet.
    
    Args:
        result: Dictionary from generate_guitar_part()
        
    Returns:
        Formatted string for display
    """
    lines = []
    lines.append("╔" + "═" * 55 + "╗")
    lines.append("║" + " GENERATED GUITAR PART ".center(55) + "║")
    lines.append("╠" + "═" * 55 + "╣")
    
    # Prompt
    prompt = result.get("prompt", "")
    if len(prompt) > 50:
        prompt = prompt[:47] + "..."
    lines.append(f"║ Prompt: {prompt}".ljust(56) + "║")
    lines.append("╠" + "─" * 55 + "╣")
    
    # Musical info
    key_mode = f"{result.get('key', '?')} {result.get('mode', '?')}"
    lines.append(f"║ Key: {key_mode}".ljust(56) + "║")
    lines.append(f"║ Tempo: {result.get('tempo', '?')} BPM".ljust(56) + "║")
    lines.append(f"║ Genre: {result.get('genre', '?')} | Emotion: {result.get('emotion', '?')}".ljust(56) + "║")
    lines.append("╠" + "─" * 55 + "╣")
    
    # Chords
    chords = result.get("chords", [])
    chord_str = " → ".join(chords)
    lines.append(f"║ Chords: {chord_str}".ljust(56) + "║")
    lines.append("╠" + "─" * 55 + "╣")
    
    # Strumming pattern
    pattern = result.get("strum_pattern", "")
    lines.append("║ Strumming Pattern:".ljust(56) + "║")
    lines.append("║   Beat:  1   &   2   &   3   &   4   &".ljust(56) + "║")
    if pattern:
        pattern_display = "   ".join(pattern)
        lines.append(f"║   Strum: {pattern_display}".ljust(56) + "║")
    lines.append("╠" + "─" * 55 + "╣")
    
    # Source
    source = result.get("source", "unknown")
    source_emoji = "🤖" if source == "neural" else "📚"
    lines.append(f"║ Generated by: {source_emoji} {source.upper()}".ljust(56) + "║")
    
    lines.append("╚" + "═" * 55 + "╝")
    
    return "\n".join(lines)


# =============================================================================
# TESTING (run this file directly to test the validator)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("HYBRID GENERATOR TESTS")
    print("=" * 70)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Part A: Validator Tests
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("PART A: VALIDATOR TESTS")
    print("─" * 70)
    
    # Test 1: Valid output
    print("\n✓ Test 1: Valid output")
    result = validate_output(["G", "D", "Em", "C"], "D_DU_DU_")
    print(f"  Chords: ['G', 'D', 'Em', 'C']")
    print(f"  Pattern: 'D_DU_DU_'")
    print(f"  {result}")
    
    # Test 2: Invalid chord
    print("\n✓ Test 2: Invalid chord")
    result = validate_output(["G", "Xm", "Em"], "D_DU_DU_")
    print(f"  Chords: ['G', 'Xm', 'Em']")
    print(f"  {result}")
    
    # Test 3: Invalid pattern length
    print("\n✓ Test 3: Invalid pattern length")
    result = validate_output(["G", "D", "Em", "C"], "D_DU_U")
    print(f"  Pattern: 'D_DU_U' (7 chars)")
    print(f"  {result}")
    
    # Test 4: Invalid pattern character
    print("\n✓ Test 4: Invalid pattern character")
    result = validate_output(["G", "D"], "D_DX_DU_")
    print(f"  Pattern: 'D_DX_DU_'")
    print(f"  {result}")
    
    # Test 5: Out-of-key warning
    print("\n✓ Test 5: Out-of-key warning (soft validation)")
    result = validate_output(["G", "A", "Em", "C"], "D_DU_DU_", key="G", mode="major")
    print(f"  Chords: ['G', 'A', 'Em', 'C'] in G major")
    print(f"  (A major is not diatonic - only Am is)")
    print(f"  {result}")
    
    # Test 6: Empty input
    print("\n✓ Test 6: Empty input")
    result = validate_output([], "")
    print(f"  {result}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Part A.5: Pattern Fixing Tests (NEW!)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("PART A.5: PATTERN FIXING TESTS")
    print("─" * 70)
    
    # Test fix 1: Too short pattern
    print("\n✓ Fix Test 1: Pattern too short (7 chars)")
    pattern = "D___DU_"
    fixed, was_fixed, desc = fix_strum_pattern(pattern)
    print(f"  Original: '{pattern}' ({len(pattern)} chars)")
    print(f"  Fixed:    '{fixed}' ({len(fixed)} chars)")
    print(f"  Was fixed: {was_fixed}")
    print(f"  Description: {desc}")
    
    # Test fix 2: Much too short
    print("\n✓ Fix Test 2: Pattern much too short (6 chars)")
    pattern = "D_D_D_"
    fixed, was_fixed, desc = fix_strum_pattern(pattern)
    print(f"  Original: '{pattern}' ({len(pattern)} chars)")
    print(f"  Fixed:    '{fixed}' ({len(fixed)} chars)")
    print(f"  Description: {desc}")
    
    # Test fix 3: Invalid character
    print("\n✓ Fix Test 3: Invalid character")
    pattern = "D_DX_DU_"
    fixed, was_fixed, desc = fix_strum_pattern(pattern)
    print(f"  Original: '{pattern}'")
    print(f"  Fixed:    '{fixed}'")
    print(f"  Description: {desc}")
    
    # Test fix 4: Already valid
    print("\n✓ Fix Test 4: Already valid (no fix needed)")
    pattern = "D_DU_DU_"
    fixed, was_fixed, desc = fix_strum_pattern(pattern)
    print(f"  Original: '{pattern}'")
    print(f"  Fixed:    '{fixed}'")
    print(f"  Was fixed: {was_fixed}")
    
    # Test fix 5: Empty pattern
    print("\n✓ Fix Test 5: Empty pattern")
    pattern = ""
    fixed, was_fixed, desc = fix_strum_pattern(pattern)
    print(f"  Original: '' (empty)")
    print(f"  Fixed:    '{fixed}'")
    print(f"  Description: {desc}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Part B: Hybrid Generator Tests (Rule-Based Only in this environment)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("PART B: HYBRID GENERATOR TESTS (with auto-fix)")
    print("─" * 70)
    print("\n(Testing neural generation with pattern auto-fix enabled)\n")
    
    test_prompts = [
        "mellow acoustic song in D major",
        "upbeat folk in G major, fast tempo",
        "sad ballad in Am",
        "energetic rock in E minor",
    ]
    
    for prompt in test_prompts:
        print(f"\n📝 Prompt: \"{prompt}\"")
        try:
            result = generate_guitar_part(
                prompt, 
                prefer_neural=True,
                verbose=True  # Show the auto-fix process
            )
            print(format_result_as_chord_sheet(result))
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
