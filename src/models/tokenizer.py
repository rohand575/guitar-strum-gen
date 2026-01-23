"""
Music Tokenizer for Guitar Chord & Strumming Pattern Generation
================================================================

This module converts between human-readable musical notation and
integer token IDs that neural networks can process.

Key Concepts:
- Chords (e.g., "G", "Em", "A7") are treated as atomic units (one token each)
- Strumming characters (D, U, _) are tokenized individually
- Special tokens (<BOS>, <EOS>, <SEP>, <PAD>) control sequence structure

Example:
    tokenizer = MusicTokenizer()
    
    # Encode: musical notation → token IDs
    ids = tokenizer.encode(
        chords=["G", "D", "Em", "C"],
        strum_pattern="D_DU_DU_"
    )
    # Result: [1, 8, 5, 19, 4, 3, 33, 35, 33, 34, 35, 33, 34, 35, 2]
    
    # Decode: token IDs → musical notation
    chords, strum = tokenizer.decode(ids)
    # Result: (["G", "D", "Em", "C"], "D_DU_DU_")

Author: Rohan Rajendra Dhanawade
Project: Master's Thesis - SRH Berlin University of Applied Sciences
"""

from typing import List, Tuple, Optional, Dict

# PyTorch is optional - only needed for encode_batch with return_tensors=True
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# VOCABULARY DEFINITION
# =============================================================================
# This vocabulary is designed based on the dataset analysis from Chat 4.
# Total: 36 unique tokens

VOCABULARY: Dict[str, int] = {
    # ─────────────────────────────────────────────────────────────────────────
    # SPECIAL TOKENS (IDs 0-3)
    # These control sequence structure and are essential for training
    # ─────────────────────────────────────────────────────────────────────────
    "<PAD>": 0,   # Padding - fills sequences to equal length (ignored in loss)
    "<BOS>": 1,   # Beginning of Sequence - signals start of generation
    "<EOS>": 2,   # End of Sequence - signals model should stop generating
    "<SEP>": 3,   # Separator - boundary between chords and strumming
    
    # ─────────────────────────────────────────────────────────────────────────
    # CHORD TOKENS (IDs 4-32) — 29 unique chords from dataset
    # These appear BEFORE the <SEP> token in sequences
    # ─────────────────────────────────────────────────────────────────────────
    
    # Natural major chords (the basic open chords guitarists learn first)
    "C": 4,
    "D": 5,
    "E": 6,
    "F": 7,
    "G": 8,
    "A": 9,
    "B": 10,
    
    # Sharp major chords (barre chord territory)
    "A#": 11,
    "C#": 12,
    "D#": 13,
    "G#": 14,
    
    # Minor chords (the "sad" versions)
    "Am": 15,
    "Bm": 16,
    "Cm": 17,
    "Dm": 18,
    "Em": 19,
    "Fm": 20,
    "Gm": 21,
    "A#m": 22,
    "C#m": 23,
    "F#m": 24,
    "G#m": 25,
    
    # Seventh chords (add color/tension)
    "A7": 26,
    "B7": 27,
    "D7": 28,
    "E7": 29,
    
    # Other chord types (suspensions, diminished)
    "Asus4": 30,
    "C#dim": 31,
    "Gdim": 32,
    
    # ─────────────────────────────────────────────────────────────────────────
    # STRUMMING TOKENS (IDs 33-35) — 3 unique characters
    # These appear AFTER the <SEP> token in sequences
    # 
    # IMPORTANT: "D" here (ID=33) is DIFFERENT from chord "D" (ID=5)!
    # The model learns to distinguish them by position relative to <SEP>
    # ─────────────────────────────────────────────────────────────────────────
    "STRUM_D": 33,  # Downstroke (strum down toward floor)
    "STRUM_U": 34,  # Upstroke (strum up toward ceiling)  
    "STRUM__": 35,  # Rest/mute (no strum, hand rests on strings)
}

# Total vocabulary size
VOCAB_SIZE = 36

# Maximum sequence length (from architecture spec)
# 1 <BOS> + 8 chords (max) + 1 <SEP> + 8 strum + 1 <EOS> + 1 buffer = 20
MAX_SEQ_LENGTH = 20


# =============================================================================
# REVERSE MAPPING (for decoding)
# =============================================================================
# We need to go from ID → token for decoding model outputs

ID_TO_TOKEN: Dict[int, str] = {v: k for k, v in VOCABULARY.items()}


# =============================================================================
# TOKENIZER CLASS
# =============================================================================

class MusicTokenizer:
    """
    Tokenizer for guitar chord progressions and strumming patterns.
    
    This class handles the conversion between musical notation and the
    integer sequences that neural networks process.
    
    Attributes:
        vocab (dict): Token to ID mapping
        id_to_token (dict): ID to token mapping (reverse)
        vocab_size (int): Total number of unique tokens (36)
        max_seq_length (int): Maximum sequence length (20)
        pad_id (int): ID of padding token (0)
        bos_id (int): ID of beginning-of-sequence token (1)
        eos_id (int): ID of end-of-sequence token (2)
        sep_id (int): ID of separator token (3)
    """
    
    def __init__(self, max_seq_length: int = MAX_SEQ_LENGTH):
        """
        Initialize the tokenizer with vocabulary and special token IDs.
        
        Args:
            max_seq_length: Maximum length for padded sequences (default: 20)
        """
        # Store vocabulary mappings
        self.vocab = VOCABULARY.copy()
        self.id_to_token = ID_TO_TOKEN.copy()
        self.vocab_size = VOCAB_SIZE
        self.max_seq_length = max_seq_length
        
        # Store special token IDs for easy access
        # These are used frequently during encoding/decoding
        self.pad_id = VOCABULARY["<PAD>"]
        self.bos_id = VOCABULARY["<BOS>"]
        self.eos_id = VOCABULARY["<EOS>"]
        self.sep_id = VOCABULARY["<SEP>"]
        
        # Build set of valid chord tokens (for validation during encoding)
        # This excludes special tokens and strumming tokens
        self._chord_tokens = {
            token for token in VOCABULARY.keys()
            if not token.startswith("<") and not token.startswith("STRUM_")
        }
        
        # Build mapping from strum characters to token IDs
        self._strum_char_to_id = {
            "D": VOCABULARY["STRUM_D"],
            "U": VOCABULARY["STRUM_U"],
            "_": VOCABULARY["STRUM__"],
        }
        
        # Reverse mapping for decoding strums
        self._strum_id_to_char = {v: k for k, v in self._strum_char_to_id.items()}
    
    def encode(
        self,
        chords: List[str],
        strum_pattern: str,
        add_special_tokens: bool = True,
        pad_to_max_length: bool = False
    ) -> List[int]:
        """
        Convert chords and strumming pattern to token IDs.
        
        This is the core encoding function that transforms musical notation
        into the integer sequence format that neural networks require.
        
        Args:
            chords: List of chord names, e.g., ["G", "D", "Em", "C"]
            strum_pattern: 8-character strumming pattern, e.g., "D_DU_DU_"
            add_special_tokens: If True, wrap with <BOS>, <SEP>, <EOS>
            pad_to_max_length: If True, pad sequence to max_seq_length
            
        Returns:
            List of token IDs representing the sequence
            
        Raises:
            ValueError: If any chord is not in vocabulary
            ValueError: If strum_pattern contains invalid characters
            
        Example:
            >>> tokenizer = MusicTokenizer()
            >>> tokenizer.encode(["G", "D"], "D_DU_DU_")
            [1, 8, 5, 3, 33, 35, 33, 34, 35, 33, 34, 35, 2]
            # [<BOS>, G, D, <SEP>, D, _, D, U, _, D, U, _, <EOS>]
        """
        token_ids = []
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 1: Add <BOS> token (if requested)
        # ─────────────────────────────────────────────────────────────────────
        if add_special_tokens:
            token_ids.append(self.bos_id)
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 2: Encode chord tokens
        # Each chord is a single token (atomic unit)
        # ─────────────────────────────────────────────────────────────────────
        for chord in chords:
            if chord not in self.vocab:
                raise ValueError(
                    f"Unknown chord '{chord}'. "
                    f"Valid chords: {sorted(self._chord_tokens)}"
                )
            token_ids.append(self.vocab[chord])
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 3: Add <SEP> token (boundary between chords and strumming)
        # ─────────────────────────────────────────────────────────────────────
        if add_special_tokens:
            token_ids.append(self.sep_id)
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 4: Encode strumming pattern (character by character)
        # Each character (D, U, _) becomes a separate token
        # ─────────────────────────────────────────────────────────────────────
        for char in strum_pattern:
            if char not in self._strum_char_to_id:
                raise ValueError(
                    f"Invalid strumming character '{char}'. "
                    f"Valid characters: D, U, _"
                )
            token_ids.append(self._strum_char_to_id[char])
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 5: Add <EOS> token (if requested)
        # ─────────────────────────────────────────────────────────────────────
        if add_special_tokens:
            token_ids.append(self.eos_id)
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 6: Pad to max length (if requested)
        # ─────────────────────────────────────────────────────────────────────
        if pad_to_max_length:
            token_ids = self.pad_sequence(token_ids)
        
        return token_ids
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True
    ) -> Tuple[List[str], str]:
        """
        Convert token IDs back to chords and strumming pattern.
        
        This reverses the encoding process, useful for interpreting
        model outputs during inference.
        
        Args:
            token_ids: List of token IDs from model output
            skip_special_tokens: If True, ignore <PAD>, <BOS>, <EOS>, <SEP>
            
        Returns:
            Tuple of (chords_list, strum_pattern_string)
            
        Example:
            >>> tokenizer = MusicTokenizer()
            >>> tokenizer.decode([1, 8, 5, 3, 33, 35, 33, 34, 2])
            (['G', 'D'], 'D_DU')
        """
        chords = []
        strum_chars = []
        
        # Track whether we've seen <SEP> to know if we're in chord or strum territory
        seen_sep = False
        
        for token_id in token_ids:
            # Skip padding tokens
            if token_id == self.pad_id:
                continue
                
            # Handle special tokens
            if token_id == self.bos_id:
                if skip_special_tokens:
                    continue
            elif token_id == self.eos_id:
                if skip_special_tokens:
                    break  # Stop at end of sequence
            elif token_id == self.sep_id:
                seen_sep = True
                continue  # Don't add <SEP> to output
            
            # Get the token string
            if token_id not in self.id_to_token:
                raise ValueError(f"Unknown token ID: {token_id}")
            
            token = self.id_to_token[token_id]
            
            # Categorize based on position (before or after <SEP>)
            if not seen_sep:
                # Before <SEP> → chord token
                chords.append(token)
            else:
                # After <SEP> → strumming token
                if token in self._strum_id_to_char:
                    # It's a strum ID, convert to character
                    strum_chars.append(self._strum_id_to_char[token_id])
                elif token.startswith("STRUM_"):
                    # It's already the STRUM_X format, extract the character
                    strum_chars.append(token[-1])  # "STRUM_D" → "D"
        
        # Join strum characters into a single string
        strum_pattern = "".join(strum_chars)
        
        return chords, strum_pattern
    
    def pad_sequence(
        self,
        token_ids: List[int],
        max_length: Optional[int] = None,
        padding_side: str = "right"
    ) -> List[int]:
        """
        Pad a sequence to a fixed length with <PAD> tokens.
        
        Neural networks require fixed-size inputs, so shorter sequences
        must be padded. We use <PAD> (ID=0) which is ignored during
        loss computation.
        
        Args:
            token_ids: Sequence to pad
            max_length: Target length (default: self.max_seq_length)
            padding_side: "right" (default) or "left"
            
        Returns:
            Padded sequence of exactly max_length tokens
            
        Example:
            >>> tokenizer = MusicTokenizer()
            >>> tokenizer.pad_sequence([1, 8, 5, 2], max_length=8)
            [1, 8, 5, 2, 0, 0, 0, 0]
        """
        if max_length is None:
            max_length = self.max_seq_length
        
        current_length = len(token_ids)
        
        # If already at or over max length, truncate
        if current_length >= max_length:
            return token_ids[:max_length]
        
        # Calculate padding needed
        padding_needed = max_length - current_length
        padding = [self.pad_id] * padding_needed
        
        # Add padding on specified side
        if padding_side == "right":
            return token_ids + padding
        else:  # left padding
            return padding + token_ids
    
    def create_attention_mask(self, token_ids: List[int]) -> List[int]:
        """
        Create an attention mask for the sequence.
        
        The attention mask tells the model which tokens are real (1)
        and which are padding (0). This prevents the model from
        "attending to" meaningless padding tokens.
        
        Args:
            token_ids: Padded sequence of token IDs
            
        Returns:
            List of 1s (real tokens) and 0s (padding)
            
        Example:
            >>> tokenizer = MusicTokenizer()
            >>> tokenizer.create_attention_mask([1, 8, 5, 2, 0, 0, 0, 0])
            [1, 1, 1, 1, 0, 0, 0, 0]
        """
        return [1 if token_id != self.pad_id else 0 for token_id in token_ids]
    
    def get_vocab_size(self) -> int:
        """Return the vocabulary size (36)."""
        return self.vocab_size
    
    def get_chord_tokens(self) -> set:
        """Return set of valid chord token strings."""
        return self._chord_tokens.copy()
    
    def encode_batch(
        self,
        batch_chords: List[List[str]],
        batch_strums: List[str],
        return_tensors: bool = True
    ) -> dict:
        """
        Encode a batch of chord/strum pairs for training.
        
        This is a convenience method for processing multiple samples
        at once, returning padded tensors ready for PyTorch.
        
        Args:
            batch_chords: List of chord lists
            batch_strums: List of strum pattern strings
            return_tensors: If True, return PyTorch tensors
            
        Returns:
            Dictionary with:
                - "input_ids": Padded token sequences
                - "attention_mask": Mask for real vs padding tokens
                
        Example:
            >>> tokenizer = MusicTokenizer()
            >>> result = tokenizer.encode_batch(
            ...     batch_chords=[["G", "D"], ["C", "Am", "F", "G"]],
            ...     batch_strums=["D_DU_DU_", "D_DUDU__"]
            ... )
        """
        input_ids = []
        attention_masks = []
        
        for chords, strum in zip(batch_chords, batch_strums):
            # Encode single sample
            ids = self.encode(
                chords=chords,
                strum_pattern=strum,
                add_special_tokens=True,
                pad_to_max_length=True
            )
            mask = self.create_attention_mask(ids)
            
            input_ids.append(ids)
            attention_masks.append(mask)
        
        # Convert to tensors if requested
        if return_tensors:
            if not TORCH_AVAILABLE:
                raise ImportError(
                    "PyTorch is required for return_tensors=True. "
                    "Install with: pip install torch"
                )
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.long)
            }
        else:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_masks
            }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"MusicTokenizer("
            f"vocab_size={self.vocab_size}, "
            f"max_seq_length={self.max_seq_length})"
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_default_tokenizer() -> MusicTokenizer:
    """Create and return a default tokenizer instance."""
    return MusicTokenizer()


# =============================================================================
# TESTING / DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    """
    Run this file directly to see the tokenizer in action:
        python -m src.models.tokenizer
    """
    print("=" * 70)
    print("MUSIC TOKENIZER DEMONSTRATION")
    print("=" * 70)
    
    # Create tokenizer
    tokenizer = MusicTokenizer()
    print(f"\n{tokenizer}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Example 1: Basic encoding
    print("\n" + "-" * 70)
    print("EXAMPLE 1: Basic Encoding")
    print("-" * 70)
    
    chords = ["G", "D", "Em", "C"]
    strum = "D_DU_DU_"
    
    print(f"Input chords: {chords}")
    print(f"Input strum:  {strum}")
    
    token_ids = tokenizer.encode(chords, strum)
    print(f"\nEncoded IDs: {token_ids}")
    print(f"Length: {len(token_ids)} tokens")
    
    # Show what each ID means
    print("\nToken breakdown:")
    for i, tid in enumerate(token_ids):
        token = tokenizer.id_to_token[tid]
        print(f"  Position {i:2d}: ID={tid:2d} → '{token}'")
    
    # Example 2: Decoding
    print("\n" + "-" * 70)
    print("EXAMPLE 2: Decoding back to musical notation")
    print("-" * 70)
    
    decoded_chords, decoded_strum = tokenizer.decode(token_ids)
    print(f"Decoded chords: {decoded_chords}")
    print(f"Decoded strum:  {decoded_strum}")
    print(f"Round-trip successful: {chords == decoded_chords and strum == decoded_strum}")
    
    # Example 3: Padding
    print("\n" + "-" * 70)
    print("EXAMPLE 3: Padding to max length")
    print("-" * 70)
    
    padded = tokenizer.encode(chords, strum, pad_to_max_length=True)
    print(f"Padded sequence ({len(padded)} tokens): {padded}")
    
    mask = tokenizer.create_attention_mask(padded)
    print(f"Attention mask: {mask}")
    
    # Example 4: Batch encoding
    print("\n" + "-" * 70)
    print("EXAMPLE 4: Batch encoding for training")
    print("-" * 70)
    
    batch_chords = [
        ["G", "D", "Em", "C"],
        ["Am", "F", "C", "G"],
        ["E", "A", "B"]
    ]
    batch_strums = ["D_DU_DU_", "D___D___", "DDUU_DUU"]
    
    batch_result = tokenizer.encode_batch(batch_chords, batch_strums, return_tensors=False)
    print(f"Batch size: {len(batch_result['input_ids'])} samples")
    print(f"Each sample length: {len(batch_result['input_ids'][0])} tokens")
    
    if TORCH_AVAILABLE:
        batch_tensor = tokenizer.encode_batch(batch_chords, batch_strums, return_tensors=True)
        print(f"Tensor shape: {batch_tensor['input_ids'].shape}")
    else:
        print("(PyTorch not available - tensor conversion skipped)")
    
    print("\n" + "=" * 70)
    print("TOKENIZER DEMONSTRATION COMPLETE ✓")
    print("=" * 70)
