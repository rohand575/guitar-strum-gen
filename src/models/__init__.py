"""
Models Subpackage

This package contains the neural network components:
    - prompt_parser.py: NLP-based feature extraction (DistilBERT)
    - prompt_features.py: Feature dataclass for extracted features
    - tokenizer.py: Custom tokenizer for chord-strum sequences
    - seq_model.py: Transformer encoder-decoder for generation

The neural model architecture:
    1. Encode the prompt → extract features (key, emotion, style, tempo)
    2. Condition the sequence model on these features
    3. Generate tokens: [CHORD_Am, STRUM_D, STRUM_U, ...]
    4. Decode tokens back to human-readable format

Key design decisions:
    - Vocabulary ~100 tokens (50 chords + strum tokens + special tokens)
    - 4-layer Transformer with 256-dim embeddings
    - Cross-attention to prompt features
"""

# Will import main classes when implemented:
# from src.models.tokenizer import GuitarTokenizer
# from src.models.seq_model import GuitarTransformer
