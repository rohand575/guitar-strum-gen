"""
Models Package
==============
Guitar Strum Generator - Thesis Project

This package contains:
- prompt_features.py: Data structures for extracted features
- prompt_parser.py: NLP module for parsing natural language prompts
- (future) tokenizer.py: Tokenization for neural sequence model
- (future) seq_model.py: Neural sequence generator
"""

from .prompt_features import (
    PromptFeatures,
    ExtractionConfidence,
    ConfidenceLevel,
    DEFAULT_FEATURES,
    VALID_GENRES,
    VALID_EMOTIONS,
    VALID_MODES,
    VALID_KEY_ROOTS
)

from .prompt_parser import (
    PromptParser,
    parse_prompt
)

__all__ = [
    # Data structures
    'PromptFeatures',
    'ExtractionConfidence', 
    'ConfidenceLevel',
    'DEFAULT_FEATURES',
    'VALID_GENRES',
    'VALID_EMOTIONS',
    'VALID_MODES',
    'VALID_KEY_ROOTS',
    # Parser
    'PromptParser',
    'parse_prompt'
]
