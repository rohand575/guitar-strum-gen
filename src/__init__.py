"""
Guitar Strum Generator - Source Package

A conversational AI system for generating symbolic guitar chord progressions
and strumming patterns from natural language prompts.

Subpackages:
    - src.data: Data schemas, loaders, and preprocessing
    - src.rules: Rule-based baseline system
    - src.models: Neural models, tokenizers, prompt parsers
    - src.train: Training scripts and utilities
    - src.evaluation: Metrics and evaluation experiments
    - src.app: User interface and API

Example usage:
    from src.app.generate import generate_guitar_part
    
    result = generate_guitar_part("upbeat folk in G major")
    print(result.chords)        # ['G', 'D', 'Em', 'C']
    print(result.strum_pattern) # 'D_DU_DU_'
"""

__version__ = "0.1.0"
__author__ = "Rohan Rajendra Dhanawade"
