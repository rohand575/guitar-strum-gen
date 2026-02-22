"""
Rules Subpackage

This package contains the rule-based baseline system:
    - harmony.py: Music theory rules for chord progressions
                  (key signatures, chord functions, common progressions)
    - strumming.py: Pattern templates by genre/style
    - prompt_parser.py: Keyword extraction from prompts (rule-based)
    - generate_rule_based.py: Main generator combining all rules

The rule-based system serves two purposes:
    1. BASELINE: Compare neural model performance against rules
    2. FALLBACK: When neural model produces invalid output, use rules

This approach is called a "hybrid system" in the thesis.
"""

"""
Rules Subpackage - Rule-based baseline system

Usage:
    from src.rules import generate_rule_based
    
    result = generate_rule_based("upbeat folk in G major")
    print(result.chords)        # ['G', 'C', 'D', 'G']
    print(result.strum_pattern) # 'D_DU_DU_'
"""

from src.rules.generate_rule_based import generate_rule_based, format_as_chord_sheet
from src.rules.harmony import select_progression, get_diatonic_chords, validate_progression
from src.rules.strumming import select_strumming_pattern
from src.rules.prompt_parser import parse_prompt, ParsedFeatures

"""
INPUT:  "Give me a nostalgic country song in D major, about 100 BPM"
                              │
                              ▼
                    ┌─────────────────┐
                    │  prompt_parser  │ → key=D, mode=major, 
                    └─────────────────┘   genre=country, tempo=100
                              │
                              ▼
                    ┌─────────────────┐
                    │    harmony      │ → chords=['D', 'A', 'G', 'D']
                    └─────────────────┘   (I-V-IV-I progression)
                              │
                              ▼
                    ┌─────────────────┐
                    │   strumming     │ → pattern='D_DU_DU_'
                    └─────────────────┘
                              │
                              ▼
OUTPUT: Complete GuitarSample ready for your thesis!
"""