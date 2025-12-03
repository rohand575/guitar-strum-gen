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

# Will import main functions when implemented:
# from src.rules.generate_rule_based import generate_rule_based
