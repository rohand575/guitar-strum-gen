"""
Evaluation Subpackage

This package contains evaluation infrastructure:
    - metrics.py: Objective metrics for generated outputs
    - experiments.py: Scripts to run evaluation experiments
    - user_study_materials.py: Generate materials for human evaluation

Objective Metrics (from thesis proposal):
    1. Chord Correctness: % of chords that belong to detected key
       - Uses music theory rules to validate chord-key relationships
       
    2. Rhythmic Consistency: Alignment with metrical grid
       - Checks if strum events fall on proper beat subdivisions
       
    3. Stylistic Diversity: Shannon entropy of pattern distributions
       - Measures variety in generated outputs (avoids repetition)

Subjective Evaluation (User Study):
    - 10 intermediate-advanced guitarists
    - Rate on 1-5 Likert scale:
        * Playability: Can you actually play this?
        * Expressiveness: Does it capture the requested mood?
        * Usefulness: Would you use this for practice/composition?
"""

# Will import when implemented:
# from src.evaluation.metrics import chord_correctness, rhythmic_consistency
