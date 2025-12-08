"""
Data Subpackage

This package handles everything related to data:
    - schema.py: Pydantic models defining our data structure
    - loader.py: Functions to load and save datasets
    - build_dataset.py: Scripts to construct the training dataset

The core data structure is the GuitarSample, which contains:
    - prompt: Natural language description
    - chords: List of chord symbols
    - strum_pattern: String like "D_DU_UD_"
    - tempo: BPM (beats per minute)
    - metadata: genre, emotion, key, etc.
"""

# When schema.py is created, we'll import the main class here:
from src.data.schema import GuitarSample
