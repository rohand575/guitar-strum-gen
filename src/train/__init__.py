"""
Train Subpackage

This package contains training infrastructure:
    - train_seq_model.py: Main training loop for the Transformer
    - utils.py: Helper functions (learning rate scheduling, checkpointing)
    - config.py: Training hyperparameters

Training will be done in Google Colab (free GPU), but the actual
training code lives here so it can be imported into notebooks.

Typical training workflow:
    1. Load dataset from /data/processed/dataset.jsonl
    2. Initialize tokenizer and model
    3. Run training loop with early stopping
    4. Save best checkpoint to /checkpoints/
    5. Evaluate on held-out test set

Key hyperparameters (from thesis proposal):
    - Optimizer: Adam, lr=5e-4
    - Batch size: 16
    - Epochs: 60 with early stopping
    - Loss: Cross-entropy on joint chord-strum tokens
"""

# Will import when implemented:
# from src.train.train_seq_model import train
