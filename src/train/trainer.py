"""
Training Loop for Guitar Pattern Generation
============================================

This module implements the complete training pipeline for the GuitarLSTM
model, including:
- Training loop with teacher forcing
- Validation evaluation
- Early stopping to prevent overfitting
- Checkpoint saving and loading
- Learning rate scheduling
- Comprehensive logging

The trainer handles all the complexity of training so you can focus on
experimenting with hyperparameters and evaluating results.

Example:
    from src.train.trainer import GuitarTrainer, create_trainer
    from src.train.dataset import create_dataloaders
    
    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(
        train_path="data/train.jsonl",
        val_path="data/val.jsonl",
        batch_size=16
    )
    
    # Create and run trainer
    trainer = create_trainer(train_loader, val_loader)
    history = trainer.train(num_epochs=100)

Author: Rohan Rajendra Dhanawade
Project: Master's Thesis - SRH Berlin University of Applied Sciences
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    raise ImportError("PyTorch is required for training. Install with: pip install torch")

# Import our modules
from src.models.lstm_model import GuitarLSTM, LSTM_CONFIG
from src.models.tokenizer import VOCAB_SIZE


# =============================================================================
# CONFIGURATION
# =============================================================================

TRAINING_CONFIG = {
    # Data
    "batch_size": 16,
    
    # Optimization
    "optimizer": "AdamW",
    "learning_rate": 5e-4,
    "weight_decay": 0.01,
    "max_epochs": 100,
    
    # Learning rate schedule
    "scheduler": "cosine",  # "cosine", "plateau", or None
    "warmup_epochs": 5,
    "min_lr": 1e-6,
    
    # Early stopping
    "patience": 15,  # Stop if no improvement for this many epochs
    "min_delta": 0.001,  # Minimum improvement to count as "better"
    
    # Loss
    "label_smoothing": 0.1,  # Helps generalization
    
    # Regularization
    "gradient_clip": 1.0,  # Clip gradients to prevent explosion
    
    # Checkpointing
    "save_every": 10,  # Save checkpoint every N epochs
    "checkpoint_dir": "checkpoints",
    
    # Logging
    "log_every": 1,  # Log metrics every N batches
    "verbose": True,
}


# =============================================================================
# TRAINING HISTORY DATACLASS
# =============================================================================

@dataclass
class TrainingHistory:
    """Stores training metrics for analysis and plotting."""
    
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    train_accuracies: List[float] = field(default_factory=list)
    val_accuracies: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    
    best_epoch: int = 0
    best_val_loss: float = float('inf')
    total_training_time: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def save(self, path: Union[str, Path]):
        """Save history to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TrainingHistory':
        """Load history from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


# =============================================================================
# TRAINER CLASS
# =============================================================================

class GuitarTrainer:
    """
    Complete training pipeline for the GuitarLSTM model.
    
    This trainer handles:
    - Forward/backward passes with teacher forcing
    - Validation evaluation
    - Early stopping based on validation loss
    - Learning rate scheduling
    - Checkpoint saving/loading
    - Comprehensive logging
    
    Attributes:
        model: GuitarLSTM model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: AdamW optimizer
        scheduler: Learning rate scheduler
        config: Training configuration dictionary
        history: TrainingHistory tracking metrics
        device: Device to train on (cpu/cuda)
        
    Example:
        >>> trainer = GuitarTrainer(model, train_loader, val_loader)
        >>> history = trainer.train(num_epochs=100)
        >>> print(f"Best validation loss: {history.best_val_loss:.4f}")
    """
    
    def __init__(
        self,
        model: GuitarLSTM,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[Dict] = None,
        device: Optional[str] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: GuitarLSTM model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            config: Training configuration (uses TRAINING_CONFIG if None)
            device: Device to train on ("cpu", "cuda", or None for auto-detect)
            checkpoint_dir: Directory for saving checkpoints
        """
        self.config = config if config else TRAINING_CONFIG.copy()
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Store dataloaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"]
        )
        
        # Setup learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup loss function with label smoothing
        # ignore_index=0 means we don't compute loss on padding tokens
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=0,  # Padding token ID
            label_smoothing=self.config["label_smoothing"]
        )
        
        # Setup checkpoint directory
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
        else:
            self.checkpoint_dir = Path(self.config["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize history
        self.history = TrainingHistory()
        
        # Early stopping state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        
        # Print setup info
        if self.config["verbose"]:
            self._print_setup_info()
    
    def _create_scheduler(self):
        """Create the learning rate scheduler."""
        scheduler_type = self.config.get("scheduler", "cosine")
        
        if scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config["max_epochs"],
                eta_min=self.config["min_lr"]
            )
        elif scheduler_type == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=self.config["min_lr"]
            )
        else:
            return None
    
    def _print_setup_info(self):
        """Print training setup information."""
        print("=" * 70)
        print("TRAINING SETUP")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print(f"Max epochs: {self.config['max_epochs']}")
        print(f"Early stopping patience: {self.config['patience']}")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        print("=" * 70)
    
    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
        """
        Move batch to device and prepare inputs/targets.
        
        For teacher forcing, we use:
        - Input: tokens [0:N-1] (everything except last)
        - Target: tokens [1:N] (everything except first)
        
        This way, the model learns to predict the next token at each position.
        
        Args:
            batch: Dictionary from DataLoader
            
        Returns:
            Tuple of (features_dict, input_ids, target_ids)
        """
        # Move all tensors to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        
        # Prepare feature indices
        features = {
            "key": batch["key_idx"].to(self.device),
            "mode": batch["mode_idx"].to(self.device),
            "genre": batch["genre_idx"].to(self.device),
            "emotion": batch["emotion_idx"].to(self.device),
            "tempo": batch["tempo_idx"].to(self.device),
        }
        
        # For teacher forcing:
        # Input:  [<BOS>, G, D, Em, C, <SEP>, D, _, D, U, ...]
        # Target: [G, D, Em, C, <SEP>, D, _, D, U, ..., <EOS>]
        # 
        # We shift by 1: input is all but last, target is all but first
        model_input = input_ids[:, :-1]  # [batch, seq_len-1]
        target = input_ids[:, 1:]        # [batch, seq_len-1]
        
        return features, model_input, target
    
    def _compute_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss with padding masking.
        
        Args:
            logits: Model predictions [batch, seq_len, vocab_size]
            target: Target token IDs [batch, seq_len]
            
        Returns:
            Scalar loss tensor
        """
        # Reshape for cross-entropy: [batch * seq_len, vocab_size]
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        target_flat = target.reshape(-1)
        
        # Compute loss (padding tokens are ignored via ignore_index=0)
        loss = self.criterion(logits_flat, target_flat)
        
        return loss
    
    def _compute_accuracy(
        self,
        logits: torch.Tensor,
        target: torch.Tensor
    ) -> float:
        """
        Compute token-level accuracy (excluding padding).
        
        Args:
            logits: Model predictions [batch, seq_len, vocab_size]
            target: Target token IDs [batch, seq_len]
            
        Returns:
            Accuracy as a float between 0 and 1
        """
        # Get predictions
        predictions = logits.argmax(dim=-1)  # [batch, seq_len]
        
        # Create mask for non-padding tokens
        mask = target != 0  # Padding token ID is 0
        
        # Compute accuracy
        correct = (predictions == target) & mask
        accuracy = correct.sum().float() / mask.sum().float()
        
        return accuracy.item()
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch (one pass through training data).
        
        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        self.model.train()  # Set to training mode
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Prepare batch
            features, input_ids, target = self._prepare_batch(batch)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(input_ids, features)
            
            # Compute loss
            loss = self._compute_loss(logits, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (prevents exploding gradients)
            if self.config["gradient_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["gradient_clip"]
                )
            
            # Update weights
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_accuracy += self._compute_accuracy(logits.detach(), target)
            num_batches += 1
            
            # Logging
            if self.config["verbose"] and batch_idx % self.config["log_every"] == 0:
                print(f"  Batch {batch_idx + 1}/{len(self.train_loader)} | "
                      f"Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return avg_loss, avg_accuracy
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """
        Evaluate on validation set.
        
        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        self.model.eval()  # Set to evaluation mode
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            # Prepare batch
            features, input_ids, target = self._prepare_batch(batch)
            
            # Forward pass (no gradient computation)
            logits = self.model(input_ids, features)
            
            # Compute metrics
            loss = self._compute_loss(logits, target)
            accuracy = self._compute_accuracy(logits, target)
            
            total_loss += loss.item()
            total_accuracy += accuracy
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return avg_loss, avg_accuracy
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """
        Check if training should stop early.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop, False otherwise
        """
        min_delta = self.config["min_delta"]
        
        if val_loss < self.best_val_loss - min_delta:
            # Improvement! Reset patience counter
            self.best_val_loss = val_loss
            self.patience_counter = 0
            
            # Save best model state
            self.best_model_state = {
                k: v.cpu().clone() for k, v in self.model.state_dict().items()
            }
            
            return False
        else:
            # No improvement
            self.patience_counter += 1
            
            if self.patience_counter >= self.config["patience"]:
                return True  # Stop training
            
            return False
    
    def _save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        filename: Optional[str] = None
    ):
        """
        Save a training checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: If True, save as 'best_model.pt'
            filename: Custom filename (optional)
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "model_config": LSTM_CONFIG,
            "history": self.history.to_dict(),
        }
        
        # Save regular checkpoint
        if filename:
            path = self.checkpoint_dir / filename
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        
        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            if self.config["verbose"]:
                print(f"  💾 Saved best model (val_loss: {self.best_val_loss:.4f})")
    
    def _load_checkpoint(self, path: Union[str, Path]) -> int:
        """
        Load a checkpoint and resume training.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            Epoch to resume from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if checkpoint["scheduler_state_dict"] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.best_val_loss = checkpoint["best_val_loss"]
        self.history = TrainingHistory(**checkpoint["history"])
        
        return checkpoint["epoch"] + 1  # Resume from next epoch
    
    def train(
        self,
        num_epochs: Optional[int] = None,
        resume_from: Optional[Union[str, Path]] = None
    ) -> TrainingHistory:
        """
        Run the full training loop.
        
        Args:
            num_epochs: Number of epochs to train (uses config if None)
            resume_from: Path to checkpoint to resume from (optional)
            
        Returns:
            TrainingHistory with all metrics
        """
        if num_epochs is None:
            num_epochs = self.config["max_epochs"]
        
        start_epoch = 0
        
        # Resume from checkpoint if specified
        if resume_from:
            start_epoch = self._load_checkpoint(resume_from)
            print(f"Resumed training from epoch {start_epoch}")
        
        training_start_time = time.time()
        
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"\nEpoch {epoch + 1}/{num_epochs} | LR: {current_lr:.2e}")
            print("-" * 50)
            
            # Training phase
            train_loss, train_acc = self.train_epoch()
            
            # Validation phase
            val_loss, val_acc = self.validate()
            
            # Update learning rate scheduler
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Store metrics
            self.history.train_losses.append(train_loss)
            self.history.val_losses.append(val_loss)
            self.history.train_accuracies.append(train_acc)
            self.history.val_accuracies.append(val_acc)
            self.history.learning_rates.append(current_lr)
            self.history.epoch_times.append(epoch_time)
            
            # Print epoch summary
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2%}")
            print(f"  Time: {epoch_time:.1f}s")
            
            # Check for best model
            is_best = val_loss < self.best_val_loss - self.config["min_delta"]
            if is_best:
                self.history.best_epoch = epoch + 1
                self.history.best_val_loss = val_loss
                print(f"  ⭐ New best model!")
            
            # Early stopping check
            should_stop = self._check_early_stopping(val_loss)
            
            # Save checkpoint
            if is_best:
                self._save_checkpoint(epoch + 1, is_best=True)
            elif (epoch + 1) % self.config["save_every"] == 0:
                self._save_checkpoint(epoch + 1, is_best=False)
            
            # Early stopping
            if should_stop:
                print(f"\n⚠️ Early stopping triggered! No improvement for {self.config['patience']} epochs.")
                break
        
        # Calculate total training time
        self.history.total_training_time = time.time() - training_start_time
        
        # Restore best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            print(f"\n✅ Restored best model from epoch {self.history.best_epoch}")
        
        # Save final history
        history_path = self.checkpoint_dir / "training_history.json"
        self.history.save(history_path)
        
        # Print summary
        self._print_training_summary()
        
        return self.history
    
    def _print_training_summary(self):
        """Print a summary of training results."""
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Total time: {self.history.total_training_time / 60:.1f} minutes")
        print(f"Epochs trained: {len(self.history.train_losses)}")
        print(f"Best epoch: {self.history.best_epoch}")
        print(f"Best validation loss: {self.history.best_val_loss:.4f}")
        print(f"Best validation accuracy: {max(self.history.val_accuracies):.2%}")
        print(f"Final training loss: {self.history.train_losses[-1]:.4f}")
        print(f"Final validation loss: {self.history.val_losses[-1]:.4f}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        print("=" * 70)
    
    def generate_samples(
        self,
        num_samples: int = 5,
        temperature: float = 0.8
    ) -> List[Dict]:
        """
        Generate sample outputs with the current model.
        
        Useful for qualitative evaluation during/after training.
        
        Args:
            num_samples: Number of samples to generate
            temperature: Sampling temperature
            
        Returns:
            List of generated samples (dicts with chords and strum)
        """
        self.model.eval()
        samples = []
        
        # Sample different feature combinations
        from src.models.feature_encoder import FEATURE_VALUES
        import random
        
        for _ in range(num_samples):
            features = {
                "key": random.choice(FEATURE_VALUES["key"]),
                "mode": random.choice(FEATURE_VALUES["mode"]),
                "genre": random.choice(FEATURE_VALUES["genre"]),
                "emotion": random.choice(FEATURE_VALUES["emotion"]),
                "tempo": random.randint(60, 160)
            }
            
            result = self.model.generate(
                features=features,
                temperature=temperature,
                max_length=20
            )
            
            samples.append({
                "features": features,
                "chords": result["chords"],
                "strum_pattern": result["strum_pattern"]
            })
        
        return samples


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_trainer(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: Optional[GuitarLSTM] = None,
    config: Optional[Dict] = None,
    device: Optional[str] = None,
    checkpoint_dir: str = "checkpoints"
) -> GuitarTrainer:
    """
    Create a trainer with default settings.
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        model: GuitarLSTM model (creates new if None)
        config: Training config (uses default if None)
        device: Device to train on (auto-detect if None)
        checkpoint_dir: Directory for checkpoints
        
    Returns:
        Configured GuitarTrainer instance
    """
    if model is None:
        model = GuitarLSTM()
    
    return GuitarTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir
    )


def quick_train(
    train_path: str,
    val_path: str,
    num_epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 5e-4,
    device: str = None,
    checkpoint_dir: str = "checkpoints"
) -> Tuple[GuitarLSTM, TrainingHistory]:
    """
    Quick training function for simple experiments.
    
    Args:
        train_path: Path to training JSONL
        val_path: Path to validation JSONL
        num_epochs: Number of epochs to train
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
        checkpoint_dir: Directory for checkpoints
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    from src.train.dataset import create_dataloaders
    
    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(
        train_path=train_path,
        val_path=val_path,
        batch_size=batch_size
    )
    
    # Create config
    config = TRAINING_CONFIG.copy()
    config["learning_rate"] = learning_rate
    config["max_epochs"] = num_epochs
    config["batch_size"] = batch_size
    
    # Create and run trainer
    trainer = create_trainer(
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir
    )
    
    history = trainer.train()
    
    return trainer.model, history


# =============================================================================
# TESTING / DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    """
    Run this file directly to test the trainer:
        python -m src.train.trainer
    """
    print("=" * 70)
    print("GUITAR TRAINER DEMONSTRATION")
    print("=" * 70)
    
    # Check if we have data files
    data_paths = [
        Path("/mnt/user-data/uploads/train.jsonl"),
        Path("data/processed/train.jsonl"),
    ]
    
    train_path = None
    val_path = None
    
    for path in data_paths:
        if path.exists():
            train_path = path
            val_path = path.parent / "val.jsonl"
            break
    
    if train_path is None:
        print("No data files found. Demonstrating with dummy data...")
        
        # Create dummy model and show config
        model = GuitarLSTM()
        print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"\nTraining configuration:")
        for key, value in TRAINING_CONFIG.items():
            print(f"  {key}: {value}")
    else:
        print(f"\nFound data at: {train_path}")
        print(f"To train, run:")
        print(f"  trainer.train(num_epochs=50)")
    
    print("\n" + "=" * 70)
    print("TRAINER DEMONSTRATION COMPLETE ✓")
    print("=" * 70)
