"""
PyTorch Dataset for Guitar Pattern Generation
==============================================

This module provides the data loading infrastructure for training
the sequence generation models. It handles:
- Loading JSONL data files
- Tokenizing chord progressions and strumming patterns
- Preparing feature indices for the conditioning encoder
- Creating batched DataLoaders for training

The dataset returns samples in a format ready for the LSTM/Transformer
models, with proper padding and attention masks.

Example:
    from src.train.dataset import GuitarDataset, create_dataloaders
    
    # Single dataset
    train_dataset = GuitarDataset("data/processed/train.jsonl")
    sample = train_dataset[0]
    
    # Full dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_path="data/processed/train.jsonl",
        val_path="data/processed/val.jsonl",
        test_path="data/processed/test.jsonl",
        batch_size=16
    )

Author: Rohan Rajendra Dhanawade
Project: Master's Thesis - SRH Berlin University of Applied Sciences
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# PyTorch imports
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Dummy classes for documentation
    class Dataset:
        pass
    class DataLoader:
        pass

# Import our tokenizer and feature config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.tokenizer import MusicTokenizer, VOCAB_SIZE, MAX_SEQ_LENGTH
from src.models.feature_encoder import (
    FEATURE_VALUES,
    EMBEDDING_CONFIG,
    TEMPO_BUCKETS,
    CONDITIONING_DIM
)


# =============================================================================
# FEATURE INDEX MAPPINGS
# =============================================================================
# These convert string feature values to integer indices for the embeddings

KEY_TO_IDX = {key: idx for idx, key in enumerate(FEATURE_VALUES["key"])}
MODE_TO_IDX = {mode: idx for idx, mode in enumerate(FEATURE_VALUES["mode"])}
GENRE_TO_IDX = {genre: idx for idx, genre in enumerate(FEATURE_VALUES["genre"])}
EMOTION_TO_IDX = {emotion: idx for idx, emotion in enumerate(FEATURE_VALUES["emotion"])}


def bucketize_tempo(tempo: Union[int, float]) -> int:
    """
    Convert continuous tempo (BPM) to a bucket index.
    
    Args:
        tempo: Beats per minute (40-200)
        
    Returns:
        Bucket index (0-9)
    """
    tempo = max(40, min(200, tempo))
    for bucket_idx, (min_bpm, max_bpm) in enumerate(TEMPO_BUCKETS):
        if min_bpm <= tempo <= max_bpm:
            return bucket_idx
    return 4  # Default to middle bucket


# =============================================================================
# GUITAR DATASET CLASS
# =============================================================================

class GuitarDataset(Dataset):
    """
    PyTorch Dataset for guitar chord progressions and strumming patterns.
    
    This dataset loads samples from a JSONL file and prepares them for
    training the sequence generation models. Each sample is tokenized
    and the features are converted to indices for the conditioning encoder.
    
    Attributes:
        samples (List[Dict]): Raw samples loaded from JSONL
        tokenizer (MusicTokenizer): Tokenizer for encoding sequences
        max_seq_length (int): Maximum sequence length (with padding)
        return_raw (bool): If True, also return raw sample data
        
    Example:
        >>> dataset = GuitarDataset("train.jsonl")
        >>> len(dataset)
        129
        >>> sample = dataset[0]
        >>> sample["input_ids"].shape
        torch.Size([20])
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: Optional[MusicTokenizer] = None,
        max_seq_length: int = MAX_SEQ_LENGTH,
        return_raw: bool = False
    ):
        """
        Initialize the dataset by loading data from JSONL file.
        
        Args:
            data_path: Path to JSONL file containing samples
            tokenizer: MusicTokenizer instance (creates default if None)
            max_seq_length: Maximum sequence length for padding
            return_raw: If True, include raw sample data in output
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for GuitarDataset. "
                "Install with: pip install torch"
            )
        
        self.data_path = Path(data_path)
        self.max_seq_length = max_seq_length
        self.return_raw = return_raw
        
        # Create or use provided tokenizer
        self.tokenizer = tokenizer if tokenizer else MusicTokenizer(max_seq_length)
        
        # Load samples from JSONL
        self.samples = self._load_jsonl(self.data_path)
        
        # Validate samples and report statistics
        self._validate_samples()
    
    def _load_jsonl(self, path: Path) -> List[Dict]:
        """
        Load samples from a JSONL file.
        
        Each line in the file should be a valid JSON object representing
        one guitar sample with chords, strum pattern, and features.
        
        Args:
            path: Path to JSONL file
            
        Returns:
            List of sample dictionaries
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If any line is invalid JSON
        """
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    raise json.JSONDecodeError(
                        f"Invalid JSON on line {line_num}: {e.msg}",
                        e.doc, e.pos
                    )
        
        return samples
    
    def _validate_samples(self):
        """
        Validate that all samples have required fields and valid values.
        
        Prints warnings for any issues found but doesn't raise exceptions
        to allow training to proceed with partial data.
        """
        required_fields = ['chords', 'strum_pattern', 'key', 'mode', 'genre', 'emotion', 'tempo']
        
        valid_count = 0
        issues = []
        
        for idx, sample in enumerate(self.samples):
            # Check required fields
            missing = [f for f in required_fields if f not in sample]
            if missing:
                issues.append(f"Sample {idx}: Missing fields {missing}")
                continue
            
            # Check feature values
            if sample['key'] not in KEY_TO_IDX:
                issues.append(f"Sample {idx}: Unknown key '{sample['key']}'")
                continue
            
            if sample['mode'] not in MODE_TO_IDX:
                issues.append(f"Sample {idx}: Unknown mode '{sample['mode']}'")
                continue
            
            if sample['genre'] not in GENRE_TO_IDX:
                issues.append(f"Sample {idx}: Unknown genre '{sample['genre']}'")
                continue
            
            if sample['emotion'] not in EMOTION_TO_IDX:
                issues.append(f"Sample {idx}: Unknown emotion '{sample['emotion']}'")
                continue
            
            # Check strum pattern length
            if len(sample['strum_pattern']) != 8:
                issues.append(f"Sample {idx}: Strum pattern length {len(sample['strum_pattern'])} != 8")
                continue
            
            valid_count += 1
        
        # Report validation results
        if issues:
            print(f"WARNING: {len(issues)} validation issues found:")
            for issue in issues[:5]:  # Show first 5
                print(f"  {issue}")
            if len(issues) > 5:
                print(f"  ... and {len(issues) - 5} more")
        
        # print(f"Loaded {valid_count}/{len(self.samples)} valid samples from {self.data_path.name}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single processed sample by index.
        
        This method tokenizes the chord progression and strumming pattern,
        creates the attention mask, and prepares feature indices for
        the conditioning encoder.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dictionary containing:
                - input_ids: Tokenized sequence [max_seq_length]
                - attention_mask: Mask for real vs padding [max_seq_length]
                - labels: Same as input_ids (for teacher forcing)
                - key_idx, mode_idx, genre_idx, emotion_idx, tempo_idx: Feature indices
                - (optional) raw: Original sample dict if return_raw=True
        """
        sample = self.samples[idx]
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 1: Tokenize the sequence (chords + strum pattern)
        # ─────────────────────────────────────────────────────────────────────
        
        input_ids = self.tokenizer.encode(
            chords=sample['chords'],
            strum_pattern=sample['strum_pattern'],
            add_special_tokens=True,
            pad_to_max_length=True
        )
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 2: Create attention mask (1 for real tokens, 0 for padding)
        # ─────────────────────────────────────────────────────────────────────
        
        attention_mask = self.tokenizer.create_attention_mask(input_ids)
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 3: Convert features to indices
        # ─────────────────────────────────────────────────────────────────────
        
        key_idx = KEY_TO_IDX.get(sample['key'], 0)
        mode_idx = MODE_TO_IDX.get(sample['mode'], 0)
        genre_idx = GENRE_TO_IDX.get(sample['genre'], 0)
        emotion_idx = EMOTION_TO_IDX.get(sample['emotion'], 0)
        tempo_idx = bucketize_tempo(sample['tempo'])
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 4: Create output dictionary with tensors
        # ─────────────────────────────────────────────────────────────────────
        
        output = {
            # Sequence data
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(input_ids, dtype=torch.long),  # Same as input for teacher forcing
            
            # Feature indices for conditioning
            "key_idx": torch.tensor(key_idx, dtype=torch.long),
            "mode_idx": torch.tensor(mode_idx, dtype=torch.long),
            "genre_idx": torch.tensor(genre_idx, dtype=torch.long),
            "emotion_idx": torch.tensor(emotion_idx, dtype=torch.long),
            "tempo_idx": torch.tensor(tempo_idx, dtype=torch.long),
        }
        
        # Optionally include raw sample
        if self.return_raw:
            output["raw"] = sample
        
        return output
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """
        Get human-readable information about a sample.
        
        Useful for debugging and understanding what the model sees.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary with sample details and tokenization breakdown
        """
        sample = self.samples[idx]
        processed = self[idx]
        
        # Decode back to verify round-trip
        decoded_chords, decoded_strum = self.tokenizer.decode(
            processed["input_ids"].tolist()
        )
        
        return {
            "raw_sample": sample,
            "token_ids": processed["input_ids"].tolist(),
            "attention_mask": processed["attention_mask"].tolist(),
            "sequence_length": processed["attention_mask"].sum().item(),
            "decoded_chords": decoded_chords,
            "decoded_strum": decoded_strum,
            "features": {
                "key": sample['key'],
                "key_idx": processed["key_idx"].item(),
                "mode": sample['mode'],
                "mode_idx": processed["mode_idx"].item(),
                "genre": sample['genre'],
                "genre_idx": processed["genre_idx"].item(),
                "emotion": sample['emotion'],
                "emotion_idx": processed["emotion_idx"].item(),
                "tempo": sample['tempo'],
                "tempo_idx": processed["tempo_idx"].item(),
            }
        }


# =============================================================================
# CUSTOM COLLATE FUNCTION
# =============================================================================

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to combine samples into a batch.
    
    This function stacks individual samples into batched tensors.
    It's passed to DataLoader to handle batch creation.
    
    Args:
        batch: List of sample dictionaries from __getitem__
        
    Returns:
        Dictionary with batched tensors:
            - input_ids: [batch_size, max_seq_length]
            - attention_mask: [batch_size, max_seq_length]
            - labels: [batch_size, max_seq_length]
            - key_idx, mode_idx, etc.: [batch_size]
    """
    return {
        # Stack sequence tensors
        "input_ids": torch.stack([s["input_ids"] for s in batch]),
        "attention_mask": torch.stack([s["attention_mask"] for s in batch]),
        "labels": torch.stack([s["labels"] for s in batch]),
        
        # Stack feature indices
        "key_idx": torch.stack([s["key_idx"] for s in batch]),
        "mode_idx": torch.stack([s["mode_idx"] for s in batch]),
        "genre_idx": torch.stack([s["genre_idx"] for s in batch]),
        "emotion_idx": torch.stack([s["emotion_idx"] for s in batch]),
        "tempo_idx": torch.stack([s["tempo_idx"] for s in batch]),
    }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_dataloaders(
    train_path: Union[str, Path],
    val_path: Union[str, Path],
    test_path: Optional[Union[str, Path]] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    shuffle_train: bool = True,
    tokenizer: Optional[MusicTokenizer] = None
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create DataLoaders for training, validation, and optionally test sets.
    
    This is the recommended way to set up data loading for training.
    
    Args:
        train_path: Path to training JSONL file
        val_path: Path to validation JSONL file
        test_path: Path to test JSONL file (optional)
        batch_size: Number of samples per batch
        num_workers: Number of parallel data loading workers
        shuffle_train: Whether to shuffle training data
        tokenizer: Shared tokenizer instance (creates default if None)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        test_loader is None if test_path not provided
        
    Example:
        >>> train_loader, val_loader, test_loader = create_dataloaders(
        ...     train_path="data/train.jsonl",
        ...     val_path="data/val.jsonl",
        ...     test_path="data/test.jsonl",
        ...     batch_size=16
        ... )
        >>> for batch in train_loader:
        ...     # batch["input_ids"].shape == [16, 20]
        ...     pass
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required. Install with: pip install torch")
    
    # Create shared tokenizer
    if tokenizer is None:
        tokenizer = MusicTokenizer()
    
    # Create datasets
    train_dataset = GuitarDataset(train_path, tokenizer=tokenizer)
    val_dataset = GuitarDataset(val_path, tokenizer=tokenizer)
    
    test_dataset = None
    if test_path:
        test_dataset = GuitarDataset(test_path, tokenizer=tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False  # Keep partial batches
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle test
            num_workers=num_workers,
            collate_fn=collate_fn
        )
    
    return train_loader, val_loader, test_loader


def get_dataset_stats(dataset: GuitarDataset) -> Dict[str, Any]:
    """
    Compute statistics about a dataset.
    
    Args:
        dataset: GuitarDataset instance
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        "num_samples": len(dataset),
        "genres": {},
        "emotions": {},
        "modes": {},
        "keys": {},
        "tempo_buckets": {},
        "avg_sequence_length": 0,
    }
    
    total_length = 0
    
    for sample in dataset.samples:
        # Count features
        stats["genres"][sample['genre']] = stats["genres"].get(sample['genre'], 0) + 1
        stats["emotions"][sample['emotion']] = stats["emotions"].get(sample['emotion'], 0) + 1
        stats["modes"][sample['mode']] = stats["modes"].get(sample['mode'], 0) + 1
        stats["keys"][sample['key']] = stats["keys"].get(sample['key'], 0) + 1
        
        # Count tempo buckets
        bucket = bucketize_tempo(sample['tempo'])
        stats["tempo_buckets"][bucket] = stats["tempo_buckets"].get(bucket, 0) + 1
        
        # Calculate sequence length (chords + strum + special tokens)
        seq_len = 1 + len(sample['chords']) + 1 + len(sample['strum_pattern']) + 1
        total_length += seq_len
    
    stats["avg_sequence_length"] = total_length / len(dataset) if dataset else 0
    
    return stats


# =============================================================================
# TESTING / DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    """
    Run this file directly to see the dataset in action:
        python -m src.train.dataset
    """
    import sys
    
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is required to run this demo.")
        print("Install with: pip install torch")
        sys.exit(1)
    
    print("=" * 70)
    print("GUITAR DATASET DEMONSTRATION")
    print("=" * 70)
    
    # Try to find data files
    data_paths = [
        Path("/mnt/user-data/uploads/train.jsonl"),
        Path("data/processed/train.jsonl"),
        Path("train.jsonl"),
    ]
    
    train_path = None
    for path in data_paths:
        if path.exists():
            train_path = path
            break
    
    if train_path is None:
        print("ERROR: Could not find train.jsonl in expected locations")
        print("Searched:", [str(p) for p in data_paths])
        sys.exit(1)
    
    print(f"\nUsing data file: {train_path}")
    
    # Create dataset
    dataset = GuitarDataset(train_path, return_raw=True)
    print(f"Loaded {len(dataset)} samples")
    
    # Example 1: Get a single sample
    print("\n" + "-" * 70)
    print("EXAMPLE 1: Single Sample")
    print("-" * 70)
    
    sample = dataset[0]
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Input IDs: {sample['input_ids'].tolist()}")
    print(f"Attention mask: {sample['attention_mask'].tolist()}")
    print(f"\nFeature indices:")
    print(f"  key_idx: {sample['key_idx'].item()}")
    print(f"  mode_idx: {sample['mode_idx'].item()}")
    print(f"  genre_idx: {sample['genre_idx'].item()}")
    print(f"  emotion_idx: {sample['emotion_idx'].item()}")
    print(f"  tempo_idx: {sample['tempo_idx'].item()}")
    
    # Example 2: Sample info with round-trip verification
    print("\n" + "-" * 70)
    print("EXAMPLE 2: Sample Info with Round-Trip Verification")
    print("-" * 70)
    
    info = dataset.get_sample_info(0)
    print(f"Original chords: {info['raw_sample']['chords']}")
    print(f"Decoded chords:  {info['decoded_chords']}")
    print(f"Original strum:  {info['raw_sample']['strum_pattern']}")
    print(f"Decoded strum:   {info['decoded_strum']}")
    print(f"Sequence length: {info['sequence_length']} tokens (including special tokens)")
    
    # Example 3: Create a DataLoader batch
    print("\n" + "-" * 70)
    print("EXAMPLE 3: DataLoader Batch")
    print("-" * 70)
    
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(loader))
    
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")
    print(f"Batch key_idx shape: {batch['key_idx'].shape}")
    
    # Example 4: Dataset statistics
    print("\n" + "-" * 70)
    print("EXAMPLE 4: Dataset Statistics")
    print("-" * 70)
    
    stats = get_dataset_stats(dataset)
    print(f"Total samples: {stats['num_samples']}")
    print(f"Average sequence length: {stats['avg_sequence_length']:.1f} tokens")
    print(f"Genre distribution: {stats['genres']}")
    print(f"Mode distribution: {stats['modes']}")
    
    print("\n" + "=" * 70)
    print("GUITAR DATASET DEMONSTRATION COMPLETE ✓")
    print("=" * 70)
