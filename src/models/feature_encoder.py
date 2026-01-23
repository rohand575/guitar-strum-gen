"""
Feature Encoder for Guitar Pattern Generation
==============================================

This module converts categorical prompt features (key, mode, genre, emotion)
and numerical tempo into a dense conditioning vector that guides the
sequence generation model.

Architecture:
    Each categorical feature → Embedding layer → dense vector
    Tempo → Bucketize → Embedding layer → dense vector
    All vectors → Concatenate → Linear layer → 128-dim output

The conditioning vector "tells" the model what kind of music to generate,
similar to how a guitarist might think "I need a happy folk song in G major".

Example:
    encoder = FeatureEncoder()
    
    features = {
        "key": "G",
        "mode": "major", 
        "genre": "folk",
        "emotion": "upbeat",
        "tempo": 110
    }
    
    conditioning = encoder(features)  # Shape: [batch_size, 128]

Author: Rohan Rajendra Dhanawade
Project: Master's Thesis - SRH Berlin University of Applied Sciences
"""

from typing import Dict, List, Optional, Union
import math

# PyTorch imports - required for this module
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy class for documentation purposes
    class nn:
        class Module:
            pass


# =============================================================================
# CONFIGURATION
# =============================================================================

# Feature categories and their possible values
# These MUST match what the DistilBERT parser outputs
FEATURE_VALUES = {
    "key": ["A", "Am", "Bm", "C", "D", "Dm", "E", "Em", "F", "Fm", "G", "Gm"],
    "mode": ["major", "minor"],
    "genre": ["acoustic", "ballad", "blues", "country", "folk", "indie", "jazz", "pop", "rock"],
    "emotion": ["dramatic", "energetic", "hopeful", "melancholic", "mellow", "nostalgic", "peaceful", "upbeat"],
}

# Embedding dimensions for each feature (from architecture spec)
EMBEDDING_CONFIG = {
    "key": {
        "num_values": 12,
        "embedding_dim": 32
    },
    "mode": {
        "num_values": 2,
        "embedding_dim": 16
    },
    "genre": {
        "num_values": 9,
        "embedding_dim": 32
    },
    "emotion": {
        "num_values": 8,
        "embedding_dim": 32
    },
    "tempo": {
        "num_buckets": 10,
        "embedding_dim": 16
    }
}

# Total conditioning dimension: 32 + 16 + 32 + 32 + 16 = 128
CONDITIONING_DIM = sum(cfg["embedding_dim"] for cfg in EMBEDDING_CONFIG.values())

# Tempo buckets for converting continuous BPM to categorical
# Each tuple is (min_bpm, max_bpm) inclusive
TEMPO_BUCKETS = [
    (40, 55),    # Bucket 0: Very slow (ballads, lullabies)
    (56, 70),    # Bucket 1: Slow (slow ballads)
    (71, 85),    # Bucket 2: Slow-moderate (emotional songs)
    (86, 100),   # Bucket 3: Moderate (standard pop)
    (101, 115),  # Bucket 4: Moderate-fast (upbeat pop)
    (116, 130),  # Bucket 5: Fast (energetic songs)
    (131, 145),  # Bucket 6: Fast-energetic (rock, punk)
    (146, 160),  # Bucket 7: Very fast (high energy)
    (161, 180),  # Bucket 8: Driving (fast rock, metal)
    (181, 200),  # Bucket 9: Maximum energy (extreme tempos)
]


# =============================================================================
# FEATURE ENCODER CLASS
# =============================================================================

class FeatureEncoder(nn.Module):
    """
    Encodes prompt features into a dense conditioning vector.
    
    This module takes categorical features (key, mode, genre, emotion) and
    numerical tempo, converts each to a learned embedding, and concatenates
    them into a single conditioning vector that guides sequence generation.
    
    The learned embeddings allow the model to discover relationships between
    features (e.g., "folk" and "country" might develop similar embeddings).
    
    Attributes:
        key_embedding: Embedding layer for musical key (12 keys × 32 dims)
        mode_embedding: Embedding layer for mode (2 modes × 16 dims)
        genre_embedding: Embedding layer for genre (9 genres × 32 dims)
        emotion_embedding: Embedding layer for emotion (8 emotions × 32 dims)
        tempo_embedding: Embedding layer for tempo bucket (10 buckets × 16 dims)
        output_projection: Linear layer for mixing features (128 → 128)
        output_dim: Final output dimension (128)
    """
    
    def __init__(self, output_dim: int = CONDITIONING_DIM):
        """
        Initialize the feature encoder with embedding layers.
        
        Args:
            output_dim: Dimension of output conditioning vector (default: 128)
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for FeatureEncoder. "
                "Install with: pip install torch"
            )
        
        super().__init__()
        
        self.output_dim = output_dim
        
        # ─────────────────────────────────────────────────────────────────────
        # Create embedding layers for each feature
        # nn.Embedding(num_categories, embedding_dim) creates a lookup table
        # ─────────────────────────────────────────────────────────────────────
        
        # Key embedding: 12 keys → 32 dimensions
        self.key_embedding = nn.Embedding(
            num_embeddings=EMBEDDING_CONFIG["key"]["num_values"],
            embedding_dim=EMBEDDING_CONFIG["key"]["embedding_dim"]
        )
        
        # Mode embedding: 2 modes → 16 dimensions
        self.mode_embedding = nn.Embedding(
            num_embeddings=EMBEDDING_CONFIG["mode"]["num_values"],
            embedding_dim=EMBEDDING_CONFIG["mode"]["embedding_dim"]
        )
        
        # Genre embedding: 9 genres → 32 dimensions
        self.genre_embedding = nn.Embedding(
            num_embeddings=EMBEDDING_CONFIG["genre"]["num_values"],
            embedding_dim=EMBEDDING_CONFIG["genre"]["embedding_dim"]
        )
        
        # Emotion embedding: 8 emotions → 32 dimensions  
        self.emotion_embedding = nn.Embedding(
            num_embeddings=EMBEDDING_CONFIG["emotion"]["num_values"],
            embedding_dim=EMBEDDING_CONFIG["emotion"]["embedding_dim"]
        )
        
        # Tempo embedding: 10 buckets → 16 dimensions
        self.tempo_embedding = nn.Embedding(
            num_embeddings=EMBEDDING_CONFIG["tempo"]["num_buckets"],
            embedding_dim=EMBEDDING_CONFIG["tempo"]["embedding_dim"]
        )
        
        # ─────────────────────────────────────────────────────────────────────
        # Create mappings from string values to indices
        # This allows us to accept string inputs like "folk" instead of index 4
        # ─────────────────────────────────────────────────────────────────────
        
        self.key_to_idx = {key: idx for idx, key in enumerate(FEATURE_VALUES["key"])}
        self.mode_to_idx = {mode: idx for idx, mode in enumerate(FEATURE_VALUES["mode"])}
        self.genre_to_idx = {genre: idx for idx, genre in enumerate(FEATURE_VALUES["genre"])}
        self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(FEATURE_VALUES["emotion"])}
        
        # ─────────────────────────────────────────────────────────────────────
        # Output projection layer
        # This linear layer "mixes" the concatenated embeddings and allows
        # the model to learn interactions between features
        # ─────────────────────────────────────────────────────────────────────
        
        concat_dim = sum(cfg["embedding_dim"] for cfg in EMBEDDING_CONFIG.values())
        
        self.output_projection = nn.Sequential(
            nn.Linear(concat_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)  # Light regularization
        )
        
        # Initialize weights using Xavier initialization
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize embedding and linear layer weights.
        
        Xavier initialization helps with training stability by keeping
        the variance of activations consistent across layers.
        """
        # Initialize embeddings with normal distribution
        for embedding in [self.key_embedding, self.mode_embedding, 
                         self.genre_embedding, self.emotion_embedding,
                         self.tempo_embedding]:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.1)
        
        # Initialize linear layers with Xavier
        for module in self.output_projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def bucketize_tempo(self, tempo: Union[int, float]) -> int:
        """
        Convert continuous tempo (BPM) to a bucket index.
        
        This allows the model to treat similar tempos the same way,
        which is more musically meaningful than treating 109 and 111 BPM
        as completely different values.
        
        Args:
            tempo: Beats per minute (40-200)
            
        Returns:
            Bucket index (0-9)
            
        Example:
            >>> encoder = FeatureEncoder()
            >>> encoder.bucketize_tempo(110)
            4  # Moderate-fast bucket
            >>> encoder.bucketize_tempo(65)
            1  # Slow bucket
        """
        # Clamp tempo to valid range
        tempo = max(40, min(200, tempo))
        
        # Find the appropriate bucket
        for bucket_idx, (min_bpm, max_bpm) in enumerate(TEMPO_BUCKETS):
            if min_bpm <= tempo <= max_bpm:
                return bucket_idx
        
        # Default to middle bucket if something goes wrong
        return 4
    
    def _convert_to_indices(
        self,
        features: Dict[str, Union[str, int, float]],
        batch_size: int,
        device  # torch.device, but we avoid the type hint for when torch isn't loaded
    ) -> Dict[str, "torch.Tensor"]:
        """
        Convert string feature values to tensor indices.
        
        Args:
            features: Dictionary with string values
            batch_size: Number of samples in batch
            device: Device to create tensors on
            
        Returns:
            Dictionary with tensor indices
        """
        # Handle key
        if isinstance(features.get("key"), str):
            key_idx = self.key_to_idx.get(features["key"], 0)
        else:
            key_idx = features.get("key", 0)
        
        # Handle mode
        if isinstance(features.get("mode"), str):
            mode_idx = self.mode_to_idx.get(features["mode"], 0)
        else:
            mode_idx = features.get("mode", 0)
        
        # Handle genre
        if isinstance(features.get("genre"), str):
            genre_idx = self.genre_to_idx.get(features["genre"], 0)
        else:
            genre_idx = features.get("genre", 0)
        
        # Handle emotion
        if isinstance(features.get("emotion"), str):
            emotion_idx = self.emotion_to_idx.get(features["emotion"], 0)
        else:
            emotion_idx = features.get("emotion", 0)
        
        # Handle tempo (bucketize if it's a raw BPM value)
        tempo_val = features.get("tempo", 100)
        if isinstance(tempo_val, (int, float)) and tempo_val > 9:
            # It's a raw BPM, bucketize it
            tempo_idx = self.bucketize_tempo(tempo_val)
        else:
            # It's already a bucket index
            tempo_idx = int(tempo_val)
        
        # Create tensors with shape [batch_size]
        return {
            "key": torch.tensor([key_idx] * batch_size, dtype=torch.long, device=device),
            "mode": torch.tensor([mode_idx] * batch_size, dtype=torch.long, device=device),
            "genre": torch.tensor([genre_idx] * batch_size, dtype=torch.long, device=device),
            "emotion": torch.tensor([emotion_idx] * batch_size, dtype=torch.long, device=device),
            "tempo": torch.tensor([tempo_idx] * batch_size, dtype=torch.long, device=device),
        }
    
    def forward(
        self,
        features: Dict[str, Union[str, int, float, "torch.Tensor"]],
        batch_size: int = 1
    ) -> "torch.Tensor":
        """
        Convert prompt features to conditioning vector.
        
        This is the main method that transforms human-readable features
        into the dense vector that guides sequence generation.
        
        Args:
            features: Dictionary containing:
                - key: Musical key ("G", "Am", etc.) or index
                - mode: "major" or "minor" or index
                - genre: Genre name or index
                - emotion: Emotion name or index
                - tempo: BPM (40-200) or bucket index (0-9)
            batch_size: Number of samples (for broadcasting single features)
            
        Returns:
            Conditioning vector of shape [batch_size, 128]
            
        Example:
            >>> encoder = FeatureEncoder()
            >>> features = {"key": "G", "mode": "major", "genre": "folk", 
            ...             "emotion": "upbeat", "tempo": 110}
            >>> conditioning = encoder(features)
            >>> conditioning.shape
            torch.Size([1, 128])
        """
        # Determine device from model parameters
        device = next(self.parameters()).device
        
        # Check if inputs are already tensors (batch mode) or need conversion
        if isinstance(features.get("key"), torch.Tensor):
            # Batch mode: features are already tensors
            key_idx = features["key"]
            mode_idx = features["mode"]
            genre_idx = features["genre"]
            emotion_idx = features["emotion"]
            tempo_idx = features["tempo"]
            batch_size = key_idx.size(0)
        else:
            # Single sample mode: convert strings to indices
            indices = self._convert_to_indices(features, batch_size, device)
            key_idx = indices["key"]
            mode_idx = indices["mode"]
            genre_idx = indices["genre"]
            emotion_idx = indices["emotion"]
            tempo_idx = indices["tempo"]
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 1: Look up embeddings for each feature
        # ─────────────────────────────────────────────────────────────────────
        
        key_embed = self.key_embedding(key_idx)       # [batch, 32]
        mode_embed = self.mode_embedding(mode_idx)     # [batch, 16]
        genre_embed = self.genre_embedding(genre_idx)  # [batch, 32]
        emotion_embed = self.emotion_embedding(emotion_idx)  # [batch, 32]
        tempo_embed = self.tempo_embedding(tempo_idx)  # [batch, 16]
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 2: Concatenate all embeddings
        # ─────────────────────────────────────────────────────────────────────
        
        # Concatenate along the feature dimension
        # Result shape: [batch, 32+16+32+32+16] = [batch, 128]
        concatenated = torch.cat([
            key_embed,
            mode_embed,
            genre_embed,
            emotion_embed,
            tempo_embed
        ], dim=-1)
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 3: Apply output projection (mixing layer)
        # ─────────────────────────────────────────────────────────────────────
        
        # The projection layer allows the model to learn interactions
        # between features (e.g., "folk" + "upbeat" might have a special meaning)
        conditioning = self.output_projection(concatenated)  # [batch, 128]
        
        return conditioning
    
    def get_output_dim(self) -> int:
        """Return the output dimension (128)."""
        return self.output_dim
    
    def get_feature_info(self) -> Dict:
        """
        Return information about feature encodings.
        
        Useful for debugging and understanding the model.
        """
        return {
            "key": {
                "values": list(self.key_to_idx.keys()),
                "embedding_dim": EMBEDDING_CONFIG["key"]["embedding_dim"]
            },
            "mode": {
                "values": list(self.mode_to_idx.keys()),
                "embedding_dim": EMBEDDING_CONFIG["mode"]["embedding_dim"]
            },
            "genre": {
                "values": list(self.genre_to_idx.keys()),
                "embedding_dim": EMBEDDING_CONFIG["genre"]["embedding_dim"]
            },
            "emotion": {
                "values": list(self.emotion_to_idx.keys()),
                "embedding_dim": EMBEDDING_CONFIG["emotion"]["embedding_dim"]
            },
            "tempo": {
                "buckets": TEMPO_BUCKETS,
                "embedding_dim": EMBEDDING_CONFIG["tempo"]["embedding_dim"]
            },
            "output_dim": self.output_dim
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_default_encoder() -> FeatureEncoder:
    """Create and return a default feature encoder instance."""
    return FeatureEncoder()


def prepare_features_for_batch(
    batch_features: List[Dict[str, Union[str, int, float]]],
    encoder: FeatureEncoder
) -> Dict[str, "torch.Tensor"]:
    """
    Prepare a batch of feature dictionaries for the encoder.
    
    Args:
        batch_features: List of feature dictionaries
        encoder: FeatureEncoder instance (for string→index mappings)
        
    Returns:
        Dictionary of tensors ready for encoder.forward()
    """
    batch_size = len(batch_features)
    
    # Collect indices for each feature
    key_indices = []
    mode_indices = []
    genre_indices = []
    emotion_indices = []
    tempo_indices = []
    
    for features in batch_features:
        # Convert each feature to index
        key_indices.append(encoder.key_to_idx.get(features.get("key", "C"), 3))
        mode_indices.append(encoder.mode_to_idx.get(features.get("mode", "major"), 0))
        genre_indices.append(encoder.genre_to_idx.get(features.get("genre", "pop"), 7))
        emotion_indices.append(encoder.emotion_to_idx.get(features.get("emotion", "mellow"), 4))
        
        # Bucketize tempo
        tempo = features.get("tempo", 100)
        tempo_indices.append(encoder.bucketize_tempo(tempo))
    
    return {
        "key": torch.tensor(key_indices, dtype=torch.long),
        "mode": torch.tensor(mode_indices, dtype=torch.long),
        "genre": torch.tensor(genre_indices, dtype=torch.long),
        "emotion": torch.tensor(emotion_indices, dtype=torch.long),
        "tempo": torch.tensor(tempo_indices, dtype=torch.long),
    }


# =============================================================================
# TESTING / DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    """
    Run this file directly to see the feature encoder in action:
        python -m src.models.feature_encoder
    """
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is required to run this demo.")
        print("Install with: pip install torch")
        exit(1)
    
    print("=" * 70)
    print("FEATURE ENCODER DEMONSTRATION")
    print("=" * 70)
    
    # Create encoder
    encoder = FeatureEncoder()
    print(f"\nFeatureEncoder created with output_dim={encoder.get_output_dim()}")
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Example 1: Single sample with string values
    print("\n" + "-" * 70)
    print("EXAMPLE 1: Single sample encoding (string inputs)")
    print("-" * 70)
    
    features = {
        "key": "G",
        "mode": "major",
        "genre": "folk",
        "emotion": "upbeat",
        "tempo": 110  # Will be bucketized to bucket 4
    }
    
    print(f"Input features: {features}")
    print(f"Tempo 110 BPM → Bucket {encoder.bucketize_tempo(110)} (moderate-fast)")
    
    conditioning = encoder(features)
    print(f"\nOutput shape: {conditioning.shape}")
    print(f"Output (first 10 values): {conditioning[0, :10].tolist()}")
    print(f"Output range: [{conditioning.min().item():.3f}, {conditioning.max().item():.3f}]")
    
    # Example 2: Different features produce different outputs
    print("\n" + "-" * 70)
    print("EXAMPLE 2: Different features → Different outputs")
    print("-" * 70)
    
    features_sad = {
        "key": "Am",
        "mode": "minor",
        "genre": "ballad",
        "emotion": "melancholic",
        "tempo": 60
    }
    
    print(f"Happy folk:  {features}")
    print(f"Sad ballad: {features_sad}")
    
    cond_happy = encoder(features)
    cond_sad = encoder(features_sad)
    
    # Compute similarity (cosine)
    similarity = torch.nn.functional.cosine_similarity(cond_happy, cond_sad, dim=1)
    print(f"\nCosine similarity between happy and sad: {similarity.item():.3f}")
    print("(Values close to 1 = similar, close to 0 = different)")
    
    # Example 3: Tempo bucketization
    print("\n" + "-" * 70)
    print("EXAMPLE 3: Tempo bucketization")
    print("-" * 70)
    
    test_tempos = [45, 65, 90, 110, 130, 150, 180]
    print("BPM → Bucket mapping:")
    for tempo in test_tempos:
        bucket = encoder.bucketize_tempo(tempo)
        bucket_range = TEMPO_BUCKETS[bucket]
        print(f"  {tempo:3d} BPM → Bucket {bucket} ({bucket_range[0]}-{bucket_range[1]} BPM)")
    
    # Example 4: Batch encoding
    print("\n" + "-" * 70)
    print("EXAMPLE 4: Batch encoding")
    print("-" * 70)
    
    batch_features = [
        {"key": "G", "mode": "major", "genre": "folk", "emotion": "upbeat", "tempo": 110},
        {"key": "Am", "mode": "minor", "genre": "ballad", "emotion": "melancholic", "tempo": 60},
        {"key": "E", "mode": "major", "genre": "rock", "emotion": "energetic", "tempo": 140},
    ]
    
    # Prepare batch
    batch_tensors = prepare_features_for_batch(batch_features, encoder)
    print(f"Batch size: {len(batch_features)}")
    
    # Encode batch
    batch_conditioning = encoder(batch_tensors)
    print(f"Batch output shape: {batch_conditioning.shape}")
    
    # Example 5: Feature info
    print("\n" + "-" * 70)
    print("EXAMPLE 5: Feature configuration info")
    print("-" * 70)
    
    info = encoder.get_feature_info()
    for feature, details in info.items():
        if feature == "tempo":
            print(f"  {feature}: {len(details['buckets'])} buckets, {details['embedding_dim']} dims")
        elif feature == "output_dim":
            print(f"  output_dim: {details}")
        else:
            print(f"  {feature}: {len(details['values'])} values, {details['embedding_dim']} dims")
    
    print("\n" + "=" * 70)
    print("FEATURE ENCODER DEMONSTRATION COMPLETE ✓")
    print("=" * 70)
