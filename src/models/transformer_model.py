"""
Transformer Model for Guitar Pattern Generation
================================================

This module implements a decoder-only Transformer architecture for
generating guitar chord progressions and strumming patterns. It serves
as a comparison model for the LSTM in the thesis ablation study.

Architecture Overview:
    1. Token embeddings + Positional encoding + Conditioning
    2. N Transformer decoder layers with masked self-attention
    3. Linear output projection to vocabulary probabilities

The Transformer processes all tokens in parallel during training,
using a causal mask to prevent tokens from attending to future positions.
During inference, it generates autoregressively like the LSTM.

Key Differences from LSTM:
    - Parallel processing (faster training)
    - Self-attention instead of recurrence
    - Explicit positional encoding
    - Better at capturing long-range dependencies

Example:
    model = GuitarTransformer()
    
    # Training
    logits = model(
        input_ids=batch["input_ids"],
        features=batch_features
    )
    
    # Inference
    result = model.generate(
        features={"key": "G", "mode": "major", ...},
        temperature=0.8
    )

Author: Rohan Rajendra Dhanawade
Project: Master's Thesis - SRH Berlin University of Applied Sciences
"""

import math
from typing import Dict, List, Optional, Tuple, Union

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    raise ImportError("PyTorch is required. Install with: pip install torch")

# Import our modules
from src.models.tokenizer import VOCAB_SIZE, MAX_SEQ_LENGTH, MusicTokenizer
from src.models.feature_encoder import FeatureEncoder, CONDITIONING_DIM


# =============================================================================
# CONFIGURATION
# =============================================================================

TRANSFORMER_CONFIG = {
    # Token embedding
    "vocab_size": VOCAB_SIZE,           # 36 tokens
    "model_dim": 256,                   # Model dimension (d_model)
    
    # Conditioning (from FeatureEncoder)
    "conditioning_dim": CONDITIONING_DIM,  # 128 dims
    
    # Transformer architecture
    "num_layers": 4,                    # Number of decoder layers
    "num_heads": 4,                     # Number of attention heads
    "feedforward_dim": 512,             # FFN intermediate dimension
    "dropout": 0.2,                     # Dropout probability
    
    # Output
    "output_size": VOCAB_SIZE,          # 36 tokens
    
    # Sequence
    "max_seq_length": MAX_SEQ_LENGTH,   # 20 tokens
    
    # Special token IDs
    "pad_id": 0,
    "bos_id": 1,
    "eos_id": 2,
    "sep_id": 3,
}


# =============================================================================
# POSITIONAL ENCODING
# =============================================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as described in "Attention is All You Need".
    
    Since Transformers have no inherent notion of sequence order (unlike RNNs),
    we add positional information through these encodings. The sinusoidal
    pattern allows the model to learn relative positions.
    
    Formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Where pos is the position and i is the dimension.
    
    Args:
        d_model: Model dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the divisor term
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter, but saved with model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            
        Returns:
            Tensor with positional encoding added [batch, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# =============================================================================
# TRANSFORMER MODEL
# =============================================================================

class GuitarTransformer(nn.Module):
    """
    Decoder-only Transformer for guitar pattern generation.
    
    This model uses masked self-attention to generate sequences
    autoregressively, conditioned on musical features.
    
    Architecture:
        - Token embedding (256 dims)
        - Positional encoding (sinusoidal)
        - Conditioning projection and addition
        - 4 Transformer decoder layers
        - Linear output projection
    
    Attributes:
        token_embedding: Embedding layer for input tokens
        positional_encoding: Sinusoidal position embeddings
        feature_encoder: Encodes features → conditioning vector
        cond_projection: Projects conditioning to model dimension
        decoder_layers: Stack of Transformer decoder layers
        output_projection: Linear layer for output probabilities
        
    Args:
        config: Configuration dictionary (uses TRANSFORMER_CONFIG if None)
        feature_encoder: Pre-built FeatureEncoder (creates new if None)
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        feature_encoder: Optional[FeatureEncoder] = None
    ):
        """Initialize the Transformer model."""
        super().__init__()
        
        # Use default config if not provided
        self.config = config if config else TRANSFORMER_CONFIG.copy()
        
        # Store dimensions
        self.vocab_size = self.config["vocab_size"]
        self.model_dim = self.config["model_dim"]
        self.cond_dim = self.config["conditioning_dim"]
        self.num_layers = self.config["num_layers"]
        self.num_heads = self.config["num_heads"]
        self.ff_dim = self.config["feedforward_dim"]
        self.dropout_rate = self.config["dropout"]
        self.max_seq_length = self.config["max_seq_length"]
        
        # Special token IDs
        self.pad_id = self.config["pad_id"]
        self.bos_id = self.config["bos_id"]
        self.eos_id = self.config["eos_id"]
        self.sep_id = self.config["sep_id"]
        
        # ─────────────────────────────────────────────────────────────────────
        # Layer 1: Token Embedding
        # ─────────────────────────────────────────────────────────────────────
        self.token_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.model_dim,
            padding_idx=self.pad_id
        )
        
        # Scale embeddings by sqrt(d_model) as in original Transformer
        self.embedding_scale = math.sqrt(self.model_dim)
        
        # ─────────────────────────────────────────────────────────────────────
        # Layer 2: Positional Encoding
        # ─────────────────────────────────────────────────────────────────────
        self.positional_encoding = PositionalEncoding(
            d_model=self.model_dim,
            max_len=self.max_seq_length + 10,  # Small buffer
            dropout=self.dropout_rate
        )
        
        # ─────────────────────────────────────────────────────────────────────
        # Layer 3: Feature Encoder and Conditioning Projection
        # ─────────────────────────────────────────────────────────────────────
        if feature_encoder is not None:
            self.feature_encoder = feature_encoder
        else:
            self.feature_encoder = FeatureEncoder(output_dim=self.cond_dim)
        
        # Project conditioning from 128 to model_dim (256)
        self.cond_projection = nn.Sequential(
            nn.Linear(self.cond_dim, self.model_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        
        # ─────────────────────────────────────────────────────────────────────
        # Layer 4: Transformer Decoder Layers
        # Using PyTorch's built-in TransformerDecoderLayer
        # ─────────────────────────────────────────────────────────────────────
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.model_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_dim,
            dropout=self.dropout_rate,
            activation='gelu',  # GELU often works better than ReLU
            batch_first=True,   # Input shape: [batch, seq, features]
            norm_first=True     # Pre-norm (more stable training)
        )
        
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=self.num_layers
        )
        
        # ─────────────────────────────────────────────────────────────────────
        # Layer 5: Output Projection
        # ─────────────────────────────────────────────────────────────────────
        self.output_projection = nn.Linear(self.model_dim, self.vocab_size)
        
        # ─────────────────────────────────────────────────────────────────────
        # Layer Normalization (final)
        # ─────────────────────────────────────────────────────────────────────
        self.final_norm = nn.LayerNorm(self.model_dim)
        
        # Dropout
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Tokenizer for generation
        self.tokenizer = MusicTokenizer()
        
        # Initialize weights
        self._init_weights()
        
        # Cache for causal mask
        self._causal_mask_cache = {}
    
    def _init_weights(self):
        """Initialize model weights."""
        # Token embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        self.token_embedding.weight.data[self.pad_id].zero_()
        
        # Output projection (tie weights with embedding is optional)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_projection.bias)
        
        # Conditioning projection
        for module in self.cond_projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate a causal (look-ahead) mask for self-attention.
        
        The mask ensures that position i can only attend to positions <= i.
        This prevents the model from "cheating" by looking at future tokens.
        
        Args:
            seq_len: Sequence length
            device: Device to create mask on
            
        Returns:
            Causal mask [seq_len, seq_len] where True = masked (can't attend)
        """
        # Check cache first
        if seq_len in self._causal_mask_cache:
            cached = self._causal_mask_cache[seq_len]
            if cached.device == device:
                return cached
        
        # Create upper triangular mask (True above diagonal)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )
        
        # Cache for reuse
        self._causal_mask_cache[seq_len] = mask
        
        return mask
    
    def _generate_padding_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Generate padding mask.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            
        Returns:
            Padding mask [batch, seq_len] where True = padded (ignore)
        """
        return input_ids == self.pad_id
    
    def forward(
        self,
        input_ids: torch.Tensor,
        features: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with teacher forcing (for training).
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            features: Dictionary of feature tensors for conditioning
            attention_mask: Optional padding mask [batch, seq_len]
            
        Returns:
            logits: Predictions [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 1: Encode features → conditioning vector
        # ─────────────────────────────────────────────────────────────────────
        conditioning = self.feature_encoder(features)  # [batch, 128]
        cond_projected = self.cond_projection(conditioning)  # [batch, 256]
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 2: Embed tokens and add positional encoding
        # ─────────────────────────────────────────────────────────────────────
        token_embeds = self.token_embedding(input_ids) * self.embedding_scale
        # [batch, seq_len, 256]
        
        token_embeds = self.positional_encoding(token_embeds)
        # [batch, seq_len, 256]
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 3: Add conditioning to every position
        # ─────────────────────────────────────────────────────────────────────
        cond_expanded = cond_projected.unsqueeze(1).expand(-1, seq_len, -1)
        # [batch, seq_len, 256]
        
        decoder_input = token_embeds + cond_expanded
        decoder_input = self.dropout(decoder_input)
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 4: Generate masks
        # ─────────────────────────────────────────────────────────────────────
        causal_mask = self._generate_causal_mask(seq_len, device)
        padding_mask = self._generate_padding_mask(input_ids)
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 5: Run through Transformer decoder
        # Note: We use the decoder in "self-attention only" mode
        # by passing the same tensor as both tgt and memory
        # ─────────────────────────────────────────────────────────────────────
        
        # Create a dummy memory (conditioning) for the decoder
        # Shape: [batch, 1, model_dim]
        memory = cond_projected.unsqueeze(1)
        
        decoder_output = self.decoder(
            tgt=decoder_input,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask
        )
        # [batch, seq_len, 256]
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 6: Final normalization and output projection
        # ─────────────────────────────────────────────────────────────────────
        decoder_output = self.final_norm(decoder_output)
        logits = self.output_projection(decoder_output)
        # [batch, seq_len, 36]
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        features: Dict[str, Union[str, int, torch.Tensor]],
        max_length: int = 20,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True
    ) -> Dict[str, Union[List[int], List[str], str]]:
        """
        Generate a chord progression and strumming pattern autoregressively.
        
        Args:
            features: Musical features (strings or tensor indices)
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_k: Top-k sampling (optional)
            top_p: Nucleus sampling threshold (optional)
            do_sample: If False, use greedy decoding
            
        Returns:
            Dictionary with chords, strum_pattern, and token_ids
        """
        self.eval()
        device = next(self.parameters()).device
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 1: Encode features → conditioning
        # ─────────────────────────────────────────────────────────────────────
        conditioning = self.feature_encoder(features, batch_size=1)
        if conditioning.device != device:
            conditioning = conditioning.to(device)
        
        cond_projected = self.cond_projection(conditioning)  # [1, 256]
        memory = cond_projected.unsqueeze(1)  # [1, 1, 256]
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 2: Start with <BOS> token
        # ─────────────────────────────────────────────────────────────────────
        generated_ids = [self.bos_id]
        current_seq = torch.tensor([[self.bos_id]], device=device)
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 3: Autoregressive generation loop
        # ─────────────────────────────────────────────────────────────────────
        for _ in range(max_length - 1):
            seq_len = current_seq.size(1)
            
            # Embed and add positional encoding
            token_embeds = self.token_embedding(current_seq) * self.embedding_scale
            token_embeds = self.positional_encoding(token_embeds)
            
            # Add conditioning
            cond_expanded = cond_projected.unsqueeze(1).expand(-1, seq_len, -1)
            decoder_input = token_embeds + cond_expanded
            
            # Generate causal mask
            causal_mask = self._generate_causal_mask(seq_len, device)
            
            # Run decoder
            decoder_output = self.decoder(
                tgt=decoder_input,
                memory=memory,
                tgt_mask=causal_mask
            )
            
            # Get logits for last position only
            decoder_output = self.final_norm(decoder_output)
            logits = self.output_projection(decoder_output[:, -1, :])  # [1, 36]
            
            # Sample next token
            next_token_id = self._sample_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample
            )
            
            generated_ids.append(next_token_id)
            
            # Stop if we generate <EOS>
            if next_token_id == self.eos_id:
                break
            
            # Append to sequence for next iteration
            next_token_tensor = torch.tensor([[next_token_id]], device=device)
            current_seq = torch.cat([current_seq, next_token_tensor], dim=1)
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 4: Decode generated sequence
        # ─────────────────────────────────────────────────────────────────────
        chords, strum_pattern = self.tokenizer.decode(generated_ids)
        
        raw_sequence = [self.tokenizer.id_to_token.get(tid, f"<UNK:{tid}>")
                       for tid in generated_ids]
        
        return {
            "token_ids": generated_ids,
            "chords": chords,
            "strum_pattern": strum_pattern,
            "raw_sequence": raw_sequence
        }
    
    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True
    ) -> int:
        """
        Sample a token from the logit distribution.
        
        Same implementation as LSTM for fair comparison.
        """
        logits = logits.squeeze(0)  # [vocab_size]
        
        # Greedy decoding
        if not do_sample:
            return logits.argmax().item()
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Top-k filtering
        if top_k is not None and top_k > 0:
            top_k = min(top_k, probs.size(-1))
            top_k_probs, top_k_indices = torch.topk(probs, top_k)
            probs = torch.zeros_like(probs).scatter_(-1, top_k_indices, top_k_probs)
            probs = probs / probs.sum()
        
        # Top-p (nucleus) filtering
        if top_p is not None and top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            sorted_mask = cumulative_probs > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            
            sorted_probs[sorted_mask] = 0
            probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)
            probs = probs / probs.sum()
        
        # Sample from distribution
        token_id = torch.multinomial(probs, num_samples=1).item()
        
        return token_id
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Count parameters in each component."""
        def count_params(module):
            return sum(p.numel() for p in module.parameters())
        
        return {
            "token_embedding": count_params(self.token_embedding),
            "positional_encoding": 0,  # Buffer, not parameters
            "feature_encoder": count_params(self.feature_encoder),
            "cond_projection": count_params(self.cond_projection),
            "decoder": count_params(self.decoder),
            "output_projection": count_params(self.output_projection),
            "final_norm": count_params(self.final_norm),
            "total": count_params(self)
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_transformer_model(config: Optional[Dict] = None) -> GuitarTransformer:
    """Create a GuitarTransformer model with the given config."""
    return GuitarTransformer(config=config)


def load_transformer_model(checkpoint_path: str, device: str = "cpu") -> GuitarTransformer:
    """
    Load a trained Transformer model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pt checkpoint
        device: Device to load on
        
    Returns:
        Loaded model in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get("config", TRANSFORMER_CONFIG)
    model = GuitarTransformer(config=config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model


# =============================================================================
# TESTING / DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    """
    Run this file directly to see the Transformer model in action:
        python -m src.models.transformer_model
    """
    print("=" * 70)
    print("GUITAR TRANSFORMER MODEL DEMONSTRATION")
    print("=" * 70)
    
    # Create model
    model = GuitarTransformer()
    print(f"\nModel created successfully!")
    
    # Count parameters
    param_counts = model.get_num_parameters()
    print("\nParameter counts:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")
    
    # Compare with LSTM
    print("\n" + "-" * 70)
    print("COMPARISON WITH LSTM")
    print("-" * 70)
    print(f"  Transformer: {param_counts['total']:,} parameters")
    print(f"  LSTM:        ~1,082,372 parameters")
    print(f"  Difference:  {param_counts['total'] - 1082372:+,}")
    
    # Example 1: Forward pass
    print("\n" + "-" * 70)
    print("EXAMPLE 1: Forward Pass (Training Mode)")
    print("-" * 70)
    
    batch_size = 4
    seq_len = 15
    
    dummy_input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
    dummy_features = {
        "key": torch.randint(0, 12, (batch_size,)),
        "mode": torch.randint(0, 2, (batch_size,)),
        "genre": torch.randint(0, 9, (batch_size,)),
        "emotion": torch.randint(0, 8, (batch_size,)),
        "tempo": torch.randint(0, 10, (batch_size,)),
    }
    
    print(f"Input shape: {dummy_input_ids.shape}")
    
    logits = model(dummy_input_ids, dummy_features)
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected: [{batch_size}, {seq_len}, {VOCAB_SIZE}]")
    
    # Example 2: Generation
    print("\n" + "-" * 70)
    print("EXAMPLE 2: Generation (Inference Mode)")
    print("-" * 70)
    
    features = {
        "key": "G",
        "mode": "major",
        "genre": "folk",
        "emotion": "upbeat",
        "tempo": 110
    }
    
    print(f"Input features: {features}")
    
    result = model.generate(features, temperature=0.8)
    
    print(f"\nGenerated chords: {result['chords']}")
    print(f"Generated strum: {result['strum_pattern']}")
    print(f"Raw sequence: {result['raw_sequence']}")
    
    # Example 3: Multiple generations
    print("\n" + "-" * 70)
    print("EXAMPLE 3: Multiple Generations (Temperature Comparison)")
    print("-" * 70)
    
    print("\nGreedy (no randomness):")
    for i in range(3):
        result = model.generate(features, do_sample=False)
        print(f"  Run {i+1}: {result['chords']} | {result['strum_pattern']}")
    
    print("\nTemperature=0.8:")
    for i in range(3):
        result = model.generate(features, temperature=0.8)
        print(f"  Run {i+1}: {result['chords']} | {result['strum_pattern']}")
    
    print("\n" + "=" * 70)
    print("GUITAR TRANSFORMER MODEL DEMONSTRATION COMPLETE ✓")
    print("=" * 70)
