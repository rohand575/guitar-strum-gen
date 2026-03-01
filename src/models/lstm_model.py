"""
LSTM Model for Guitar Pattern Generation
========================================

This module implements the LSTM-based sequence generation model that
produces chord progressions and strumming patterns conditioned on
musical features (key, mode, genre, emotion, tempo).

Architecture Overview:
    1. Feature Encoder converts prompt features → 128-dim conditioning vector
    2. Conditioning vector → Linear layer → Initial hidden state h₀
    3. At each timestep: [token_embedding + conditioning] → LSTM → output
    4. Output layer → Probability distribution over 36 tokens

Training uses teacher forcing: the model receives the correct previous
token at each step, rather than its own predictions.

Inference uses autoregressive generation: starting from <BOS>, the model
generates one token at a time until <EOS> or max length.

Example:
    model = GuitarLSTM()
    
    # Training
    logits = model(
        input_ids=batch["input_ids"],      # [batch, seq_len]
        features=batch_features,            # Dict of feature tensors
        attention_mask=batch["attention_mask"]
    )
    
    # Inference
    generated = model.generate(
        features={"key": "G", "mode": "major", "genre": "folk", ...},
        max_length=20,
        temperature=0.8
    )

Author: Rohan Rajendra Dhanawade
Project: Master's Thesis - SRH Berlin University of Applied Sciences
"""

from typing import Dict, List, Optional, Tuple, Union
import math

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    raise ImportError("PyTorch is required for GuitarLSTM. Install with: pip install torch")

# Import our modules
from src.models.tokenizer import VOCAB_SIZE, MAX_SEQ_LENGTH, MusicTokenizer
from src.models.feature_encoder import FeatureEncoder, CONDITIONING_DIM


# =============================================================================
# CONFIGURATION
# =============================================================================

LSTM_CONFIG = {
    # Token embedding
    "vocab_size": VOCAB_SIZE,           # 36 tokens
    "token_embedding_dim": 64,          # Each token → 64-dim vector
    
    # Conditioning (from FeatureEncoder)
    "conditioning_dim": CONDITIONING_DIM,  # 128 dims
    
    # LSTM architecture
    "lstm_input_size": 64 + 128,        # token_embed + conditioning = 192
    "lstm_hidden_size": 256,            # Hidden state dimension
    "lstm_num_layers": 2,               # Stacked LSTM layers
    "lstm_dropout": 0.2,                # Dropout between LSTM layers
    
    # Output
    "output_size": VOCAB_SIZE,          # 36 tokens
    
    # Sequence
    "max_seq_length": MAX_SEQ_LENGTH,   # 20 tokens
    
    # Special token IDs (must match tokenizer)
    "pad_id": 0,
    "bos_id": 1,
    "eos_id": 2,
    "sep_id": 3,
}


# =============================================================================
# LSTM MODEL CLASS
# =============================================================================

class GuitarLSTM(nn.Module):
    """
    LSTM-based model for generating guitar chord progressions and strumming patterns.
    
    This model takes musical features (key, mode, genre, emotion, tempo) and
    generates a sequence of tokens representing chords and a strumming pattern.
    
    The architecture uses:
    - Learned token embeddings (64 dims)
    - Feature encoder for conditioning (128 dims) 
    - 2-layer LSTM with conditioning concatenated at each timestep
    - Linear output projection to vocabulary probabilities
    
    Attributes:
        token_embedding: Embedding layer for input tokens
        feature_encoder: Encodes features → conditioning vector
        cond_to_hidden: Projects conditioning → initial hidden state
        lstm: 2-layer LSTM for sequence modeling
        output_projection: Linear layer for output probabilities
        
    Args:
        config: Configuration dictionary (uses LSTM_CONFIG if None)
        feature_encoder: Pre-built FeatureEncoder (creates new if None)
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        feature_encoder: Optional[FeatureEncoder] = None
    ):
        """Initialize the LSTM model with all layers."""
        super().__init__()
        
        # Use default config if not provided
        self.config = config if config else LSTM_CONFIG.copy()
        
        # Store dimensions for easy access
        self.vocab_size = self.config["vocab_size"]
        self.token_dim = self.config["token_embedding_dim"]
        self.cond_dim = self.config["conditioning_dim"]
        self.hidden_size = self.config["lstm_hidden_size"]
        self.num_layers = self.config["lstm_num_layers"]
        self.max_seq_length = self.config["max_seq_length"]
        
        # Special token IDs
        self.pad_id = self.config["pad_id"]
        self.bos_id = self.config["bos_id"]
        self.eos_id = self.config["eos_id"]
        self.sep_id = self.config["sep_id"]
        
        # ─────────────────────────────────────────────────────────────────────
        # Layer 1: Token Embedding
        # Converts token IDs (0-35) to dense vectors (64 dims)
        # ─────────────────────────────────────────────────────────────────────
        self.token_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.token_dim,
            padding_idx=self.pad_id  # Padding tokens get zero embedding
        )
        
        # ─────────────────────────────────────────────────────────────────────
        # Layer 2: Feature Encoder
        # Converts (key, mode, genre, emotion, tempo) → conditioning [128]
        # ─────────────────────────────────────────────────────────────────────
        if feature_encoder is not None:
            self.feature_encoder = feature_encoder
        else:
            self.feature_encoder = FeatureEncoder(output_dim=self.cond_dim)
        
        # ─────────────────────────────────────────────────────────────────────
        # Layer 3: Conditioning → Initial Hidden State
        # Transforms 128-dim conditioning to 256-dim hidden state
        # ─────────────────────────────────────────────────────────────────────
        self.cond_to_hidden = nn.Sequential(
            nn.Linear(self.cond_dim, self.hidden_size),
            nn.Tanh()  # Tanh keeps values in [-1, 1], good for LSTM init
        )
        
        # Also need to initialize cell state (LSTM has both h and c)
        self.cond_to_cell = nn.Sequential(
            nn.Linear(self.cond_dim, self.hidden_size),
            nn.Tanh()
        )
        
        # ─────────────────────────────────────────────────────────────────────
        # Layer 4: LSTM
        # Input: token_embed (64) + conditioning (128) = 192
        # Hidden: 256 dims
        # Stacked: 2 layers
        # ─────────────────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=self.token_dim + self.cond_dim,  # 64 + 128 = 192
            hidden_size=self.hidden_size,               # 256
            num_layers=self.num_layers,                 # 2
            dropout=self.config["lstm_dropout"] if self.num_layers > 1 else 0,
            batch_first=True  # Input shape: [batch, seq, features]
        )
        
        # ─────────────────────────────────────────────────────────────────────
        # Layer 5: Output Projection
        # Converts LSTM hidden state (256) to vocabulary logits (36)
        # ─────────────────────────────────────────────────────────────────────
        self.output_projection = nn.Linear(self.hidden_size, self.vocab_size)
        
        # ─────────────────────────────────────────────────────────────────────
        # Dropout for regularization
        # ─────────────────────────────────────────────────────────────────────
        self.dropout = nn.Dropout(self.config["lstm_dropout"])
        
        # Initialize weights
        self._init_weights()
        
        # Create tokenizer for generation
        self.tokenizer = MusicTokenizer()
    
    def _init_weights(self):
        """
        Initialize model weights for stable training.
        
        Uses Xavier initialization for linear layers and orthogonal
        initialization for LSTM weights.
        """
        # Token embeddings: normal distribution
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.1)
        # Keep padding embedding at zero
        self.token_embedding.weight.data[self.pad_id].zero_()
        
        # Linear layers: Xavier
        for module in [self.cond_to_hidden, self.cond_to_cell]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        
        # Output projection: Xavier
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        
        # LSTM: orthogonal initialization (helps with gradient flow)
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
                # Set forget gate bias to 1 (helps remember long sequences)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
    
    def _init_hidden(
        self,
        conditioning: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize LSTM hidden state from conditioning vector.
        
        The conditioning vector (128 dims) is projected to create the
        initial hidden state h₀ (256 dims). This "primes" the LSTM with
        information about what kind of music to generate.
        
        Args:
            conditioning: Conditioning vector [batch, 128]
            
        Returns:
            Tuple of (h₀, c₀) each with shape [num_layers, batch, hidden_size]
        """
        batch_size = conditioning.size(0)
        
        # Project conditioning to hidden state dimension
        h = self.cond_to_hidden(conditioning)  # [batch, 256]
        c = self.cond_to_cell(conditioning)    # [batch, 256]
        
        # Reshape for LSTM: [num_layers, batch, hidden_size]
        # We repeat the same initialization for both LSTM layers
        h = h.unsqueeze(0).repeat(self.num_layers, 1, 1)  # [2, batch, 256]
        c = c.unsqueeze(0).repeat(self.num_layers, 1, 1)  # [2, batch, 256]
        
        return h, c
    
    def forward(
        self,
        input_ids: torch.Tensor,
        features: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with teacher forcing (for training).
        
        During training, we feed the model the correct previous tokens
        at each timestep. This is more stable than using the model's
        own predictions.
        
        Args:
            input_ids: Token IDs [batch, seq_len] - the target sequence
            features: Dictionary of feature tensors for conditioning
                      (key_idx, mode_idx, genre_idx, emotion_idx, tempo_idx)
            attention_mask: Mask for padding [batch, seq_len] (optional)
            return_hidden: If True, also return final hidden state
            
        Returns:
            logits: Unnormalized predictions [batch, seq_len, vocab_size]
            (optional) hidden: Final hidden state tuple (h, c)
            
        Example:
            >>> model = GuitarLSTM()
            >>> logits = model(
            ...     input_ids=batch["input_ids"],  # [16, 20]
            ...     features={"key_idx": batch["key_idx"], ...}
            ... )
            >>> logits.shape
            torch.Size([16, 20, 36])
        """
        batch_size, seq_len = input_ids.shape
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 1: Encode features → conditioning vector [batch, 128]
        # ─────────────────────────────────────────────────────────────────────
        conditioning = self.feature_encoder(features)
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 2: Initialize hidden state from conditioning
        # ─────────────────────────────────────────────────────────────────────
        h_0, c_0 = self._init_hidden(conditioning)
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 3: Embed input tokens [batch, seq_len] → [batch, seq_len, 64]
        # ─────────────────────────────────────────────────────────────────────
        token_embeds = self.token_embedding(input_ids)
        token_embeds = self.dropout(token_embeds)
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 4: Concatenate conditioning at every timestep
        # conditioning: [batch, 128] → [batch, seq_len, 128]
        # lstm_input: [batch, seq_len, 64+128] = [batch, seq_len, 192]
        # ─────────────────────────────────────────────────────────────────────
        conditioning_expanded = conditioning.unsqueeze(1).expand(-1, seq_len, -1)
        lstm_input = torch.cat([token_embeds, conditioning_expanded], dim=-1)
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 5: Run LSTM
        # lstm_output: [batch, seq_len, 256]
        # ─────────────────────────────────────────────────────────────────────
        lstm_output, (h_n, c_n) = self.lstm(lstm_input, (h_0, c_0))
        lstm_output = self.dropout(lstm_output)
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 6: Project to vocabulary logits
        # logits: [batch, seq_len, 36]
        # ─────────────────────────────────────────────────────────────────────
        logits = self.output_projection(lstm_output)
        
        if return_hidden:
            return logits, (h_n, c_n)
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
        
        Starting from <BOS>, the model generates one token at a time,
        using its own predictions as input for the next step.
        
        Args:
            features: Musical features (can be strings or tensor indices)
                - key: "G", "Am", etc. or index
                - mode: "major" or "minor" or index
                - genre: "folk", "rock", etc. or index
                - emotion: "upbeat", "melancholic", etc. or index
                - tempo: BPM (40-200) or bucket index
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top-k most likely tokens
            top_p: If set, use nucleus sampling with this threshold
            do_sample: If False, use greedy decoding (always pick most likely)
            
        Returns:
            Dictionary containing:
                - token_ids: List of generated token IDs
                - chords: List of chord strings
                - strum_pattern: Strumming pattern string
                - raw_sequence: Human-readable token sequence
                
        Example:
            >>> model = GuitarLSTM()
            >>> result = model.generate(
            ...     features={"key": "G", "mode": "major", "genre": "folk",
            ...               "emotion": "upbeat", "tempo": 110},
            ...     temperature=0.8
            ... )
            >>> result["chords"]
            ['G', 'D', 'Em', 'C']
            >>> result["strum_pattern"]
            'D_DU_DU_'
        """
        self.eval()  # Set to evaluation mode
        device = next(self.parameters()).device
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 1: Encode features → conditioning vector
        # ─────────────────────────────────────────────────────────────────────
        conditioning = self.feature_encoder(features, batch_size=1)
        if conditioning.device != device:
            conditioning = conditioning.to(device)
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 2: Initialize hidden state
        # ─────────────────────────────────────────────────────────────────────
        h, c = self._init_hidden(conditioning)
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 3: Start with <BOS> token
        # ─────────────────────────────────────────────────────────────────────
        current_token = torch.tensor([[self.bos_id]], device=device)
        generated_ids = [self.bos_id]
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 4: Autoregressive generation loop
        # ─────────────────────────────────────────────────────────────────────
        for _ in range(max_length - 1):  # -1 because we already have <BOS>
            # Embed current token
            token_embed = self.token_embedding(current_token)  # [1, 1, 64]
            
            # Concatenate with conditioning
            lstm_input = torch.cat([token_embed, conditioning.unsqueeze(1)], dim=-1)
            
            # Run one LSTM step
            lstm_output, (h, c) = self.lstm(lstm_input, (h, c))
            
            # Get logits for next token
            logits = self.output_projection(lstm_output[:, -1, :])  # [1, 36]
            
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
            
            # Prepare for next iteration
            current_token = torch.tensor([[next_token_id]], device=device)
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 5: Decode generated sequence
        # ─────────────────────────────────────────────────────────────────────
        chords, strum_pattern = self.tokenizer.decode(generated_ids)
        
        # Create human-readable sequence
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
        
        Supports multiple sampling strategies:
        - Greedy: Always pick the most likely token
        - Temperature: Scale logits before softmax
        - Top-k: Only consider the k most likely tokens
        - Top-p (nucleus): Only consider tokens with cumulative prob ≥ p
        
        Args:
            logits: Unnormalized scores [1, vocab_size]
            temperature: Sampling temperature (higher = more random)
            top_k: Number of top tokens to consider
            top_p: Cumulative probability threshold for nucleus sampling
            do_sample: If False, use greedy decoding
            
        Returns:
            Sampled token ID
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
            probs = probs / probs.sum()  # Renormalize
        
        # Top-p (nucleus) filtering
        if top_p is not None and top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Find cutoff index
            sorted_mask = cumulative_probs > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            
            # Zero out tokens beyond cutoff
            sorted_probs[sorted_mask] = 0
            probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)
            probs = probs / probs.sum()  # Renormalize
        
        # Sample from distribution
        token_id = torch.multinomial(probs, num_samples=1).item()
        
        return token_id
    
    def get_num_parameters(self) -> Dict[str, int]:
        """
        Count the number of parameters in each component.
        
        Returns:
            Dictionary with parameter counts
        """
        def count_params(module):
            return sum(p.numel() for p in module.parameters())
        
        return {
            "token_embedding": count_params(self.token_embedding),
            "feature_encoder": count_params(self.feature_encoder),
            "cond_to_hidden": count_params(self.cond_to_hidden),
            "cond_to_cell": count_params(self.cond_to_cell),
            "lstm": count_params(self.lstm),
            "output_projection": count_params(self.output_projection),
            "total": count_params(self)
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_model(config: Optional[Dict] = None) -> GuitarLSTM:
    """Create a GuitarLSTM model with the given config."""
    return GuitarLSTM(config=config)


def load_model(checkpoint_path: str, device: str = "cpu") -> GuitarLSTM:
    """
    Load a trained model from a checkpoint.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        device: Device to load the model on ("cpu" or "cuda")
        
    Returns:
        Loaded GuitarLSTM model in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model with saved config
    config = checkpoint.get("config", LSTM_CONFIG)
    model = GuitarLSTM(config=config)
    
    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model


# =============================================================================
# TESTING / DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    """
    Run this file directly to see the LSTM model in action:
        python -m src.models.lstm_model
    """
    print("=" * 70)
    print("GUITAR LSTM MODEL DEMONSTRATION")
    print("=" * 70)
    
    # Create model
    model = GuitarLSTM()
    print(f"\nModel created successfully!")
    
    # Count parameters
    param_counts = model.get_num_parameters()
    print("\nParameter counts:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")
    
    # Example 1: Forward pass (training mode)
    print("\n" + "-" * 70)
    print("EXAMPLE 1: Forward Pass (Training Mode)")
    print("-" * 70)
    
    batch_size = 4
    seq_len = 15
    
    # Create dummy batch
    dummy_input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
    dummy_features = {
        "key_idx": torch.randint(0, 12, (batch_size,)),
        "mode_idx": torch.randint(0, 2, (batch_size,)),
        "genre_idx": torch.randint(0, 9, (batch_size,)),
        "emotion_idx": torch.randint(0, 8, (batch_size,)),
        "tempo_idx": torch.randint(0, 10, (batch_size,)),
    }
    
    print(f"Input shape: {dummy_input_ids.shape}")
    
    # Forward pass
    logits = model(dummy_input_ids, dummy_features)
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: [{batch_size}, {seq_len}, {VOCAB_SIZE}]")
    
    # Example 2: Generation (inference mode)
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
    
    result = model.generate(
        features=features,
        max_length=20,
        temperature=0.8,
        do_sample=True
    )
    
    print(f"\nGenerated token IDs: {result['token_ids']}")
    print(f"Generated chords: {result['chords']}")
    print(f"Generated strum: {result['strum_pattern']}")
    print(f"Raw sequence: {result['raw_sequence']}")
    
    # Example 3: Greedy vs Temperature sampling
    print("\n" + "-" * 70)
    print("EXAMPLE 3: Sampling Strategies Comparison")
    print("-" * 70)
    
    print("\nGreedy (temperature=0, no randomness):")
    for i in range(3):
        result = model.generate(features, temperature=1.0, do_sample=False)
        print(f"  Run {i+1}: {result['chords']} | {result['strum_pattern']}")
    
    print("\nWith temperature=0.8 (some randomness):")
    for i in range(3):
        result = model.generate(features, temperature=0.8, do_sample=True)
        print(f"  Run {i+1}: {result['chords']} | {result['strum_pattern']}")
    
    print("\n" + "=" * 70)
    print("GUITAR LSTM MODEL DEMONSTRATION COMPLETE")
    print("=" * 70)
