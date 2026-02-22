# 🎸 Model Architecture Specification
## Guitar Chord & Strumming Pattern Generator

**Document Version:** 1.0  
**Phase:** Model Design (Architecture Only)  
**Date:** January 2026  
**Author:** Rohan Rajendra Dhanawade  
**Purpose:** Technical specification for model implementation

---

## 1. Executive Summary

This document specifies the neural architecture for generating symbolic guitar chord progressions and strumming patterns from natural language prompts. Two sequence models will be implemented and compared:

1. **LSTM Model** (Primary) — Simpler, more stable for small datasets
2. **Transformer Model** (Comparison) — Modern architecture for ablation study

Both models share the same tokenizer, feature encoder, and training pipeline.

---

## 2. System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           COMPLETE SYSTEM PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    USER PROMPT: "upbeat folk in G major"                                    │
│                         │                                                   │
│                         ▼                                                   │
│    ┌─────────────────────────────────────────┐                             │
│    │         DISTILBERT PARSER               │  ◄── Already built           │
│    │     (Neural Prompt Feature Extractor)   │                             │
│    └─────────────────────────────────────────┘                             │
│                         │                                                   │
│                         ▼                                                   │
│              ┌───────────────────┐                                         │
│              │  PROMPT FEATURES  │                                         │
│              │  ─────────────────│                                         │
│              │  key: "G"         │                                         │
│              │  mode: "major"    │                                         │
│              │  genre: "folk"    │                                         │
│              │  emotion: "upbeat"│                                         │
│              │  tempo: 110       │                                         │
│              └───────────────────┘                                         │
│                         │                                                   │
│                         ▼                                                   │
│    ┌─────────────────────────────────────────┐                             │
│    │         FEATURE ENCODER                 │  ◄── NEW                     │
│    │   (Embeddings → Conditioning Vector)    │                             │
│    └─────────────────────────────────────────┘                             │
│                         │                                                   │
│                         ▼                                                   │
│              ┌───────────────────┐                                         │
│              │ CONDITIONING      │                                         │
│              │ VECTOR (128 dims) │                                         │
│              └───────────────────┘                                         │
│                         │                                                   │
│          ┌──────────────┴──────────────┐                                   │
│          ▼                             ▼                                   │
│    ┌───────────┐                 ┌───────────┐                             │
│    │   LSTM    │                 │TRANSFORMER│  ◄── NEW                     │
│    │  MODEL    │                 │   MODEL   │                             │
│    └───────────┘                 └───────────┘                             │
│          │                             │                                   │
│          └──────────────┬──────────────┘                                   │
│                         ▼                                                   │
│    ┌─────────────────────────────────────────┐                             │
│    │            TOKENIZER                    │  ◄── NEW                     │
│    │      (Decode token IDs → text)          │                             │
│    └─────────────────────────────────────────┘                             │
│                         │                                                   │
│                         ▼                                                   │
│    ┌─────────────────────────────────────────┐                             │
│    │           VALIDATOR                     │  ◄── Already built           │
│    │   (Check harmonic correctness)          │                             │
│    └─────────────────────────────────────────┘                             │
│                         │                                                   │
│                         ▼                                                   │
│              ┌───────────────────┐                                         │
│              │   FINAL OUTPUT    │                                         │
│              │  ─────────────────│                                         │
│              │  chords: [G,D,Em,C]│                                        │
│              │  strum: "D_DU_DU_"│                                         │
│              └───────────────────┘                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Tokenization Specification

### 3.1 Vocabulary Design Decisions

| Element | Tokenization Strategy | Rationale |
|---------|----------------------|-----------|
| Chords | One token per chord | Chords are atomic musical units |
| Strumming | Character-by-character | D, U, _ are atomic rhythmic events |
| Sequence | Combined with `<SEP>` | Single model, simpler training |

### 3.2 Complete Vocabulary

```python
VOCABULARY = {
    # ═══════════════════════════════════════════════════════════
    # SPECIAL TOKENS (IDs 0-3)
    # ═══════════════════════════════════════════════════════════
    "<PAD>": 0,   # Padding token (fills sequences to equal length)
    "<BOS>": 1,   # Beginning of sequence
    "<EOS>": 2,   # End of sequence  
    "<SEP>": 3,   # Separator between chords and strumming
    
    # ═══════════════════════════════════════════════════════════
    # CHORD TOKENS (IDs 4-32) — 29 unique chords from dataset
    # ═══════════════════════════════════════════════════════════
    # Natural major chords
    "C": 4, "D": 5, "E": 6, "F": 7, "G": 8, "A": 9, "B": 10,
    
    # Sharp major chords
    "A#": 11, "C#": 12, "D#": 13, "G#": 14,
    
    # Minor chords
    "Am": 15, "Bm": 16, "Cm": 17, "Dm": 18, "Em": 19, "Fm": 20, "Gm": 21,
    "A#m": 22, "C#m": 23, "F#m": 24, "G#m": 25,
    
    # Seventh chords
    "A7": 26, "B7": 27, "D7": 28, "E7": 29,
    
    # Other chord types
    "Asus4": 30, "C#dim": 31, "Gdim": 32,
    
    # ═══════════════════════════════════════════════════════════
    # STRUMMING TOKENS (IDs 33-35) — 3 unique characters
    # ═══════════════════════════════════════════════════════════
    "D": 33,  # Downstroke
    "U": 34,  # Upstroke
    "_": 35,  # Rest (muted/silent)
}

VOCAB_SIZE = 36  # Total vocabulary size
```

### 3.3 Sequence Format

**Example:** Folk song in G major, upbeat

```
Raw output:
  chords: ["G", "D", "Em", "C"]
  strum:  "D_DU_DU_"

Tokenized sequence:
┌─────┬───┬───┬────┬───┬─────┬───┬───┬───┬───┬───┬───┬───┬───┬─────┐
│<BOS>│ G │ D │ Em │ C │<SEP>│ D │ _ │ D │ U │ _ │ D │ U │ _ │<EOS>│
└─────┴───┴───┴────┴───┴─────┴───┴───┴───┴───┴───┴───┴───┴───┴─────┘
│  1  │ 8 │ 5 │ 19 │ 4 │  3  │33 │35 │33 │34 │35 │33 │34 │35 │  2  │

Sequence length: 15 tokens (typical)
Max sequence length: 20 tokens (with padding)
  - 1 <BOS> + 8 chords (max) + 1 <SEP> + 8 strum + 1 <EOS> + 1 buffer = 20
```

### 3.4 Important Note: Token ID Collision

⚠️ **The chord "D" and strumming "D" use DIFFERENT token IDs!**

| Token | Context | ID | Meaning |
|-------|---------|-----|---------|
| D | Before `<SEP>` | 5 | D major chord |
| D | After `<SEP>` | 33 | Downstroke |

The model learns to distinguish them by position (before vs. after `<SEP>`).

---

## 4. Feature Encoding Specification

### 4.1 Input Features (from DistilBERT Parser)

| Feature | Type | Possible Values | Count |
|---------|------|-----------------|-------|
| key | categorical | A, Am, Bm, C, D, Dm, E, Em, F, Fm, G, Gm | 12 |
| mode | categorical | major, minor | 2 |
| genre | categorical | acoustic, ballad, blues, country, folk, indie, jazz, pop, rock | 9 |
| emotion | categorical | dramatic, energetic, hopeful, melancholic, mellow, nostalgic, peaceful, upbeat | 8 |
| tempo | numerical | 40-200 BPM | continuous |

### 4.2 Embedding Dimensions

```python
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
        "num_buckets": 10,  # Bucketized: [40-55], [56-70], ..., [186-200]
        "embedding_dim": 16
    }
}

TOTAL_CONDITIONING_DIM = 32 + 16 + 32 + 32 + 16 = 128
```

### 4.3 Tempo Bucketization

```python
TEMPO_BUCKETS = [
    (40, 55),    # Bucket 0: Very slow
    (56, 70),    # Bucket 1: Slow
    (71, 85),    # Bucket 2: Slow-moderate
    (86, 100),   # Bucket 3: Moderate
    (101, 115),  # Bucket 4: Moderate-fast
    (116, 130),  # Bucket 5: Fast
    (131, 145),  # Bucket 6: Fast-energetic
    (146, 160),  # Bucket 7: Very fast
    (161, 180),  # Bucket 8: Driving
    (181, 200),  # Bucket 9: Maximum energy
]
```

### 4.4 Feature Encoder Architecture

```
INPUT FEATURES                    EMBEDDINGS                         OUTPUT
───────────────                   ──────────                         ──────

key: "G" ─────────┐
                  │  ┌────────────────────┐
                  ├─▶│ Key Embedding      │──▶ [32 dims] ─┐
                     │ (12 × 32 lookup)   │               │
                     └────────────────────┘               │
                                                          │
mode: "major" ────┐                                       │
                  │  ┌────────────────────┐               │
                  ├─▶│ Mode Embedding     │──▶ [16 dims] ─┤
                     │ (2 × 16 lookup)    │               │
                     └────────────────────┘               │
                                                          │
genre: "folk" ────┐                                       ├──▶ CONCATENATE
                  │  ┌────────────────────┐               │         │
                  ├─▶│ Genre Embedding    │──▶ [32 dims] ─┤         │
                     │ (9 × 32 lookup)    │               │         ▼
                     └────────────────────┘               │    ┌─────────┐
                                                          │    │ Linear  │
emotion: "upbeat"─┐                                       │    │ 128→128 │
                  │  ┌────────────────────┐               │    │ + ReLU  │
                  ├─▶│ Emotion Embedding  │──▶ [32 dims] ─┤    └────┬────┘
                     │ (8 × 32 lookup)    │               │         │
                     └────────────────────┘               │         ▼
                                                          │   CONDITIONING
tempo: 110 ───────┐                                       │     VECTOR
      │           │  ┌────────────────────┐               │   [128 dims]
      │           │  │ Bucketize          │               │
      ▼           ├─▶│ (110 → bucket 4)   │               │
   bucket 4          │                    │               │
                     │ Tempo Embedding    │──▶ [16 dims] ─┘
                     │ (10 × 16 lookup)   │
                     └────────────────────┘
```

---

## 5. LSTM Model Architecture

### 5.1 Architecture Diagram

```
                    CONDITIONING VECTOR (128 dims)
                              │
            ┌─────────────────┴─────────────────┐
            │                                   │
            ▼                                   │
    ┌───────────────┐                           │
    │ Linear Layer  │                           │
    │  (128 → 256)  │                           │
    │    + tanh     │                           │
    └───────┬───────┘                           │
            │                                   │
            ▼                                   │
    Initial Hidden State (h₀)                   │
    [1, batch, 256]                             │
            │                                   │
            │     ┌─────────────────────────────┤
            │     │                             │
            ▼     ▼                             │
    ╔═══════════════════════════════════════╗   │
    ║           LSTM LAYER                  ║   │
    ║  ─────────────────────────────────    ║   │
    ║  input_size: 64 + 128 = 192           ║   │
    ║  hidden_size: 256                     ║   │
    ║  num_layers: 2                        ║   │
    ║  dropout: 0.2                         ║   │
    ║  batch_first: True                    ║   │
    ╚═══════════════════════════════════════╝   │
            │                                   │
            │  At each timestep:               │
            │  ┌──────────────────────────┐     │
            │  │ input = [token_embed]    │     │
            │  │       + [conditioning]   │◄────┘ (concatenated at every step)
            │  │       = [64 + 128]       │
            │  │       = [192 dims]       │
            │  └──────────────────────────┘
            │
            ▼
    ┌───────────────┐
    │ Output Layer  │
    │  (256 → 36)   │  ◄── 36 = vocab size
    │   + softmax   │
    └───────────────┘
            │
            ▼
    Token Probabilities
    [batch, seq_len, 36]
```

### 5.2 LSTM Hyperparameters

```python
LSTM_CONFIG = {
    # Token embedding
    "vocab_size": 36,
    "token_embedding_dim": 64,
    
    # Conditioning
    "conditioning_dim": 128,
    
    # LSTM architecture
    "lstm_input_size": 64 + 128,  # token_embed + conditioning = 192
    "lstm_hidden_size": 256,
    "lstm_num_layers": 2,
    "lstm_dropout": 0.2,
    
    # Output
    "output_size": 36,  # vocab_size
    
    # Sequence
    "max_seq_length": 20,
}
```

### 5.3 LSTM Forward Pass (Pseudocode)

```python
def forward(self, prompt_features, target_sequence=None):
    """
    Args:
        prompt_features: dict with keys: key, mode, genre, emotion, tempo
        target_sequence: [batch, seq_len] - for teacher forcing during training
    
    Returns:
        logits: [batch, seq_len, vocab_size]
    """
    # 1. Encode features → conditioning vector
    conditioning = self.feature_encoder(prompt_features)  # [batch, 128]
    
    # 2. Initialize hidden state from conditioning
    h0 = self.cond_to_hidden(conditioning)  # [batch, 256]
    h0 = h0.unsqueeze(0).repeat(2, 1, 1)    # [2, batch, 256] for 2 layers
    c0 = torch.zeros_like(h0)               # Cell state starts at zero
    
    # 3. Autoregressive generation
    if target_sequence is not None:
        # TRAINING: Teacher forcing
        token_embeds = self.token_embedding(target_sequence)  # [batch, seq, 64]
        
        # Concatenate conditioning at every step
        conditioning_expanded = conditioning.unsqueeze(1).expand(-1, seq_len, -1)
        lstm_input = torch.cat([token_embeds, conditioning_expanded], dim=-1)
        
        lstm_output, _ = self.lstm(lstm_input, (h0, c0))
        logits = self.output_layer(lstm_output)
    else:
        # INFERENCE: Generate one token at a time
        logits = self.generate_autoregressive(conditioning, h0, c0)
    
    return logits
```

---

## 6. Transformer Model Architecture

### 6.1 Key Difference: Conditioning via Prefix Tokens

Instead of hidden state initialization, the Transformer uses **prefix tokens**:

```
LSTM approach:
  conditioning → hidden state h₀
  
Transformer approach:
  conditioning → special prefix tokens prepended to sequence
  
Sequence with prefix:
┌──────┬────────┬───────┬───────────┬─────┬───┬───┬────┬───┬─────┬─────────────┬─────┐
│<FOLK>│<UPBEAT>│<G_MAJ>│<TEMPO_110>│<BOS>│ G │ D │ Em │ C │<SEP>│ D _ D U ... │<EOS>│
└──────┴────────┴───────┴───────────┴─────┴───┴───┴────┴───┴─────┴─────────────┴─────┘
   ↑        ↑        ↑         ↑
   └────────┴────────┴─────────┘
         CONDITIONING PREFIX
         (model attends to these)
```

### 6.2 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TRANSFORMER DECODER-ONLY                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input: [<FOLK>, <UPBEAT>, <G_MAJ>, <TEMPO_110>, <BOS>, G, D, Em, ...]    │
│                              │                                              │
│                              ▼                                              │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │                    TOKEN EMBEDDING                                │     │
│   │                  (vocab_size × 64)                                │     │
│   │                                                                   │     │
│   │   Note: Conditioning features get their OWN embedding tokens:    │     │
│   │   <FOLK> = token 37, <UPBEAT> = token 45, etc.                   │     │
│   │   (extends vocabulary from 36 to ~65)                            │     │
│   └──────────────────────────────────────────────────────────────────┘     │
│                              │                                              │
│                              ▼                                              │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │                POSITIONAL ENCODING                                │     │
│   │              (sinusoidal or learned)                              │     │
│   └──────────────────────────────────────────────────────────────────┘     │
│                              │                                              │
│                              ▼                                              │
│   ╔══════════════════════════════════════════════════════════════════╗     │
│   ║              TRANSFORMER DECODER BLOCK (×4)                       ║     │
│   ║  ┌────────────────────────────────────────────────────────────┐  ║     │
│   ║  │            MASKED SELF-ATTENTION                           │  ║     │
│   ║  │  • 4 attention heads                                       │  ║     │
│   ║  │  • Each position attends to previous positions only        │  ║     │
│   ║  │  • Conditioning tokens are ALWAYS visible                  │  ║     │
│   ║  └────────────────────────────────────────────────────────────┘  ║     │
│   ║                              │                                    ║     │
│   ║                              ▼                                    ║     │
│   ║  ┌────────────────────────────────────────────────────────────┐  ║     │
│   ║  │            FEED-FORWARD NETWORK                            │  ║     │
│   ║  │  Linear(256 → 512) → ReLU → Linear(512 → 256)             │  ║     │
│   ║  └────────────────────────────────────────────────────────────┘  ║     │
│   ║                              │                                    ║     │
│   ║              (Layer Norm + Residual at each step)                ║     │
│   ╚══════════════════════════════════════════════════════════════════╝     │
│                              │                                              │
│                              ▼                                              │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │                    OUTPUT PROJECTION                              │     │
│   │                    (256 → vocab_size)                             │     │
│   └──────────────────────────────────────────────────────────────────┘     │
│                              │                                              │
│                              ▼                                              │
│                     Token Probabilities                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Transformer Hyperparameters

```python
TRANSFORMER_CONFIG = {
    # Token embedding (extended vocab for conditioning tokens)
    "vocab_size": 36 + 29,  # 36 base + 29 conditioning tokens = 65
    "token_embedding_dim": 256,
    
    # Transformer architecture
    "num_layers": 4,
    "num_heads": 4,
    "d_model": 256,
    "d_ff": 512,  # Feed-forward hidden dimension
    "dropout": 0.1,
    
    # Sequence
    "max_seq_length": 25,  # 20 + up to 5 conditioning prefix tokens
}

# Additional conditioning tokens for Transformer
CONDITIONING_TOKENS = {
    # Genre tokens (IDs 36-44)
    "<ACOUSTIC>": 36, "<BALLAD>": 37, "<BLUES>": 38, "<COUNTRY>": 39,
    "<FOLK>": 40, "<INDIE>": 41, "<JAZZ>": 42, "<POP>": 43, "<ROCK>": 44,
    
    # Emotion tokens (IDs 45-52)
    "<DRAMATIC>": 45, "<ENERGETIC>": 46, "<HOPEFUL>": 47, "<MELANCHOLIC>": 48,
    "<MELLOW>": 49, "<NOSTALGIC>": 50, "<PEACEFUL>": 51, "<UPBEAT>": 52,
    
    # Key tokens (IDs 53-64)
    "<KEY_A>": 53, "<KEY_Am>": 54, "<KEY_Bm>": 55, "<KEY_C>": 56,
    "<KEY_D>": 57, "<KEY_Dm>": 58, "<KEY_E>": 59, "<KEY_Em>": 60,
    "<KEY_F>": 61, "<KEY_Fm>": 62, "<KEY_G>": 63, "<KEY_Gm>": 64,
}
```

---

## 7. Training Specification

### 7.1 Training Configuration

```python
TRAINING_CONFIG = {
    # Data
    "train_samples": 129,
    "val_samples": 27,
    "test_samples": 29,
    "batch_size": 16,
    
    # Optimization
    "optimizer": "AdamW",
    "learning_rate": 5e-4,
    "weight_decay": 0.01,
    "max_epochs": 100,
    
    # Learning rate schedule
    "scheduler": "CosineAnnealingLR",
    "warmup_epochs": 5,
    
    # Early stopping
    "patience": 15,
    "min_delta": 0.001,
    
    # Loss
    "loss_function": "CrossEntropyLoss",
    "label_smoothing": 0.1,
    
    # Regularization
    "dropout": 0.2,  # LSTM
    "gradient_clip": 1.0,
}
```

### 7.2 Training Loop (High-Level)

```
For each epoch:
    ┌─────────────────────────────────────────────────────────────────┐
    │ TRAINING PHASE                                                  │
    ├─────────────────────────────────────────────────────────────────┤
    │ For each batch:                                                 │
    │   1. Extract prompt features from batch                         │
    │   2. Get target sequences (with teacher forcing)                │
    │   3. Forward pass → logits                                      │
    │   4. Compute cross-entropy loss                                 │
    │   5. Backward pass                                              │
    │   6. Gradient clipping                                          │
    │   7. Optimizer step                                             │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │ VALIDATION PHASE                                                │
    ├─────────────────────────────────────────────────────────────────┤
    │ For each batch:                                                 │
    │   1. Forward pass (no gradient)                                 │
    │   2. Compute validation loss                                    │
    │   3. Generate samples (autoregressive)                          │
    │   4. Compute accuracy metrics                                   │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │ CHECKPOINTING                                                   │
    ├─────────────────────────────────────────────────────────────────┤
    │ If val_loss improved:                                           │
    │   - Save model checkpoint                                       │
    │   - Reset patience counter                                      │
    │ Else:                                                           │
    │   - Increment patience counter                                  │
    │   - If patience exhausted → stop training                       │
    └─────────────────────────────────────────────────────────────────┘
```

### 7.3 Teacher Forcing

During training, we use **teacher forcing** — feeding the ground truth tokens as input, rather than the model's own predictions:

```
Without teacher forcing (slow, error propagates):
  <BOS> → model predicts "G" → use "G" as next input → model predicts "E" (wrong!)
                                                        → error propagates...

With teacher forcing (fast, stable):
  <BOS> → model predicts "G" → ignore prediction, use ground truth "G"
  "G"   → model predicts "D" → ignore prediction, use ground truth "D"
  "D"   → model predicts "Em" → ignore prediction, use ground truth "Em"
  ...
  
  All predictions are compared to ground truth in parallel
  Loss = CrossEntropy(predictions, ground_truth)
```

---

## 8. Inference Specification

### 8.1 Autoregressive Generation

During inference, we generate one token at a time:

```
Step 1: Input = [<BOS>]
        Model predicts → distribution over vocab
        Sample or argmax → "G"
        
Step 2: Input = [<BOS>, G]
        Model predicts → distribution over vocab
        Sample or argmax → "D"
        
Step 3: Input = [<BOS>, G, D]
        Model predicts → distribution over vocab
        Sample or argmax → "Em"
        
... continue until <EOS> or max_length ...

Final output: [<BOS>, G, D, Em, C, <SEP>, D, _, D, U, _, D, U, _, <EOS>]
```

### 8.2 Sampling Strategies

```python
INFERENCE_CONFIG = {
    # Greedy decoding (deterministic)
    "greedy": {
        "description": "Always pick highest probability token",
        "use_case": "Reproducible outputs, evaluation"
    },
    
    # Temperature sampling (stochastic)
    "temperature": {
        "temperature": 0.8,  # Lower = more focused, Higher = more random
        "description": "Scale logits before softmax",
        "use_case": "Creative variety"
    },
    
    # Top-k sampling
    "top_k": {
        "k": 10,
        "description": "Sample from top-k most likely tokens",
        "use_case": "Balanced creativity/quality"
    },
    
    # Top-p (nucleus) sampling
    "top_p": {
        "p": 0.9,
        "description": "Sample from smallest set with cumulative prob ≥ p",
        "use_case": "Dynamic vocabulary restriction"
    },
}
```

---

## 9. File Structure

```
src/
├── models/
│   ├── __init__.py
│   ├── tokenizer.py          # NEW: Vocabulary, encode/decode
│   ├── feature_encoder.py    # NEW: Prompt features → conditioning vector
│   ├── lstm_model.py         # NEW: LSTM sequence model
│   ├── transformer_model.py  # NEW: Transformer sequence model
│   ├── prompt_parser.py      # EXISTING: Rule-based parser
│   ├── neural_parser.py      # EXISTING: DistilBERT parser
│   └── inference.py          # EXISTING: Will be extended
│
├── train/
│   ├── __init__.py
│   ├── trainer.py            # NEW: Training loop, checkpointing
│   ├── dataset.py            # NEW: PyTorch Dataset class
│   └── metrics.py            # NEW: Training metrics
│
└── configs/
    └── model_config.yaml     # NEW: All hyperparameters

notebooks/
├── 03_train_lstm.ipynb       # NEW: LSTM training notebook
└── 04_train_transformer.ipynb # NEW: Transformer training notebook
```

---

## 10. Evaluation Plan

### 10.1 Metrics to Implement

| Metric | What It Measures | How |
|--------|------------------|-----|
| **Chord Accuracy** | % chords in correct key | Compare to diatonic chords |
| **Progression Validity** | Musical sensibility | Check against known patterns |
| **Strum Pattern Validity** | Correct format | 8 chars, only D/U/_ |
| **Diversity** | Output variety | Unique outputs / total |
| **Perplexity** | Model confidence | exp(avg loss) |

### 10.2 Comparison Framework

```
┌────────────────────────────────────────────────────────────────┐
│                    ABLATION STUDY PLAN                         │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Experiment 1: Rule-based vs LSTM vs Transformer              │
│  ─────────────────────────────────────────────────            │
│  • Same test set                                               │
│  • Same prompt features                                        │
│  • Compare all metrics                                         │
│                                                                │
│  Experiment 2: Conditioning ablation                           │
│  ─────────────────────────────────────────────────            │
│  • Full conditioning vs partial vs none                        │
│  • Which features matter most?                                 │
│                                                                │
│  Experiment 3: Model size ablation                             │
│  ─────────────────────────────────────────────────            │
│  • LSTM: 1 layer vs 2 layers                                   │
│  • Transformer: 2 layers vs 4 layers                           │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 11. Summary of Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Models | LSTM + Transformer | Ablation study for thesis |
| Implementation order | LSTM first | Simpler to debug |
| Strumming tokenization | Character-by-character | D, U, _ are atomic |
| Chord tokenization | One token per chord | Chords are atomic |
| Sequence format | Combined with `<SEP>` | Single model |
| Feature encoding | Learned embeddings | Captures relationships |
| LSTM conditioning | Initial h₀ + concatenation | Maximum info flow |
| Transformer conditioning | Prefix tokens | Natural for attention |
| Training | Teacher forcing + early stopping | Stable training |
| Inference | Temperature/top-k sampling | Creative variety |

---

## 12. Next Steps

1. **Implement Tokenizer** (`src/models/tokenizer.py`)
2. **Implement Feature Encoder** (`src/models/feature_encoder.py`)
3. **Implement LSTM Model** (`src/models/lstm_model.py`)
4. **Create Training Dataset** (`src/train/dataset.py`)
5. **Implement Training Loop** (`src/train/trainer.py`)
6. **Train LSTM on Colab** (`notebooks/03_train_lstm.ipynb`)
7. **Implement Transformer Model** (`src/models/transformer_model.py`)
8. **Train Transformer** (`notebooks/04_train_transformer.ipynb`)
9. **Compare Results**

---

**Document Complete** ✓

*This specification guides the model implementation work.*
