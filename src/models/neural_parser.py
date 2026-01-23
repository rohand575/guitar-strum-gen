"""
Neural Prompt Parser using DistilBERT
=====================================
Guitar Strum Generator - Thesis Project
Author: Rohan Rajendra Dhanawade

This module uses DistilBERT (a transformer-based neural network) to extract
musical features from natural language prompts. Unlike the rule-based parser
which uses if-else logic, this uses ACTUAL machine learning:

1. DistilBERT Encoder: Converts text into 768-dimensional embeddings
2. Classification Heads: Neural networks that predict emotion/genre from embeddings
3. Fine-tuning: We train the model on YOUR dataset to learn music-specific patterns

Architecture:
    ┌─────────────┐
    │ User Prompt │ "something moody for a rainy evening"
    └──────┬──────┘
           │
           ▼
    ┌─────────────────┐
    │   DistilBERT    │  Pre-trained transformer (66M parameters)
    │    Encoder      │  Converts text → 768-dim vector
    └──────┬──────────┘
           │
           ▼
    ┌─────────────────┐
    │  [CLS] Token    │  768-dimensional representation of entire prompt
    │   Embedding     │  Captures semantic meaning
    └──────┬──────────┘
           │
     ┌─────┴─────┐
     │           │
     ▼           ▼
┌─────────┐ ┌─────────┐
│ Emotion │ │  Genre  │
│ Head    │ │  Head   │
│ (MLP)   │ │ (MLP)   │
└────┬────┘ └────┬────┘
     │           │
     ▼           ▼
┌─────────┐ ┌─────────┐
│melanchol│ │  indie  │
│  ic     │ │         │
└─────────┘ └─────────┘

This is REAL AI/ML - the model LEARNS patterns from your training data!
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import numpy as np

# Import our existing structures
from .prompt_features import (
    PromptFeatures,
    ExtractionConfidence,
    VALID_GENRES,
    VALID_EMOTIONS,
    DEFAULT_FEATURES
)


# =============================================================================
# LABEL MAPPINGS
# =============================================================================

# Convert string labels to integers for classification
EMOTION_TO_IDX = {emotion: idx for idx, emotion in enumerate(VALID_EMOTIONS)}
IDX_TO_EMOTION = {idx: emotion for emotion, idx in EMOTION_TO_IDX.items()}

GENRE_TO_IDX = {genre: idx for idx, genre in enumerate(VALID_GENRES)}
IDX_TO_GENRE = {idx: genre for genre, idx in GENRE_TO_IDX.items()}

NUM_EMOTIONS = len(VALID_EMOTIONS)  # 8
NUM_GENRES = len(VALID_GENRES)      # 9


# =============================================================================
# DATASET CLASS FOR TRAINING
# =============================================================================

class PromptDataset(Dataset):
    """
    PyTorch Dataset for training the neural prompt parser.
    
    This converts your JSONL data into tensors that DistilBERT can process.
    Each sample becomes:
    - input_ids: Tokenized prompt (integers representing words/subwords)
    - attention_mask: Which tokens are real vs padding
    - emotion_label: Integer label for emotion (0-7)
    - genre_label: Integer label for genre (0-8)
    
    Example:
        >>> dataset = PromptDataset('data/processed/train.jsonl', tokenizer)
        >>> sample = dataset[0]
        >>> print(sample['input_ids'].shape)  # torch.Size([128])
    """
    
    def __init__(
        self, 
        data_path: str, 
        tokenizer: DistilBertTokenizer,
        max_length: int = 128
    ):
        """
        Initialize dataset from JSONL file.
        
        Args:
            data_path: Path to JSONL file with training samples
            tokenizer: DistilBERT tokenizer for converting text to tokens
            max_length: Maximum sequence length (longer prompts get truncated)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        # Load data from JSONL
        with open(data_path, 'r') as f:
            for line in f:
                sample = json.loads(line)
                self.samples.append(sample)
        
        print(f"Loaded {len(self.samples)} samples from {data_path}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training sample.
        
        Returns dictionary with:
        - input_ids: Token IDs for the prompt
        - attention_mask: 1 for real tokens, 0 for padding
        - emotion_label: Integer class label
        - genre_label: Integer class label
        """
        sample = self.samples[idx]
        
        # Tokenize the prompt using DistilBERT tokenizer
        # This converts "sad folk song" → [101, 6874, 2775, 2299, 102, 0, 0, ...]
        encoding = self.tokenizer(
            sample['prompt'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Get labels (convert string → integer)
        emotion_label = EMOTION_TO_IDX.get(
            sample.get('emotion', 'mellow').lower(), 
            EMOTION_TO_IDX['mellow']
        )
        genre_label = GENRE_TO_IDX.get(
            sample.get('genre', 'pop').lower(),
            GENRE_TO_IDX['pop']
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),      # [128]
            'attention_mask': encoding['attention_mask'].squeeze(0),  # [128]
            'emotion_label': torch.tensor(emotion_label, dtype=torch.long),
            'genre_label': torch.tensor(genre_label, dtype=torch.long)
        }


# =============================================================================
# NEURAL MODEL DEFINITION
# =============================================================================

class DistilBertPromptParser(nn.Module):
    """
    Neural Prompt Parser using DistilBERT.
    
    This is a REAL neural network with:
    - 66 million parameters in DistilBERT encoder
    - Classification heads for emotion and genre
    - Trainable on YOUR dataset
    
    Architecture:
        Input Text
            │
            ▼
        ┌──────────────────────────────────────┐
        │         DistilBERT Encoder           │
        │  (6 transformer layers, 768-dim)     │
        │  Pre-trained on Wikipedia/Books      │
        └──────────────────┬───────────────────┘
                           │
                    [CLS] embedding (768-dim)
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
        ┌───────────┐            ┌───────────┐
        │  Emotion  │            │   Genre   │
        │   Head    │            │   Head    │
        │           │            │           │
        │ 768→256   │            │ 768→256   │
        │  ReLU     │            │  ReLU     │
        │ Dropout   │            │ Dropout   │
        │ 256→8     │            │ 256→9     │
        └─────┬─────┘            └─────┬─────┘
              │                         │
              ▼                         ▼
        [8 emotions]              [9 genres]
         softmax                   softmax
    """
    
    def __init__(
        self, 
        num_emotions: int = NUM_EMOTIONS,
        num_genres: int = NUM_GENRES,
        dropout_rate: float = 0.3,
        freeze_bert: bool = False
    ):
        """
        Initialize the neural parser.
        
        Args:
            num_emotions: Number of emotion classes (8)
            num_genres: Number of genre classes (9)
            dropout_rate: Dropout for regularization (prevents overfitting)
            freeze_bert: If True, don't update DistilBERT weights (faster training)
        """
        super().__init__()
        
        # Load pre-trained DistilBERT
        # This is the "magic" - 66M parameters pre-trained on massive text!
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Optionally freeze BERT weights (use pre-trained only)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            print("DistilBERT weights frozen (using as feature extractor only)")
        else:
            print("DistilBERT weights trainable (fine-tuning mode)")
        
        # Get hidden size from BERT config
        hidden_size = self.bert.config.hidden_size  # 768
        
        # Emotion classification head (MLP: Multi-Layer Perceptron)
        # 768 → 256 → 8 (8 emotion classes)
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_emotions)
        )
        
        # Genre classification head
        # 768 → 256 → 9 (9 genre classes)
        self.genre_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_genres)
        )
        
        print(f"Model initialized:")
        print(f"  - DistilBERT: {sum(p.numel() for p in self.bert.parameters()):,} parameters")
        print(f"  - Emotion head: {sum(p.numel() for p in self.emotion_classifier.parameters()):,} parameters")
        print(f"  - Genre head: {sum(p.numel() for p in self.genre_classifier.parameters()):,} parameters")
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            input_ids: Tokenized input [batch_size, seq_length]
            attention_mask: Mask for padding [batch_size, seq_length]
        
        Returns:
            emotion_logits: Raw scores for each emotion class [batch_size, 8]
            genre_logits: Raw scores for each genre class [batch_size, 9]
        """
        # Pass through DistilBERT
        # Output shape: [batch_size, seq_length, 768]
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get the [CLS] token embedding (first token)
        # This represents the entire sequence
        # Shape: [batch_size, 768]
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Pass through classification heads
        emotion_logits = self.emotion_classifier(cls_embedding)  # [batch_size, 8]
        genre_logits = self.genre_classifier(cls_embedding)      # [batch_size, 9]
        
        return emotion_logits, genre_logits
    
    def predict(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions with probabilities.
        
        Returns:
            emotion_pred: Predicted emotion indices
            emotion_probs: Probability distribution over emotions
            genre_pred: Predicted genre indices
            genre_probs: Probability distribution over genres
        """
        self.eval()
        with torch.no_grad():
            emotion_logits, genre_logits = self.forward(input_ids, attention_mask)
            
            # Convert logits to probabilities using softmax
            emotion_probs = torch.softmax(emotion_logits, dim=-1)
            genre_probs = torch.softmax(genre_logits, dim=-1)
            
            # Get predicted classes
            emotion_pred = torch.argmax(emotion_probs, dim=-1)
            genre_pred = torch.argmax(genre_probs, dim=-1)
        
        return emotion_pred, emotion_probs, genre_pred, genre_probs


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_neural_parser(
    train_path: str,
    val_path: str = None,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    device: str = None,
    save_path: str = None
) -> DistilBertPromptParser:
    """
    Train the neural prompt parser on your dataset.
    
    This is where the LEARNING happens! The model:
    1. Reads prompts from your dataset
    2. Makes predictions (initially random)
    3. Compares to ground truth labels
    4. Adjusts weights to reduce error (backpropagation)
    5. Repeats until convergent
    
    Args:
        train_path: Path to training JSONL file
        val_path: Optional path to validation JSONL file
        epochs: Number of training epochs
        batch_size: Samples per batch
        learning_rate: How fast to update weights (2e-5 is good for BERT)
        device: 'cuda' or 'cpu'
        save_path: Where to save the trained model
    
    Returns:
        Trained model
    """
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")
    
    # Initialize tokenizer and model
    print("\nLoading DistilBERT tokenizer and model...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertPromptParser(freeze_bert=False)
    model.to(device)
    
    # Create datasets
    print("\nPreparing datasets...")
    train_dataset = PromptDataset(train_path, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if val_path:
        val_dataset = PromptDataset(val_path, tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Loss function (Cross-Entropy for classification)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer (AdamW is standard for transformers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_emotion = 0
        correct_genre = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            emotion_labels = batch['emotion_label'].to(device)
            genre_labels = batch['genre_label'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            emotion_logits, genre_logits = model(input_ids, attention_mask)
            
            # Calculate loss (sum of emotion and genre losses)
            emotion_loss = criterion(emotion_logits, emotion_labels)
            genre_loss = criterion(genre_logits, genre_labels)
            loss = emotion_loss + genre_loss
            
            # Backward pass (THIS IS WHERE LEARNING HAPPENS!)
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            
            # Calculate accuracy
            emotion_preds = torch.argmax(emotion_logits, dim=-1)
            genre_preds = torch.argmax(genre_logits, dim=-1)
            correct_emotion += (emotion_preds == emotion_labels).sum().item()
            correct_genre += (genre_preds == genre_labels).sum().item()
            total_samples += len(emotion_labels)
        
        # Epoch statistics
        avg_loss = total_loss / len(train_loader)
        emotion_acc = correct_emotion / total_samples * 100
        genre_acc = correct_genre / total_samples * 100
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Emotion Accuracy: {emotion_acc:.1f}%")
        print(f"  Genre Accuracy: {genre_acc:.1f}%")
        
        # Validation
        if val_path:
            model.eval()
            val_correct_emotion = 0
            val_correct_genre = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    emotion_labels = batch['emotion_label'].to(device)
                    genre_labels = batch['genre_label'].to(device)
                    
                    emotion_logits, genre_logits = model(input_ids, attention_mask)
                    
                    emotion_preds = torch.argmax(emotion_logits, dim=-1)
                    genre_preds = torch.argmax(genre_logits, dim=-1)
                    
                    val_correct_emotion += (emotion_preds == emotion_labels).sum().item()
                    val_correct_genre += (genre_preds == genre_labels).sum().item()
                    val_total += len(emotion_labels)
            
            val_emotion_acc = val_correct_emotion / val_total * 100
            val_genre_acc = val_correct_genre / val_total * 100
            print(f"  Val Emotion Acc: {val_emotion_acc:.1f}%")
            print(f"  Val Genre Acc: {val_genre_acc:.1f}%")
        
        print("-" * 40)
    
    # Save model
    if save_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'emotion_to_idx': EMOTION_TO_IDX,
            'genre_to_idx': GENRE_TO_IDX,
        }, save_path)
        print(f"\nModel saved to {save_path}")
    
    return model


# =============================================================================
# HYBRID NEURAL + RULE-BASED PARSER
# =============================================================================

class HybridPromptParser:
    """
    Hybrid parser combining Neural (DistilBERT) + Rule-based approaches.
    
    Strategy:
    1. Use RULES for explicit mentions (high confidence)
       - "in E minor" → key=E, mode=minor (regex)
       - "at 120 bpm" → tempo=120 (regex)
    
    2. Use NEURAL MODEL for implicit/ambiguous features
       - "moody dusk vibes" → DistilBERT predicts: emotion=melancholic
       - "something for driving" → DistilBERT predicts: genre=rock, emotion=energetic
    
    This gives you the BEST OF BOTH WORLDS:
    - Neural: Semantic understanding, handles ambiguity
    - Rules: Precision for explicit mentions, guaranteed format
    """
    
    def __init__(
        self,
        model_path: str = None,
        device: str = None
    ):
        """
        Initialize hybrid parser.
        
        Args:
            model_path: Path to trained DistilBERT model checkpoint
            device: 'cuda' or 'cpu'
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer
        print("Loading DistilBERT tokenizer...")
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Load neural model
        if model_path and Path(model_path).exists():
            print(f"Loading trained model from {model_path}...")
            self.model = DistilBertPromptParser()
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            self.neural_available = True
            print("Neural model loaded successfully!")
        else:
            print("No trained model found. Using untrained model (will need training).")
            self.model = DistilBertPromptParser()
            self.model.to(self.device)
            self.model.eval()
            self.neural_available = True
        
        # Import rule-based parser for explicit extractions
        from .prompt_parser import PromptParser
        self.rule_parser = PromptParser(verbose=False)
    
    def parse(self, prompt: str) -> PromptFeatures:
        """
        Parse prompt using hybrid neural + rule-based approach.
        
        Strategy:
        1. Run rule-based parser first (for explicit mentions)
        2. If emotion/genre confidence is low, use neural model
        3. Combine results with appropriate confidence scores
        """
        # Step 1: Rule-based extraction (for explicit mentions)
        rule_features = self.rule_parser.parse(prompt)
        
        # Step 2: Neural prediction (for semantic understanding)
        neural_emotion, neural_emotion_conf, neural_genre, neural_genre_conf = \
            self._neural_predict(prompt)
        
        # Step 3: Combine results (neural wins for low-confidence rule extractions)
        final_emotion = rule_features.emotion
        final_emotion_conf = rule_features.confidence.emotion
        
        final_genre = rule_features.genre
        final_genre_conf = rule_features.confidence.genre
        
        # Use neural if rule confidence is low (< 0.5)
        if rule_features.confidence.emotion < 0.5 and neural_emotion_conf > 0.5:
            final_emotion = neural_emotion
            final_emotion_conf = neural_emotion_conf
            rule_features.warnings.append(
                f"Emotion inferred by neural model: {neural_emotion} ({neural_emotion_conf:.0%})"
            )
        
        if rule_features.confidence.genre < 0.5 and neural_genre_conf > 0.5:
            final_genre = neural_genre
            final_genre_conf = neural_genre_conf
            rule_features.warnings.append(
                f"Genre inferred by neural model: {neural_genre} ({neural_genre_conf:.0%})"
            )
        
        # Build final features
        features = PromptFeatures(
            key=rule_features.key,
            mode=rule_features.mode,
            genre=final_genre,
            emotion=final_emotion,
            tempo=rule_features.tempo,
            time_signature=rule_features.time_signature,
            original_prompt=prompt,
            confidence=ExtractionConfidence(
                key=rule_features.confidence.key,
                mode=rule_features.confidence.mode,
                genre=final_genre_conf,
                emotion=final_emotion_conf,
                tempo=rule_features.confidence.tempo
            ),
            extracted_chords=rule_features.extracted_chords,
            warnings=rule_features.warnings,
            explicitly_stated=rule_features.explicitly_stated
        )
        
        return features
    
    def _neural_predict(self, prompt: str) -> Tuple[str, float, str, float]:
        """
        Get neural model predictions with confidence scores.
        
        Returns:
            (emotion, emotion_confidence, genre, genre_confidence)
        """
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        emotion_pred, emotion_probs, genre_pred, genre_probs = \
            self.model.predict(input_ids, attention_mask)
        
        # Get predictions and confidences
        emotion_idx = emotion_pred.item()
        emotion = IDX_TO_EMOTION[emotion_idx]
        emotion_conf = emotion_probs[0, emotion_idx].item()
        
        genre_idx = genre_pred.item()
        genre = IDX_TO_GENRE[genre_idx]
        genre_conf = genre_probs[0, genre_idx].item()
        
        return emotion, emotion_conf, genre, genre_conf


# =============================================================================
# DEMO / TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("NEURAL PROMPT PARSER DEMO")
    print("=" * 70)
    
    # Check if CUDA available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Test 1: Show model architecture
    print("\n" + "=" * 70)
    print("TEST 1: Model Architecture")
    print("=" * 70)
    
    model = DistilBertPromptParser()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"This is a REAL neural network with 66M+ parameters!")
    
    # Test 2: Tokenization demo
    print("\n" + "=" * 70)
    print("TEST 2: DistilBERT Tokenization")
    print("=" * 70)
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    test_prompt = "moody atmospheric indie track"
    tokens = tokenizer.tokenize(test_prompt)
    token_ids = tokenizer.encode(test_prompt)
    
    print(f"\nOriginal: '{test_prompt}'")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    print("\nNote: DistilBERT uses subword tokenization (WordPiece)")
    print("Unknown words are split into known subwords!")
    
    # Test 3: Forward pass (without training)
    print("\n" + "=" * 70)
    print("TEST 3: Forward Pass (Untrained Model)")
    print("=" * 70)
    
    encoding = tokenizer(
        test_prompt,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    
    model.eval()
    with torch.no_grad():
        emotion_logits, genre_logits = model(
            encoding['input_ids'],
            encoding['attention_mask']
        )
    
    emotion_probs = torch.softmax(emotion_logits, dim=-1)
    genre_probs = torch.softmax(genre_logits, dim=-1)
    
    print(f"\nPrompt: '{test_prompt}'")
    print(f"\nEmotion predictions (untrained, expect random):")
    for idx, prob in enumerate(emotion_probs[0]):
        print(f"  {IDX_TO_EMOTION[idx]}: {prob.item():.1%}")
    
    print(f"\nGenre predictions (untrained, expect random):")
    for idx, prob in enumerate(genre_probs[0]):
        print(f"  {IDX_TO_GENRE[idx]}: {prob.item():.1%}")
    
    print("\nNote: Model is UNTRAINED - predictions are random!")
    print("    After training on your dataset, predictions will be meaningful.")
    
    # Test 4: Training demo (if dataset exists)
    print("\n" + "=" * 70)
    print("TEST 4: Training Check")
    print("=" * 70)
    
    train_path = Path("data/processed/train.jsonl")
    if train_path.exists():
        print(f"\nTraining data found at {train_path}")
        print("  To train the model, run:")
        print("  >>> model = train_neural_parser('data/processed/train.jsonl', epochs=5)")
    else:
        print(f"\nTraining data not found at {train_path}")
        print("  Create your dataset first, then train!")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nThis is REAL AI/ML using:")
    print("  - DistilBERT: 66M parameter transformer")
    print("  - PyTorch: Industry-standard deep learning framework")
    print("  - Transfer Learning: Pre-trained on billions of words")
    print("  - Fine-tuning: Learns YOUR music-specific patterns")
