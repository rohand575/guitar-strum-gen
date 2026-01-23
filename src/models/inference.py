"""
Neural Parser Inference Script
==============================
Guitar Strum Generator - Thesis Project
Author: Rohan Rajendra Dhanawade

This script shows how to LOAD and USE the trained DistilBERT model
after you've trained it in Google Colab.

SETUP:
1. Train the model in Colab using notebooks/05_train_neural_parser.ipynb
2. Download neural_parser_checkpoint.pt from Colab
3. Put it in: checkpoints/neural_parser_checkpoint.pt
4. Run this script!

Usage:
    python -m src.models.inference
    
    # Or in your code:
    from src.models.inference import NeuralParserInference
    parser = NeuralParserInference('checkpoints/neural_parser_checkpoint.pt')
    result = parser.predict("moody atmospheric indie track")
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Tuple, Optional
import json

# These imports will work when torch is installed
try:
    from transformers import DistilBertTokenizer, DistilBertModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch/Transformers not installed. Install with:")
    print("   pip install torch transformers")


# =============================================================================
# CONSTANTS (must match training!)
# =============================================================================

VALID_EMOTIONS = [
    'upbeat', 'melancholic', 'mellow', 'energetic',
    'peaceful', 'dramatic', 'hopeful', 'nostalgic'
]

VALID_GENRES = [
    'pop', 'rock', 'folk', 'ballad', 'country',
    'blues', 'jazz', 'indie', 'acoustic'
]

EMOTION_TO_IDX = {e: i for i, e in enumerate(VALID_EMOTIONS)}
IDX_TO_EMOTION = {i: e for e, i in EMOTION_TO_IDX.items()}

GENRE_TO_IDX = {g: i for i, g in enumerate(VALID_GENRES)}
IDX_TO_GENRE = {i: g for g, i in GENRE_TO_IDX.items()}


# =============================================================================
# MODEL DEFINITION (same as training)
# =============================================================================

class DistilBertPromptParser(nn.Module):
    """
    The neural model architecture.
    Must match exactly what was used during training!
    """
    
    def __init__(self, num_emotions: int = 8, num_genres: int = 9, dropout_rate: float = 0.3):
        super().__init__()
        
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        hidden_size = self.bert.config.hidden_size  # 768
        
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_emotions)
        )
        
        self.genre_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_genres)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        emotion_logits = self.emotion_classifier(cls_embedding)
        genre_logits = self.genre_classifier(cls_embedding)
        
        return emotion_logits, genre_logits


# =============================================================================
# INFERENCE CLASS (what you use in your system)
# =============================================================================

class NeuralParserInference:
    """
    Easy-to-use class for making predictions with the trained model.
    
    Example:
        >>> parser = NeuralParserInference('checkpoints/neural_parser_checkpoint.pt')
        >>> result = parser.predict("moody atmospheric indie track")
        >>> print(result)
        {
            'emotion': 'melancholic',
            'emotion_confidence': 0.89,
            'genre': 'indie', 
            'genre_confidence': 0.92
        }
    """
    
    def __init__(self, checkpoint_path: str, device: str = None):
        """
        Load the trained model from checkpoint.
        
        Args:
            checkpoint_path: Path to neural_parser_checkpoint.pt
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and Transformers required. Install with: pip install torch transformers")
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        print("Loading DistilBERT tokenizer...")
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Load model architecture
        print("Loading model architecture...")
        self.model = DistilBertPromptParser()
        
        # Load trained weights from checkpoint
        print(f"Loading trained weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully!")
        print(f"   Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def predict(self, prompt: str) -> Dict:
        """
        Make prediction on a single prompt.
        
        Args:
            prompt: Natural language input like "moody indie track"
        
        Returns:
            Dictionary with predictions and confidence scores
        """
        # Tokenize the prompt
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Make prediction (no gradient computation needed)
        with torch.no_grad():
            emotion_logits, genre_logits = self.model(input_ids, attention_mask)
            
            # Convert to probabilities
            emotion_probs = torch.softmax(emotion_logits, dim=-1)
            genre_probs = torch.softmax(genre_logits, dim=-1)
            
            # Get predictions
            emotion_idx = torch.argmax(emotion_probs, dim=-1).item()
            genre_idx = torch.argmax(genre_probs, dim=-1).item()
        
        return {
            'prompt': prompt,
            'emotion': IDX_TO_EMOTION[emotion_idx],
            'emotion_confidence': emotion_probs[0, emotion_idx].item(),
            'genre': IDX_TO_GENRE[genre_idx],
            'genre_confidence': genre_probs[0, genre_idx].item(),
            'all_emotion_probs': {
                IDX_TO_EMOTION[i]: emotion_probs[0, i].item() 
                for i in range(len(VALID_EMOTIONS))
            },
            'all_genre_probs': {
                IDX_TO_GENRE[i]: genre_probs[0, i].item()
                for i in range(len(VALID_GENRES))
            }
        }
    
    def predict_batch(self, prompts: list) -> list:
        """
        Make predictions on multiple prompts efficiently.
        
        Args:
            prompts: List of prompt strings
        
        Returns:
            List of prediction dictionaries
        """
        return [self.predict(p) for p in prompts]


# =============================================================================
# INTEGRATION WITH HYBRID PARSER
# =============================================================================

def create_hybrid_parser(checkpoint_path: str = None):
    """
    Create a hybrid parser that combines neural + rule-based.
    
    This is what you use in your final system!
    
    Args:
        checkpoint_path: Path to trained model (optional)
    
    Returns:
        HybridPromptParser instance
    """
    from .neural_parser import HybridPromptParser
    return HybridPromptParser(model_path=checkpoint_path)


# =============================================================================
# DEMO / TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("NEURAL PARSER INFERENCE DEMO")
    print("=" * 70)
    
    # Check for checkpoint file
    checkpoint_path = Path("checkpoints/neural_parser_checkpoint.pt")
    
    if not checkpoint_path.exists():
        print(f"\nCheckpoint not found at: {checkpoint_path}")
        print("\nTo use this script:")
        print("1. Train the model in Google Colab")
        print("   - Upload notebooks/05_train_neural_parser.ipynb to Colab")
        print("   - Upload your train.jsonl dataset")
        print("   - Run all cells")
        print("   - Download neural_parser_checkpoint.pt")
        print("")
        print("2. Put the checkpoint in the right place:")
        print(f"   - Create folder: mkdir -p checkpoints")
        print(f"   - Move file: mv ~/Downloads/neural_parser_checkpoint.pt checkpoints/")
        print("")
        print("3. Run this script again!")
        
        # Create the checkpoints directory
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nCreated {checkpoint_path.parent}/ directory for you")
        
    else:
        print(f"\nFound checkpoint at: {checkpoint_path}")
        
        if TORCH_AVAILABLE:
            # Load the model
            parser = NeuralParserInference(str(checkpoint_path))
            
            # Test prompts
            test_prompts = [
                "sad folk song in E minor",
                "upbeat pop in G major",
                "moody atmospheric vibes",  # Ambiguous!
                "something chill for the evening",  # Very ambiguous!
                "driving rock anthem",
                "peaceful morning acoustic",
            ]
            
            print("\n" + "=" * 70)
            print("PREDICTIONS")
            print("=" * 70)
            
            for prompt in test_prompts:
                result = parser.predict(prompt)
                print(f"\n\"{prompt}\"")
                print(f"   Emotion: {result['emotion']} ({result['emotion_confidence']:.1%})")
                print(f"   Genre: {result['genre']} ({result['genre_confidence']:.1%})")
        else:
            print("\nWarning: Cannot run inference - PyTorch not installed")
            print("   Install with: pip install torch transformers")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
