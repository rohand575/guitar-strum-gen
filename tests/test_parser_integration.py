"""
Integration Test: Prompt Parser → Generation Pipeline
======================================================

This script demonstrates how the prompt parser integrates with
the rest of the guitar strum generator system.

Usage:
    python tests/test_parser_integration.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.prompt_parser import PromptParser, parse_prompt
from src.models.prompt_features import PromptFeatures


def demonstrate_integration():
    """Show how the parser connects to the generation system."""
    
    print("=" * 70)
    print("INTEGRATION TEST: Prompt Parser → Generation System")
    print("=" * 70)
    
    # Create parser
    parser = PromptParser()
    
    # Test prompts representing different user scenarios
    test_cases = [
        # Full specification (high confidence)
        "Create a melancholic folk song in E minor with slow tempo",
        
        # Partial specification (medium confidence)
        "upbeat pop in G",
        
        # Minimal specification (low confidence - needs defaults)
        "something chill",
        
        # With explicit chords
        "use Am, G, C, F for a nostalgic indie track",
    ]
    
    for i, prompt in enumerate(test_cases, 1):
        print(f"\n{'─' * 70}")
        print(f"SCENARIO {i}: \"{prompt}\"")
        print("─" * 70)
        
        # Parse the prompt
        features = parser.parse(prompt)
        
        # Display extracted features
        print(f"\n📝 Extracted Features:")
        print(f"   Key: {features.full_key}")
        print(f"   Genre: {features.genre}")
        print(f"   Emotion: {features.emotion}")
        print(f"   Tempo: {features.tempo} BPM ({features.tempo_category})")
        
        if features.extracted_chords:
            print(f"   Explicit Chords: {' → '.join(features.extracted_chords)}")
        
        print(f"\n📊 Confidence Analysis:")
        print(f"   Overall: {features.confidence.overall():.2f}")
        print(f"   High confidence: {features.is_high_confidence}")
        
        # Show what was explicit vs defaulted
        print(f"\n🔍 What was detected vs defaulted:")
        for field, was_explicit in features.explicitly_stated.items():
            status = "✓ detected" if was_explicit else "○ defaulted"
            print(f"   {field}: {status}")
        
        # Show warnings if any
        if features.warnings:
            print(f"\n⚠️  Warnings:")
            for warning in features.warnings:
                print(f"   - {warning}")
        
        # Demonstrate decision logic for generation
        print(f"\n🎯 Generation Decision:")
        if features.is_high_confidence:
            print("   → Use NEURAL model (high confidence in inputs)")
        else:
            print("   → Consider RULE-BASED fallback (low confidence)")
            if features.confidence.key == 0.0:
                print("   → Could ASK user: 'What key would you like?'")
        
        # Show the features dict that would be passed to generator
        print(f"\n📦 Data for Generator:")
        features_dict = features.to_dict()
        print(f"   {{'key': '{features_dict['key']}', 'mode': '{features_dict['mode']}', ...")
    
    print("\n" + "=" * 70)
    print("INTEGRATION TEST COMPLETE")
    print("=" * 70)
    
    # Return features from last test for further testing
    return features


def test_parser_with_mock_generator():
    """
    Show how parser output would feed into the generator.
    
    This demonstrates the contract between parser and generator.
    """
    print("\n" + "=" * 70)
    print("MOCK GENERATOR INTEGRATION")
    print("=" * 70)
    
    # Parse a prompt
    features = parse_prompt("melancholic folk ballad in Am at 65 bpm")
    
    # This is what the generator would receive:
    generator_input = {
        'key': features.key,
        'mode': features.mode,
        'genre': features.genre,
        'emotion': features.emotion,
        'tempo': features.tempo,
        'time_signature': features.time_signature,
        'confidence': features.confidence.overall(),
        'use_neural': features.is_high_confidence
    }
    
    print("\nGenerator would receive:")
    for k, v in generator_input.items():
        print(f"  {k}: {v}")
    
    # Mock generation output
    print("\nMock generator output:")
    print("  Chords: Am → F → C → G")
    print("  Pattern: D_D_DU_U")
    print("  ✓ Based on HIGH confidence parsing")


if __name__ == "__main__":
    demonstrate_integration()
    test_parser_with_mock_generator()
