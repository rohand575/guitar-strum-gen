#!/usr/bin/env python3
"""
Test Script for Rule-Based Baseline System

Run: python tests/test_rules.py
"""

import sys
sys.path.insert(0, '.')


def test_harmony():
    """Test the harmony module."""
    print("\n" + "=" * 60)
    print("Testing harmony.py")
    print("=" * 60)
    
    from src.rules.harmony import (
        build_scale, get_diatonic_chords, select_progression, 
        validate_progression, degrees_to_chords
    )
    
    scale = build_scale("G", "major")
    assert scale == ['G', 'A', 'B', 'C', 'D', 'E', 'F#'], f"Wrong scale: {scale}"
    print(f"✅ G major scale: {scale}")
    
    chords = get_diatonic_chords("G", "major")
    assert chords == ['G', 'Am', 'Bm', 'C', 'D', 'Em', 'F#dim'], f"Wrong chords: {chords}"
    print(f"✅ G major chords: {chords}")
    
    prog = degrees_to_chords("C", "major", [1, 5, 6, 4])
    assert prog == ['C', 'G', 'Am', 'F'], f"Wrong progression: {prog}"
    print(f"✅ I-V-vi-IV in C: {prog}")
    
    print("\n✅ All harmony tests passed!")


def test_strumming():
    """Test the strumming module."""
    print("\n" + "=" * 60)
    print("Testing strumming.py")
    print("=" * 60)
    
    from src.rules.strumming import (
        select_strumming_pattern, get_tempo_category, count_strokes
    )
    
    assert get_tempo_category(70) == "slow"
    assert get_tempo_category(110) == "moderate"
    assert get_tempo_category(150) == "fast"
    print("✅ Tempo categorization works")
    
    pattern = select_strumming_pattern("folk", 110)
    assert len(pattern) == 8
    assert all(c in "DU_" for c in pattern)
    print(f"✅ Folk pattern at 110 BPM: {pattern}")
    
    assert count_strokes("D_DU_UD_") == 5
    print("✅ Stroke counting works")
    
    print("\n✅ All strumming tests passed!")


def test_prompt_parser():
    """Test the prompt parser module."""
    print("\n" + "=" * 60)
    print("Testing prompt_parser.py")
    print("=" * 60)
    
    from src.rules.prompt_parser import (
        parse_prompt, extract_key_and_mode, extract_genre, extract_emotion
    )
    
    key, mode = extract_key_and_mode("song in G major")
    assert key == "G" and mode == "major"
    print(f"✅ 'in G major' → key={key}, mode={mode}")
    
    key, mode = extract_key_and_mode("ballad in Am")
    assert key == "A" and mode == "minor"
    print(f"✅ 'in Am' → key={key}, mode={mode}")
    
    genre = extract_genre("upbeat folk song")
    assert genre == "folk"
    print(f"✅ 'folk song' → genre={genre}")
    
    emotion = extract_emotion("melancholic ballad")
    assert emotion == "melancholic"
    print(f"✅ 'melancholic' → emotion={emotion}")
    
    print("\n✅ All prompt parser tests passed!")


def test_full_generator():
    """Test the complete rule-based generator."""
    print("\n" + "=" * 60)
    print("Testing generate_rule_based.py")
    print("=" * 60)
    
    from src.rules.generate_rule_based import generate_rule_based, format_as_chord_sheet
    
    result = generate_rule_based("upbeat folk in G major at 110 BPM")
    assert result.key == "G"
    assert result.mode == "major"
    assert result.genre == "folk"
    assert len(result.chords) >= 2
    assert len(result.strum_pattern) == 8
    print(f"✅ Generated: {result.chords}, pattern={result.strum_pattern}")
    
    result = generate_rule_based("melancholic ballad in Am")
    assert result.mode == "minor"
    print(f"✅ Minor key works: {result.chords}")
    
    sheet = format_as_chord_sheet(result)
    assert "CHORD SHEET" in sheet
    print("✅ Chord sheet formatting works")
    
    print("\n✅ All generator tests passed!")


def main():
    """Run all tests."""
    print("=" * 60)
    print("RULE-BASED SYSTEM - TEST SUITE")
    print("=" * 60)
    
    try:
        test_harmony()
        test_strumming()
        test_prompt_parser()
        test_full_generator()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 60)
        print("\nYour rule-based system is ready!")
        print("Try: python -m src.rules.generate_rule_based \"your prompt\"")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
