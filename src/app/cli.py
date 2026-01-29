"""
Command Line Interface for Guitar Strum Generator
==================================================

This module provides a user-friendly command-line interface for generating
guitar chord progressions and strumming patterns from natural language prompts.

Usage Examples:
    # Basic usage - generate from a prompt
    python -m src.app.cli "mellow acoustic song in D major"
    
    # Interactive mode - keep generating
    python -m src.app.cli --interactive
    
    # Verbose mode - see processing details
    python -m src.app.cli "sad ballad in Am" --verbose
    
    # JSON output - for scripting/integration
    python -m src.app.cli "upbeat folk in G" --json
    
    # Show help
    python -m src.app.cli --help

Author: Rohan Rajendra Dhanawade
Project: Master's Thesis - SRH Berlin University of Applied Sciences
Chat: 9 - User Interface (CLI)
"""

import argparse
import sys
import json
from typing import Dict, Optional

# =============================================================================
# PART 1: ARGUMENT PARSER SETUP
# =============================================================================
# 
# argparse is Python's built-in library for handling command-line arguments.
# It automatically generates help messages and handles errors gracefully.
#
# Think of it like a form that validates user input before your program runs.
# =============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the command-line argument parser.
    
    This function defines all the options users can pass to the CLI.
    
    Returns:
        Configured ArgumentParser object
    """
    # Create the main parser with description shown in --help
    parser = argparse.ArgumentParser(
        prog="guitar-gen",
        description="""
🎸 Guitar Strum Generator - Generate chord progressions and strumming patterns
from natural language descriptions.

Examples:
  "mellow acoustic song in D major"
  "upbeat folk music, fast tempo"  
  "sad ballad in A minor"
  "energetic rock in E minor"
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Part of Master's Thesis: "A Conversational AI System for Symbolic Guitar 
Strumming Pattern and Chord Progression Generation"
Author: Rohan Rajendra Dhanawade | SRH Berlin University of Applied Sciences
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Positional argument: the prompt (optional if using --interactive)
    # ─────────────────────────────────────────────────────────────────────────
    parser.add_argument(
        "prompt",
        nargs="?",  # "?" means optional (0 or 1 argument)
        type=str,
        help="Natural language description of the guitar part you want"
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Mode flags
    # ─────────────────────────────────────────────────────────────────────────
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",  # If present, sets to True
        help="Enter interactive mode (keep generating until you quit)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed processing information"
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Output format options
    # ─────────────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON (useful for scripting)"
    )
    
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Show minimal output (just chords and pattern)"
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Generation options
    # ─────────────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--rules-only",
        action="store_true",
        help="Use only rule-based generation (skip neural model)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Neural model temperature (0.1-2.0, default: 0.8). Lower = more predictable"
    )
    
    return parser


# =============================================================================
# PART 2: OUTPUT FORMATTING FUNCTIONS
# =============================================================================
#
# These functions take the raw dictionary output from generate_guitar_part()
# and format it nicely for display in the terminal.
#
# Good CLI design = information is easy to read at a glance.
# =============================================================================

def format_header() -> str:
    """
    Generate the application header/banner.
    
    Returns:
        Formatted header string
    """
    return """
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   🎸  G U I T A R   S T R U M   G E N E R A T O R                        ║
║                                                                           ║
║   Generate chord progressions and strumming patterns from text prompts    ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""


def format_result_pretty(result: Dict) -> str:
    """
    Format the generation result as a beautiful ASCII display.
    
    This is the main output format users see. It shows:
    - The original prompt
    - Musical metadata (key, tempo, genre, emotion)
    - Generated chords with arrows
    - Strumming pattern with beat markers
    - Generation source (neural vs rule-based)
    
    Args:
        result: Dictionary from generate_guitar_part()
        
    Returns:
        Formatted string for terminal display
    """
    lines = []
    
    # ─────────────────────────────────────────────────────────────────────────
    # Section 1: Prompt
    # ─────────────────────────────────────────────────────────────────────────
    prompt = result.get("prompt", "")
    lines.append("┌" + "─" * 73 + "┐")
    lines.append("│" + " YOUR PROMPT ".center(73) + "│")
    lines.append("├" + "─" * 73 + "┤")
    
    # Handle long prompts by wrapping
    if len(prompt) > 69:
        prompt_display = prompt[:66] + "..."
    else:
        prompt_display = prompt
    lines.append("│  \"" + prompt_display + "\"".ljust(70) + "│")
    lines.append("└" + "─" * 73 + "┘")
    lines.append("")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Section 2: Musical Metadata
    # ─────────────────────────────────────────────────────────────────────────
    key = result.get("key", "?")
    mode = result.get("mode", "?")
    tempo = result.get("tempo", "?")
    genre = result.get("genre", "?")
    emotion = result.get("emotion", "?")
    
    lines.append("┌" + "─" * 73 + "┐")
    lines.append("│" + " MUSICAL DETAILS ".center(73) + "│")
    lines.append("├" + "─" * 35 + "┬" + "─" * 37 + "┤")
    lines.append(f"│  🎹 Key:     {key} {mode}".ljust(37) + f"│  ⏱️  Tempo:   {tempo} BPM".ljust(38) + "│")
    lines.append(f"│  🎵 Genre:   {genre}".ljust(37) + f"│  💭 Emotion: {emotion}".ljust(38) + "│")
    lines.append("└" + "─" * 35 + "┴" + "─" * 37 + "┘")
    lines.append("")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Section 3: Chord Progression
    # ─────────────────────────────────────────────────────────────────────────
    chords = result.get("chords", [])
    chord_str = "  →  ".join(chords) if chords else "(no chords)"
    
    lines.append("┌" + "─" * 73 + "┐")
    lines.append("│" + " CHORD PROGRESSION ".center(73) + "│")
    lines.append("├" + "─" * 73 + "┤")
    lines.append("│" + "│".center(73) + "│")
    
    # Center the chord progression
    chord_line = f"│   {chord_str}   │"
    lines.append("│" + chord_str.center(73) + "│")
    lines.append("│" + "│".center(73) + "│")
    lines.append("└" + "─" * 73 + "┘")
    lines.append("")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Section 4: Strumming Pattern
    # ─────────────────────────────────────────────────────────────────────────
    pattern = result.get("strum_pattern", "")
    
    lines.append("┌" + "─" * 73 + "┐")
    lines.append("│" + " STRUMMING PATTERN ".center(73) + "│")
    lines.append("├" + "─" * 73 + "┤")
    lines.append("│" + "│".center(73) + "│")
    
    # Beat numbers row
    beat_row = "Beat:    1       &       2       &       3       &       4       &"
    lines.append("│  " + beat_row.ljust(71) + "│")
    
    # Pattern row with spacing
    if pattern:
        # Add spacing between each character for readability
        pattern_spaced = "       ".join(pattern)
        pattern_row = f"Strum:   {pattern_spaced}"
    else:
        pattern_row = "Strum:   (no pattern)"
    lines.append("│  " + pattern_row.ljust(71) + "│")
    
    lines.append("│" + "│".center(73) + "│")
    
    # Add legend
    lines.append("│  " + "Legend: D = Downstroke, U = Upstroke, _ = Rest".ljust(71) + "│")
    lines.append("└" + "─" * 73 + "┘")
    lines.append("")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Section 5: Generation Source
    # ─────────────────────────────────────────────────────────────────────────
    source = result.get("source", "unknown")
    
    if source == "neural":
        source_emoji = "🤖"
        source_text = "Neural Model (LSTM)"
        pattern_fixed = result.get("pattern_was_fixed", False)
        if pattern_fixed:
            source_text += " [pattern auto-corrected]"
    else:
        source_emoji = "📚"
        source_text = "Rule-Based System"
        fallback_reason = result.get("fallback_reason", "")
        if fallback_reason:
            source_text += f" ({fallback_reason})"
    
    lines.append(f"Generated by: {source_emoji} {source_text}")
    lines.append("")
    
    return "\n".join(lines)


def format_result_compact(result: Dict) -> str:
    """
    Format the result in a minimal, compact style.
    
    Useful when you just want the essentials without decoration.
    
    Args:
        result: Dictionary from generate_guitar_part()
        
    Returns:
        Compact formatted string
    """
    chords = result.get("chords", [])
    pattern = result.get("strum_pattern", "")
    key = result.get("key", "?")
    mode = result.get("mode", "?")
    tempo = result.get("tempo", "?")
    
    lines = [
        f"Key: {key} {mode} | Tempo: {tempo} BPM",
        f"Chords: {' → '.join(chords)}",
        f"Pattern: {pattern}",
        f"         1 & 2 & 3 & 4 &"
    ]
    
    return "\n".join(lines)


def format_result_json(result: Dict) -> str:
    """
    Format the result as JSON.
    
    Useful for piping to other programs or saving to files.
    
    Args:
        result: Dictionary from generate_guitar_part()
        
    Returns:
        JSON string
    """
    # Create a clean copy without non-serializable objects
    output = {
        "prompt": result.get("prompt", ""),
        "chords": result.get("chords", []),
        "strum_pattern": result.get("strum_pattern", ""),
        "tempo": result.get("tempo", 0),
        "key": result.get("key", ""),
        "mode": result.get("mode", ""),
        "genre": result.get("genre", ""),
        "emotion": result.get("emotion", ""),
        "source": result.get("source", ""),
        "pattern_was_fixed": result.get("pattern_was_fixed", False),
    }
    
    return json.dumps(output, indent=2)


# =============================================================================
# PART 3: GENERATION WRAPPER
# =============================================================================
#
# This function wraps the call to generate_guitar_part() with error handling
# and user-friendly messages.
# =============================================================================

def generate_from_prompt(
    prompt: str,
    verbose: bool = False,
    use_rules_only: bool = False,
    temperature: float = 0.8
) -> Optional[Dict]:
    """
    Generate guitar part from a prompt with error handling.
    
    This is a wrapper around generate_guitar_part() that adds:
    - Import error handling (if PyTorch not installed)
    - User-friendly error messages
    - Verbose output option
    
    Args:
        prompt: Natural language description
        verbose: Show processing details
        use_rules_only: Skip neural model, use only rules
        temperature: Neural model temperature
        
    Returns:
        Result dictionary or None if error
    """
    try:
        # Import the generator (may fail if dependencies missing)
        from src.app.generate import generate_guitar_part
        
        # Call the generator
        result = generate_guitar_part(
            prompt=prompt,
            prefer_neural=not use_rules_only,
            temperature=temperature,
            verbose=verbose
        )
        
        return result
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\nThis usually means a required package is not installed.")
        print("Try: pip install torch pydantic")
        return None
        
    except Exception as e:
        print(f"\n❌ Generation Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None


# =============================================================================
# PART 4: INTERACTIVE MODE
# =============================================================================
#
# Interactive mode lets users keep generating without restarting the program.
# This is useful for experimenting with different prompts.
# =============================================================================

def run_interactive_mode(
    verbose: bool = False,
    use_rules_only: bool = False,
    temperature: float = 0.8,
    output_json: bool = False,
    output_compact: bool = False
):
    """
    Run the CLI in interactive mode.
    
    Users can keep entering prompts until they type 'quit' or 'exit'.
    
    Args:
        verbose: Show processing details
        use_rules_only: Skip neural model
        temperature: Neural model temperature
        output_json: Output as JSON
        output_compact: Output in compact format
    """
    print(format_header())
    print("Interactive Mode - Type your prompts below.")
    print("Commands: 'quit' or 'exit' to stop, 'help' for examples")
    print("─" * 75)
    print()
    
    while True:
        try:
            # Get user input
            prompt = input("🎸 Enter prompt: ").strip()
            
            # Check for exit commands
            if prompt.lower() in ["quit", "exit", "q"]:
                print("\n👋 Thanks for using Guitar Strum Generator! Rock on! 🎸")
                break
            
            # Check for help command
            if prompt.lower() in ["help", "h", "?"]:
                print_example_prompts()
                continue
            
            # Check for empty input
            if not prompt:
                print("⚠️  Please enter a prompt (or 'quit' to exit)")
                continue
            
            # Generate
            print()  # Add spacing
            result = generate_from_prompt(
                prompt=prompt,
                verbose=verbose,
                use_rules_only=use_rules_only,
                temperature=temperature
            )
            
            if result:
                # Format and display output
                if output_json:
                    print(format_result_json(result))
                elif output_compact:
                    print(format_result_compact(result))
                else:
                    print(format_result_pretty(result))
            
            print("─" * 75)
            print()
            
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except EOFError:
            # Handle Ctrl+D (end of input)
            print("\n\n👋 Goodbye!")
            break


def print_example_prompts():
    """Print example prompts to help users get started."""
    print("""
╭─────────────────────────────────────────────────────────────────────────╮
│                          EXAMPLE PROMPTS                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Try these to get started:                                              │
│                                                                         │
│  • "mellow acoustic song in D major"                                    │
│  • "upbeat folk in G, fast tempo"                                       │
│  • "sad ballad in A minor"                                              │
│  • "energetic rock in E minor"                                          │
│  • "happy pop song"                                                     │
│  • "melancholic indie track"                                            │
│  • "chill jazz in Bb major"                                             │
│  • "powerful anthem in C major, slow"                                   │
│                                                                         │
│  Tips:                                                                  │
│  • Specify a key (C, D, G, Am, Em, etc.)                               │
│  • Add tempo hints (slow, fast, moderate)                               │
│  • Include genre (rock, folk, pop, jazz, etc.)                         │
│  • Describe the mood (happy, sad, mellow, energetic)                   │
│                                                                         │
╰─────────────────────────────────────────────────────────────────────────╯
""")


# =============================================================================
# PART 5: SINGLE GENERATION MODE
# =============================================================================
#
# For when the user passes a prompt directly as a command-line argument.
# =============================================================================

def run_single_generation(
    prompt: str,
    verbose: bool = False,
    use_rules_only: bool = False,
    temperature: float = 0.8,
    output_json: bool = False,
    output_compact: bool = False
):
    """
    Run a single generation and exit.
    
    Args:
        prompt: The natural language prompt
        verbose: Show processing details
        use_rules_only: Skip neural model
        temperature: Neural model temperature
        output_json: Output as JSON
        output_compact: Output in compact format
    """
    # Only show header for non-JSON output
    if not output_json:
        print(format_header())
    
    # Generate
    result = generate_from_prompt(
        prompt=prompt,
        verbose=verbose,
        use_rules_only=use_rules_only,
        temperature=temperature
    )
    
    if result:
        # Format and display output
        if output_json:
            print(format_result_json(result))
        elif output_compact:
            print(format_result_compact(result))
        else:
            print(format_result_pretty(result))
    else:
        sys.exit(1)  # Exit with error code


# =============================================================================
# PART 6: MAIN ENTRY POINT
# =============================================================================
#
# This is where the program starts when you run:
#   python -m src.app.cli "your prompt here"
# =============================================================================

def main():
    """
    Main entry point for the CLI.
    
    Parses arguments and dispatches to the appropriate mode:
    - Interactive mode if --interactive flag is set
    - Single generation mode otherwise
    """
    # Create and parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Validate arguments
    # ─────────────────────────────────────────────────────────────────────────
    
    # If neither interactive mode nor prompt provided, show help
    if not args.interactive and not args.prompt:
        parser.print_help()
        print("\n⚠️  Please provide a prompt or use --interactive mode")
        sys.exit(1)
    
    # Validate temperature range
    if args.temperature < 0.1 or args.temperature > 2.0:
        print(f"⚠️  Temperature must be between 0.1 and 2.0 (got {args.temperature})")
        sys.exit(1)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Dispatch to appropriate mode
    # ─────────────────────────────────────────────────────────────────────────
    
    if args.interactive:
        # Interactive mode
        run_interactive_mode(
            verbose=args.verbose,
            use_rules_only=args.rules_only,
            temperature=args.temperature,
            output_json=args.json,
            output_compact=args.compact
        )
    else:
        # Single generation mode
        run_single_generation(
            prompt=args.prompt,
            verbose=args.verbose,
            use_rules_only=args.rules_only,
            temperature=args.temperature,
            output_json=args.json,
            output_compact=args.compact
        )


# =============================================================================
# SCRIPT EXECUTION
# =============================================================================
#
# This block runs when the file is executed directly:
#   python src/app/cli.py "prompt"
# or
#   python -m src.app.cli "prompt"
# =============================================================================

if __name__ == "__main__":
    main()
