"""
Application Layer - User Interfaces
====================================

This package contains the user-facing interfaces for the Guitar Strum Generator:

- generate.py: Core hybrid generation logic (neural + rule-based)
- cli.py: Command-line interface
- ui_streamlit.py: Web interface (Streamlit)

Usage:
    # CLI
    python -m src.app.cli "your prompt here"
    python -m src.app.cli --interactive
    
    # Streamlit Web UI
    streamlit run src/app/ui_streamlit.py
    
    # Programmatic
    from src.app.generate import generate_guitar_part
    result = generate_guitar_part("mellow acoustic in D major")
"""

from .generate import (
    generate_guitar_part,
    format_result_as_chord_sheet,
    validate_output,
    ValidationResult,
)

from .cli import (
    main as cli_main,
    format_result_pretty,
    format_result_compact,
    format_result_json,
)

# Note: ui_streamlit is not imported here because it's meant to be run
# as a standalone Streamlit app, not imported as a module.

__all__ = [
    # Core generation
    "generate_guitar_part",
    "format_result_as_chord_sheet",
    "validate_output",
    "ValidationResult",
    # CLI
    "cli_main",
    "format_result_pretty",
    "format_result_compact",
    "format_result_json",
]
