"""
App Subpackage

This package contains the user-facing application:
    - generate.py: Main entry point that combines all components
    - ui_streamlit.py: Web-based UI using Streamlit
    - api.py: REST API using FastAPI (alternative to Streamlit)

The generate.py module is the "glue" that:
    1. Takes a natural language prompt
    2. Parses it to extract features
    3. Generates output using neural model
    4. Validates output using music theory rules
    5. Falls back to rule-based system if invalid
    6. Returns structured JSON result

Usage options:
    - CLI: python -m src.app.generate "your prompt"
    - Streamlit: streamlit run src/app/ui_streamlit.py
    - API: uvicorn src.app.api:app --reload

This is what your thesis examiner will demo!
"""

# Will import when implemented:
# from src.app.generate import generate_guitar_part
