"""
Streamlit Web Interface for Guitar Strum Generator
===================================================

This module provides a beautiful, interactive web interface for generating
guitar chord progressions and strumming patterns from natural language prompts.

How to Run:
    cd your-project-folder
    streamlit run src/app/ui_streamlit.py

Or with custom port:
    streamlit run src/app/ui_streamlit.py --server.port 8501

Features:
    - Natural language input for describing guitar parts
    - Real-time generation with neural model + rule-based fallback
    - Visual chord progression display
    - Interactive strumming pattern visualization
    - Generation history
    - Download results as JSON
    - Adjustable generation parameters

Author: Rohan Rajendra Dhanawade
Project: Master's Thesis - SRH Berlin University of Applied Sciences
Chat: 9 - User Interface (Streamlit)
"""

import streamlit as st
import json
from datetime import datetime
from typing import Dict, List, Optional

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
# This MUST be the first Streamlit command in the script!
# It sets up the page title, icon, and layout.
# =============================================================================

st.set_page_config(
    page_title="Guitar Strum Generator",
    page_icon="🎸",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": """
        ## 🎸 Guitar Strum Generator
        
        A conversational AI system for generating guitar chord progressions
        and strumming patterns from natural language descriptions.
        
        **Author:** Rohan Rajendra Dhanawade  
        **Institution:** SRH Berlin University of Applied Sciences  
        **Thesis:** M.Sc. Computer Science (Big Data & AI)
        """
    }
)


# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================
# Streamlit allows injecting custom CSS to make the app look better.
# This is optional but makes the app more visually appealing.
# =============================================================================

def inject_custom_css():
    """Inject custom CSS for better styling."""
    st.markdown("""
    <style>
    /* Main container padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Chord boxes styling */
    .chord-box {
        display: inline-block;
        padding: 15px 25px;
        margin: 5px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Arrow between chords */
    .chord-arrow {
        display: inline-block;
        padding: 0 10px;
        font-size: 24px;
        color: #888;
    }
    
    /* Strum pattern styling */
    .strum-beat {
        display: inline-block;
        width: 50px;
        text-align: center;
        font-family: monospace;
    }
    
    .strum-down {
        color: #2ecc71;
        font-size: 28px;
        font-weight: bold;
    }
    
    .strum-up {
        color: #e74c3c;
        font-size: 28px;
        font-weight: bold;
    }
    
    .strum-rest {
        color: #95a5a6;
        font-size: 28px;
    }
    
    /* Info cards */
    .info-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Success message styling */
    .success-banner {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    /* History item styling */
    .history-item {
        background-color: #f1f3f4;
        border-left: 4px solid #667eea;
        padding: 10px 15px;
        margin: 5px 0;
        border-radius: 0 5px 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
# Streamlit reruns the entire script on every interaction.
# We use session_state to persist data between reruns.
# Think of it as the app's "memory".
# =============================================================================

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    
    # Current generation result
    if "current_result" not in st.session_state:
        st.session_state.current_result = None
    
    # History of all generations in this session
    if "generation_history" not in st.session_state:
        st.session_state.generation_history = []
    
    # Track if generator is loaded
    if "generator_loaded" not in st.session_state:
        st.session_state.generator_loaded = False
    
    # Error message storage
    if "error_message" not in st.session_state:
        st.session_state.error_message = None


# =============================================================================
# GENERATOR LOADING (CACHED)
# =============================================================================
# We use @st.cache_resource to load the generator only ONCE,
# not on every script rerun. This is crucial for performance
# since loading the neural model takes time.
# =============================================================================

@st.cache_resource
def load_generator():
    """
    Load the guitar generator module.
    
    This function is cached, meaning it only runs once per session,
    not on every Streamlit rerun.
    
    Returns:
        The generate_guitar_part function, or None if import fails
    """
    try:
        from src.app.generate import generate_guitar_part
        return generate_guitar_part
    except ImportError as e:
        st.error(f"❌ Failed to import generator: {e}")
        st.info("Make sure you're running from the project root directory.")
        return None


# =============================================================================
# GENERATION FUNCTION
# =============================================================================

def generate_guitar_part_safe(
    prompt: str,
    use_neural: bool = True,
    temperature: float = 0.8
) -> Optional[Dict]:
    """
    Safely generate a guitar part with error handling.
    
    Args:
        prompt: Natural language description
        use_neural: Whether to try neural model first
        temperature: Neural model temperature
        
    Returns:
        Result dictionary or None if error
    """
    generator = load_generator()
    
    if generator is None:
        st.session_state.error_message = "Generator not loaded"
        return None
    
    try:
        result = generator(
            prompt=prompt,
            prefer_neural=use_neural,
            temperature=temperature,
            verbose=False
        )
        st.session_state.error_message = None
        return result
        
    except Exception as e:
        st.session_state.error_message = str(e)
        return None


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================
# These functions create the visual components of the app.
# =============================================================================

def display_header():
    """Display the app header."""
    st.markdown("""
    # 🎸 Guitar Strum Generator
    
    *Generate chord progressions and strumming patterns from natural language descriptions*
    """)
    st.markdown("---")


def display_chord_progression(chords: List[str]):
    """
    Display chords as beautiful visual boxes.
    
    Args:
        chords: List of chord names
    """
    if not chords:
        st.warning("No chords generated")
        return
    
    # Build HTML for chord display
    chord_html = '<div style="text-align: center; padding: 20px;">'
    
    for i, chord in enumerate(chords):
        chord_html += f'<span class="chord-box">{chord}</span>'
        if i < len(chords) - 1:
            chord_html += '<span class="chord-arrow">→</span>'
    
    chord_html += '</div>'
    
    st.markdown(chord_html, unsafe_allow_html=True)


def display_strumming_pattern(pattern: str):
    """
    Display strumming pattern with visual indicators.
    
    Args:
        pattern: 8-character strumming pattern (D, U, _)
    """
    if not pattern:
        st.warning("No strumming pattern generated")
        return
    
    # Create beat labels
    beats = ["1", "&", "2", "&", "3", "&", "4", "&"]
    
    # Map pattern characters to display symbols
    symbol_map = {
        "D": ("↓", "strum-down"),
        "U": ("↑", "strum-up"),
        "_": ("·", "strum-rest")
    }
    
    # Build the display
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        # Beat row
        beat_html = '<div style="text-align: center; margin-bottom: 5px;">'
        for beat in beats:
            beat_html += f'<span class="strum-beat" style="color: #666; font-size: 14px;">{beat}</span>'
        beat_html += '</div>'
        
        # Pattern row
        pattern_html = '<div style="text-align: center;">'
        for char in pattern[:8]:  # Ensure max 8 characters
            symbol, css_class = symbol_map.get(char, ("?", "strum-rest"))
            pattern_html += f'<span class="strum-beat {css_class}">{symbol}</span>'
        pattern_html += '</div>'
        
        st.markdown(beat_html, unsafe_allow_html=True)
        st.markdown(pattern_html, unsafe_allow_html=True)
        
        # Show pattern string
        st.markdown(f'<p style="text-align: center; color: #888; margin-top: 10px;">Pattern: <code>{pattern}</code></p>', unsafe_allow_html=True)
        
        # Legend
        st.markdown("""
        <p style="text-align: center; font-size: 12px; color: #aaa;">
        ↓ = Downstroke &nbsp;&nbsp; ↑ = Upstroke &nbsp;&nbsp; · = Rest
        </p>
        """, unsafe_allow_html=True)


def display_metadata(result: Dict):
    """
    Display musical metadata in a nice grid.
    
    Args:
        result: Generation result dictionary
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🎹 Key",
            value=f"{result.get('key', '?')} {result.get('mode', '?')}"
        )
    
    with col2:
        st.metric(
            label="⏱️ Tempo",
            value=f"{result.get('tempo', '?')} BPM"
        )
    
    with col3:
        st.metric(
            label="🎵 Genre",
            value=result.get('genre', '?').title()
        )
    
    with col4:
        st.metric(
            label="💭 Emotion",
            value=result.get('emotion', '?').title()
        )


def display_generation_source(result: Dict):
    """
    Display information about how the result was generated.
    
    Args:
        result: Generation result dictionary
    """
    source = result.get("source", "unknown")
    
    if source == "neural":
        icon = "🤖"
        text = "Neural Model (LSTM)"
        color = "#2ecc71"
        
        # Check if pattern was fixed
        if result.get("pattern_was_fixed", False):
            text += " • Pattern auto-corrected"
    else:
        icon = "📚"
        text = "Rule-Based System"
        color = "#3498db"
        
        # Show fallback reason if available
        fallback = result.get("fallback_reason", "")
        if fallback:
            text += f" • {fallback.replace('_', ' ').title()}"
    
    st.markdown(f"""
    <div style="text-align: center; padding: 10px; background-color: {color}22; 
                border-radius: 5px; border-left: 4px solid {color};">
        <span style="font-size: 20px;">{icon}</span>
        <span style="margin-left: 10px;">Generated by: <strong>{text}</strong></span>
    </div>
    """, unsafe_allow_html=True)


def display_full_result(result: Dict):
    """
    Display the complete generation result.
    
    Args:
        result: Generation result dictionary
    """
    # Metadata section
    st.markdown("### 📊 Musical Details")
    display_metadata(result)
    
    st.markdown("---")
    
    # Chord progression section
    st.markdown("### 🎶 Chord Progression")
    display_chord_progression(result.get("chords", []))
    
    st.markdown("---")
    
    # Strumming pattern section
    st.markdown("### 🥁 Strumming Pattern")
    display_strumming_pattern(result.get("strum_pattern", ""))
    
    st.markdown("---")
    
    # Generation source
    display_generation_source(result)


def display_download_buttons(result: Dict):
    """
    Display buttons to download/copy the result.
    
    Args:
        result: Generation result dictionary
    """
    col1, col2, col3 = st.columns(3)
    
    # Prepare JSON for download
    download_data = {
        "prompt": result.get("prompt", ""),
        "chords": result.get("chords", []),
        "strum_pattern": result.get("strum_pattern", ""),
        "tempo": result.get("tempo", 0),
        "key": result.get("key", ""),
        "mode": result.get("mode", ""),
        "genre": result.get("genre", ""),
        "emotion": result.get("emotion", ""),
        "source": result.get("source", ""),
        "generated_at": datetime.now().isoformat()
    }
    
    json_str = json.dumps(download_data, indent=2)
    
    with col1:
        st.download_button(
            label="💾 Download JSON",
            data=json_str,
            file_name="guitar_part.json",
            mime="application/json"
        )
    
    with col2:
        # Create a text format for copying
        text_format = f"""Prompt: {result.get('prompt', '')}
Key: {result.get('key', '')} {result.get('mode', '')}
Tempo: {result.get('tempo', '')} BPM
Chords: {' → '.join(result.get('chords', []))}
Pattern: {result.get('strum_pattern', '')}
         1 & 2 & 3 & 4 &"""
        
        st.download_button(
            label="📄 Download Text",
            data=text_format,
            file_name="guitar_part.txt",
            mime="text/plain"
        )
    
    with col3:
        # Show the copyable text in an expander
        with st.expander("📋 Copy to Clipboard"):
            st.code(text_format, language=None)
            st.caption("Select and copy the text above")


# =============================================================================
# SIDEBAR COMPONENTS
# =============================================================================

def render_sidebar() -> Dict:
    """
    Render the sidebar with settings and examples.
    
    Returns:
        Dictionary of settings values
    """
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        st.markdown("---")
        
        # Generation mode
        st.markdown("### Generation Mode")
        use_neural = st.radio(
            "Select mode:",
            options=[True, False],
            format_func=lambda x: "🤖 Hybrid (Neural + Rules)" if x else "📚 Rules Only",
            index=0,
            help="Hybrid mode tries the neural model first, falling back to rules if needed."
        )
        
        st.markdown("---")
        
        # Temperature slider (only relevant for neural mode)
        st.markdown("### Neural Settings")
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=0.8,
            step=0.1,
            help="Controls randomness. Lower = more predictable, Higher = more creative.",
            disabled=not use_neural
        )
        
        st.markdown("---")
        
        # Example prompts
        st.markdown("### 💡 Example Prompts")
        st.markdown("*Click to use:*")
        
        examples = [
            "mellow acoustic song in D major",
            "upbeat folk in G, fast tempo",
            "sad ballad in A minor",
            "energetic rock in E minor",
            "happy pop song in C",
            "melancholic indie track",
            "chill jazz in Bb major",
            "powerful anthem, slow tempo"
        ]
        
        for example in examples:
            if st.button(f"• {example}", key=f"ex_{example}"):
                st.session_state.example_prompt = example
                st.rerun()
        
        st.markdown("---")
        
        # About section
        st.markdown("### ℹ️ About")
        st.markdown("""
        This is part of a Master's thesis project on conversational AI 
        for symbolic music generation.
        
        **Author:** Rohan Rajendra Dhanawade  
        **Institution:** SRH Berlin
        """)
        
        return {
            "use_neural": use_neural,
            "temperature": temperature
        }


def render_history_sidebar():
    """Render generation history in the sidebar."""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📜 Generation History")
        
        if not st.session_state.generation_history:
            st.caption("No generations yet")
        else:
            # Show last 5 generations
            for i, item in enumerate(reversed(st.session_state.generation_history[-5:])):
                with st.expander(f"#{len(st.session_state.generation_history) - i}: {item['prompt'][:30]}..."):
                    st.write(f"**Chords:** {' → '.join(item['chords'])}")
                    st.write(f"**Pattern:** `{item['pattern']}`")
                    st.write(f"**Source:** {item['source']}")
            
            if len(st.session_state.generation_history) > 5:
                st.caption(f"Showing last 5 of {len(st.session_state.generation_history)} generations")
            
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.generation_history = []
                st.rerun()


# =============================================================================
# MAIN INPUT SECTION
# =============================================================================

def render_input_section() -> Optional[str]:
    """
    Render the main input section.
    
    Returns:
        The prompt to generate, or None if no generation requested
    """
    # Check if an example was clicked
    initial_value = ""
    if "example_prompt" in st.session_state:
        initial_value = st.session_state.example_prompt
        del st.session_state.example_prompt
    
    # Create input area
    st.markdown("### 🎤 Describe Your Guitar Part")
    
    prompt = st.text_area(
        "Enter a natural language description:",
        value=initial_value,
        height=100,
        placeholder="e.g., mellow acoustic song in D major, slow tempo",
        help="Describe the style, mood, key, and tempo of the guitar part you want."
    )
    
    # Generate button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_clicked = st.button(
            "🎵 Generate Guitar Part",
            type="primary",
            use_container_width=True
        )
    
    if generate_clicked and prompt.strip():
        return prompt.strip()
    elif generate_clicked and not prompt.strip():
        st.warning("⚠️ Please enter a prompt first!")
    
    return None


# =============================================================================
# TIPS AND HELP SECTION
# =============================================================================

def render_tips_section():
    """Render helpful tips for users."""
    with st.expander("💡 Tips for Better Results", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Specify Musical Elements:**
            - **Key:** C, D, G, Am, Em, Bb, etc.
            - **Mode:** major, minor
            - **Tempo:** slow, fast, moderate, or specific BPM
            """)
        
        with col2:
            st.markdown("""
            **Describe the Feel:**
            - **Genre:** rock, folk, pop, jazz, blues, indie
            - **Emotion:** happy, sad, mellow, energetic, melancholic
            - **Style:** acoustic, electric, ballad, anthem
            """)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    
    # Initialize
    inject_custom_css()
    initialize_session_state()
    
    # Render header
    display_header()
    
    # Render sidebar and get settings
    settings = render_sidebar()
    render_history_sidebar()
    
    # Main content area
    prompt_to_generate = render_input_section()
    
    # Tips section
    render_tips_section()
    
    # Handle generation
    if prompt_to_generate:
        with st.spinner("🎸 Generating your guitar part..."):
            result = generate_guitar_part_safe(
                prompt=prompt_to_generate,
                use_neural=settings["use_neural"],
                temperature=settings["temperature"]
            )
        
        if result:
            st.session_state.current_result = result
            
            # Add to history
            st.session_state.generation_history.append({
                "prompt": prompt_to_generate,
                "chords": result.get("chords", []),
                "pattern": result.get("strum_pattern", ""),
                "source": result.get("source", "unknown"),
                "timestamp": datetime.now().isoformat()
            })
            
            st.success("✅ Generation complete!")
    
    # Display current result if exists
    if st.session_state.current_result:
        st.markdown("---")
        st.markdown("## 🎼 Generated Result")
        
        # Show the prompt that was used
        st.markdown(f'**Prompt:** *"{st.session_state.current_result.get("prompt", "")}"*')
        
        st.markdown("---")
        
        # Display the full result
        display_full_result(st.session_state.current_result)
        
        st.markdown("---")
        
        # Download buttons
        st.markdown("### 📥 Export")
        display_download_buttons(st.session_state.current_result)
    
    # Display any errors
    if st.session_state.error_message:
        st.error(f"❌ Error: {st.session_state.error_message}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 12px;">
    🎸 Guitar Strum Generator | Master's Thesis Project | SRH Berlin University of Applied Sciences
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
