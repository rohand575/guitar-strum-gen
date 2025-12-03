"""
Setup configuration for Guitar Strum Generator package.

This allows you to install the project with:
    pip install -e .

After installation, you can import modules like:
    from src.data.schema import GuitarSample
    from src.rules.generate_rule_based import generate
"""

from setuptools import setup, find_packages

# Read the README for long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    # -------------------------
    # Basic Package Information
    # -------------------------
    name="guitar-strum-gen",
    version="0.1.0",
    author="Rohan Rajendra Dhanawade",
    author_email="rohan.dhanawade@example.com",  # Update with your email
    description="Conversational AI for Symbolic Guitar Strumming Pattern and Chord Progression Generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # -------------------------
    # Package Discovery
    # -------------------------
    # This tells setuptools to find all packages under the current directory
    packages=find_packages(where="."),
    package_dir={"": "."},
    
    # -------------------------
    # Python Version Requirement
    # -------------------------
    python_requires=">=3.9",
    
    # -------------------------
    # Dependencies
    # -------------------------
    # Core dependencies (installed automatically)
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pydantic>=2.0.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    
    # Optional dependencies (install with pip install -e ".[dev]")
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "flake8>=6.1.0",
            "black>=23.7.0",
            "mypy>=1.5.0",
            "ipdb>=0.13.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "notebook>=7.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
        "app": [
            "streamlit>=1.25.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
        ],
    },
    
    # -------------------------
    # Entry Points (CLI commands)
    # -------------------------
    entry_points={
        "console_scripts": [
            # This creates a command-line tool: guitar-gen "your prompt here"
            "guitar-gen=src.app.generate:main",
        ],
    },
    
    # -------------------------
    # Metadata
    # -------------------------
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="guitar, music generation, symbolic music, NLP, transformer, chord progression, strumming",
)
