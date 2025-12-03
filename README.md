# 🎸 Guitar Strum Generator

**A Conversational AI System for Symbolic Guitar Strumming Pattern and Chord Progression Generation**

> Master Thesis Project — M.Sc. Computer Science (Big Data & AI)  
> SRH Berlin University of Applied Sciences  
> Author: Rohan Rajendra Dhanawade

---

## 📋 Overview

This system generates **symbolic guitar notation** (chord progressions + strumming patterns) from natural language prompts. Unlike audio-based AI music systems, this produces **editable, readable outputs** that guitarists can directly use.

**Example:**
```
Input:  "Give me a melancholic ballad in Am with a slow, gentle rhythm"
Output: 
  Chords: Am → F → C → G
  Strum:  D _ D U _ U D _  (tempo: 70 BPM)
```

---

## 🎯 Key Features

- **Natural Language Input**: Describe what you want in plain English
- **Symbolic Output**: Get chord progressions + strumming patterns (no audio files)
- **Hybrid Architecture**: Neural model with rule-based fallback for reliability
- **Guitar-Specific**: Designed specifically for guitar idioms, not generic MIDI

---

## 🏗️ Project Structure

```
guitar-strum-gen/
├── data/
│   ├── raw/              # Original datasets (GuitarSet, etc.)
│   └── processed/        # Cleaned, annotated data (JSONL)
│
├── notebooks/
│   ├── 01_explore_dataset.ipynb
│   ├── 02_build_dataset.ipynb
│   ├── 03_training.ipynb
│   └── 04_evaluation.ipynb
│
├── src/
│   ├── data/             # Schema definitions & data loaders
│   ├── rules/            # Rule-based baseline system
│   ├── models/           # Neural models & tokenizers
│   ├── train/            # Training scripts
│   ├── evaluation/       # Metrics & experiments
│   └── app/              # API / UI demo
│
├── tests/                # Unit tests
├── docs/                 # Documentation & diagrams
├── outputs/              # Generated chord sheets
└── configs/              # Hyperparameters & settings
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/guitar-strum-gen.git
cd guitar-strum-gen
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .  # Install the project in editable mode
```

### 4. Run the Demo
```bash
python -m src.app.generate "upbeat pop progression in G major"
```

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER PROMPT (Natural Language)              │
│            "melancholic ballad in Am, slow tempo"               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PROMPT PARSER                              │
│   Extracts: key=Am, emotion=melancholic, style=ballad,         │
│             tempo=slow (60-80 BPM)                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   SEQUENCE GENERATOR                            │
│                                                                 │
│   ┌─────────────────┐    ┌─────────────────────────────┐       │
│   │  Neural Model   │───▶│  Valid Output?              │       │
│   │  (Transformer)  │    │  • Chords in key?           │       │
│   └─────────────────┘    │  • Strum pattern valid?     │       │
│                          └─────────────────────────────┘       │
│                                    │                            │
│                          ┌────────┴────────┐                   │
│                          │                 │                    │
│                         YES               NO                    │
│                          │                 │                    │
│                          ▼                 ▼                    │
│                    Use Neural        Rule-Based                 │
│                      Output           Fallback                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SYMBOLIC OUTPUT                            │
│                                                                 │
│   {                                                             │
│     "chords": ["Am", "F", "C", "G"],                           │
│     "strum_pattern": "D_DU_UD_",                               │
│     "tempo": 72,                                                │
│     "time_signature": "4/4"                                     │
│   }                                                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Data Format

Each training example follows this JSON schema:

```json
{
  "id": "sample_001",
  "prompt": "upbeat folk strum in G major at moderate tempo",
  "chords": ["G", "D", "Em", "C"],
  "strum_pattern": "D_DU_DU_",
  "tempo": 110,
  "time_signature": "4/4",
  "genre": "folk",
  "emotion": "upbeat",
  "key": "G",
  "mode": "major"
}
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Chord Correctness** | % of chords that fit the detected key |
| **Rhythmic Consistency** | Alignment with metrical grid |
| **Stylistic Diversity** | Entropy of pattern distributions |
| **User Ratings** | Playability, expressiveness, usefulness (1-5) |

---

## 🔧 Development

### For Local Development
```bash
# Install in development mode
pip install -e ".[dev]"

# Run linting
flake8 src/

# Run type checking
mypy src/
```

### For Colab Training
Upload the notebooks from `notebooks/` to Google Colab and follow the instructions within each notebook.

---

## 📚 References

Key papers informing this work:
- Bhandari et al. (2025) - Text2MIDI
- Muhamed et al. (2021) - Symbolic Music Generation with Transformer-GANs
- Sarmento et al. (2023) - GTR-CTRL
- de Berardinis et al. (2023) - ChoCo Chord Corpus

---

## 📄 License

This project is for academic purposes as part of a Master's thesis.

---

## 👤 Author

**Rohan Rajendra Dhanawade**  
M.Sc. Computer Science — Big Data & AI  
SRH Berlin University of Applied Sciences

Supervisors:
- Prof. Dr. Alexander I. Iliev (First Supervisor)
- Nazneen Mansoor (Second Supervisor)
