# 🎵 CoverComposer: AI-Powered Music Track Generator

CoverComposer is an AI-powered web application that generates custom instrumental tracks based on user-defined parameters such as **mood**, **genre**, **tempo**, and **style**. Leveraging FastAPI, Markov Chain-based melody generation, MIDI creation, and audio synthesis, it provides a seamless, interactive music generation experience.

## 🚀 Features

- **AI Melody Generation** using Markov Chain probabilistic transitions
- **Multi-Track MIDI** creation (Melody + Bass + Drums)
- **4 Moods**: Happy (Major), Sad (Minor), Calm (Pentatonic), Energetic (Dorian)
- **4 Genres**: Pop, Rock, Jazz, Electronic — each with unique instrument mappings
- **2 Styles**: Simple (steady) vs Complex (dynamic rhythms & volume)
- **Real-time Audio Synthesis** — generates playable WAV files
- **Sleek Dark UI** with animated audio visualizer bars
- **Download & Regenerate** functionality

## 📁 Project Structure

```
CoverComposer/
├── main.py              # Core FastAPI backend + AI music generation
├── requirements.txt     # Python dependencies
├── soundfont.sf2        # (Optional) SoundFont for FluidSynth
├── static/
│   ├── style.css        # Dark theme CSS with animations
│   └── output/          # Generated .mid and .wav files
├── templates/
│   ├── index.html       # Landing page with generation form
│   └── result.html      # Result page with audio player
└── README.md
```

## 🛠️ Setup & Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn main:app --reload
```

Then open **http://127.0.0.1:8000** in your browser.

## 🧠 Technical Architecture

### Music Generation Pipeline
1. **User Input** → Mood, Genre, Tempo, Style
2. **Markov Chain** → Generates melody using mood-based musical scales
3. **Note Styling** → Applies rhythm & dynamics based on style
4. **MIDI Creation** → 3-track MIDI (melody, bass, drums)
5. **Audio Synthesis** → Converts MIDI to WAV

### AI Component: Markov Chains
- Transition probabilities built from musical scale intervals
- Stepwise motion preferred (35% weight)
- Probabilistic variation ensures unique compositions every time

### Audio Synthesis
- **Primary**: FluidSynth + SoundFont (if available)
- **Fallback**: NumPy-powered synthesizer with:
  - Multiple waveforms (sine, square, saw, triangle)
  - ADSR envelope shaping
  - Synthesized percussion (kick, snare, hi-hat)

## 📝 Requirements

- Python 3.8+
- FastAPI, Uvicorn, Jinja2
- MIDIUtil, Mido
- NumPy, SciPy
- (Optional) FluidSynth + SoundFont for premium audio

## 🎯 Usage Scenarios

1. **Random Sampling**: Create random music samples with adjustable tempo
2. **Genre-based**: Choose between Pop, Rock, Jazz, Electronic for genre-specific beats
3. **Mood-based**: Select moods and styles for personalized compositions
