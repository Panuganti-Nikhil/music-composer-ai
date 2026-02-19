"""
CoverComposer: AI-Powered Music Track Generator
================================================
Core backend built with FastAPI. Handles:
- Melody generation using mood-based scales + Markov Chains
- MIDI training system to learn from real music files
- Instrument selection and MIDI creation
- Audio rendering using FluidSynth/SoundFont (or pure Python fallback)
- Integration of user input (mood, genre, tempo, style)
- Track history and Markov Chain visualization
"""

import os
import random
import struct
import wave
import math
import time
import json
import glob
from pathlib import Path
from collections import defaultdict

from fastapi import FastAPI, Form, Request, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from midiutil import MIDIFile
import mido
import requests as http_requests

# ============================================================
# App Configuration
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"
OUTPUT_FOLDER = STATIC_DIR / "output"
SOUNDFONT_PATH = BASE_DIR / "soundfont.sf2"
TRAINING_DIR = BASE_DIR / "training_data"
TRAINED_MODELS_DIR = BASE_DIR / "trained_models"

# Create required folders
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
TRAINING_DIR.mkdir(parents=True, exist_ok=True)
TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Gemini AI Configuration (Google AI API)
# ============================================================
GEMINI_API_KEY = "AIzaSyBqqHKhsG21WQsqB_Obgo24VToJcKsSdd4"
GEMINI_MODEL = "gemini-2.0-flash"  # Fast and free-tier friendly
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

# Initialize FastAPI
app = FastAPI(title="CoverComposer", description="AI-Powered Music Track Generator")

# Enable CORS
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Load templates
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# ============================================================
# Check FluidSynth availability
# ============================================================
FLUIDSYNTH_AVAILABLE = False
try:
    if SOUNDFONT_PATH.exists():
        # Try FluidSynth binary path
        fluidsynth_bin = BASE_DIR / "fluidsynth_install" / "bin"
        if fluidsynth_bin.exists():
            os.add_dll_directory(str(fluidsynth_bin))
        import fluidsynth
        FLUIDSYNTH_AVAILABLE = True
        print("✅ FluidSynth available - using high-quality synthesis")
except Exception as e:
    print(f"⚠️ FluidSynth not available ({e}) - using built-in synthesizer")

# ============================================================
# Genre → General MIDI Instrument Mapping
# ============================================================
GENRE_INSTRUMENTS = {
    "pop": {
        "melody": 0,    # Acoustic Grand Piano
        "bass": 33,     # Electric Bass (finger)
        "pad": 88,      # Pad 1 (new age)
        "name": "Pop"
    },
    "rock": {
        "melody": 29,   # Overdriven Guitar
        "bass": 34,     # Electric Bass (pick)
        "pad": 30,      # Distortion Guitar
        "name": "Rock"
    },
    "jazz": {
        "melody": 0,    # Acoustic Grand Piano
        "bass": 32,     # Acoustic Bass
        "pad": 65,      # Alto Sax
        "name": "Jazz"
    },
    "electronic": {
        "melody": 80,   # Lead 1 (square)
        "bass": 38,     # Synth Bass 1
        "pad": 89,      # Pad 2 (warm)
        "name": "Electronic"
    },
}

# ============================================================
# Mood → Musical Scale Mapping
# Each mood uses a DIFFERENT key and octave range so they
# sound clearly distinct from each other.
# ============================================================
MOOD_SCALES = {
    "happy": {
        "scale": [67, 69, 71, 72, 74, 76, 78, 79, 81, 83, 84, 86, 88, 90, 91],
        "description": "Bright G Major scale (upper register)",
        "rhythm_feel": "bouncy",
        "dur_pool": [0.25, 0.25, 0.5, 0.5, 0.75] # Short and snappy
    },
    "sad": {
        "scale": [57, 59, 60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81],
        "description": "Emotional A Natural Minor scale",
        "rhythm_feel": "slow",
        "dur_pool": [0.5, 1.0, 1.0, 1.5, 2.0] # Long, sustained
    },
    "calm": {
        "scale": [63, 65, 68, 70, 72, 75, 77, 80, 82, 84, 87, 89],
        "description": "Dreamy Eb Pentatonic (wide intervals)",
        "rhythm_feel": "flowing",
        "dur_pool": [0.75, 1.0, 1.25, 1.5] # Smooth transitions
    },
    "energetic": {
        "scale": [62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83, 84, 86],
        "description": "Driving D Dorian mode",
        "rhythm_feel": "driving",
        "dur_pool": [0.25, 0.25, 0.25, 0.5] # Fast and repetitive
    },
}


# ============================================================
# MIDI Training System — Learn from Real Music
# ============================================================
def extract_notes_from_midi(midi_path):
    """
    Extract (note, duration) tuples from a MIDI file.
    Durations are quantized to 16th notes (0.25 beats) to simplify the state space.
    """
    try:
        mid = mido.MidiFile(str(midi_path))
        notes = []
        ticks_per_beat = mid.ticks_per_beat
        
        for track in mid.tracks:
            active_notes = {}
            current_time = 0
            
            for msg in track:
                current_time += msg.time
                
                if msg.type == 'note_on' and msg.velocity > 0:
                    active_notes[msg.note] = current_time
                elif (msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)):
                    if msg.note in active_notes:
                        start_time = active_notes.pop(msg.note)
                        duration_ticks = current_time - start_time
                        duration_beats = duration_ticks / ticks_per_beat
                        
                        # Quantize duration to nearest 0.25 (16th note)
                        quantized = round(duration_beats * 4) / 4
                        quantized = max(0.25, min(4.0, quantized)) # Clamp between 16th note and whole note
                        
                        notes.append((msg.note, quantized))
                        
        return notes
    except Exception as e:
        print(f"⚠️ Error reading {midi_path}: {e}")
        return []


def train_markov_from_notes(note_sequences):
    """
    Train a Markov Chain on (note, duration) tuples.
    State is now: "60_0.5" -> {"62_0.25": 10, ...}
    """
    transition_counts = defaultdict(lambda: defaultdict(int))
    total_transitions = 0

    for sequence in note_sequences:
        for i in range(len(sequence) - 1):
            curr_note, curr_dur = sequence[i]
            next_note, next_dur = sequence[i+1]
            
            # Create string keys for JSON serialization "note_duration"
            state = f"{curr_note}_{curr_dur}"
            next_state = f"{next_note}_{next_dur}"
            
            transition_counts[state][next_state] += 1
            total_transitions += 1

    # Convert to probabilities
    transitions = {}
    for state, targets in transition_counts.items():
        total = sum(targets.values())
        transitions[state] = {
            target: count / total
            for target, count in targets.items()
        }

    return transitions, total_transitions


def train_from_midi_folder(folder_path, model_name="custom"):
    """
    Train a Rhythm-Aware Markov Chain model.
    """
    midi_files = list(Path(folder_path).glob("*.mid")) + list(Path(folder_path).glob("*.midi"))

    if not midi_files:
        return None, 0, 0

    all_sequences = []
    total_notes_analyzed = 0
    
    print(f"📂 Analyzing {len(midi_files)} files for rhythm and melody...")
    
    for midi_file in midi_files:
        notes = extract_notes_from_midi(midi_file)
        if notes:
            all_sequences.append(notes)
            total_notes_analyzed += len(notes)

    if not all_sequences:
        return None, 0, 0

    transitions, _ = train_markov_from_notes(all_sequences)

    # Save trained model
    model_data = {
        "name": model_name,
        "type": "rhythm_markov_v2", # Versioning tag
        "trained_on": len(midi_files),
        "total_notes_analyzed": total_notes_analyzed,
        "unique_states": len(transitions),
        "timestamp": time.time(),
        "transitions": transitions,
        "files_used": [f.name for f in midi_files]
    }

    model_path = TRAINED_MODELS_DIR / f"{model_name}.json"
    with open(model_path, 'w') as f:
        json.dump(model_data, f, indent=2)

    print(f"✅ Model '{model_name}' trained! Learned {len(transitions)} unique rhythmic phrases.")
    return transitions, len(midi_files), total_notes_analyzed


def load_trained_model(model_name):
    """
    Load a previously trained Markov Chain model from JSON.
    Returns the transition matrix or None if not found.
    """
    model_path = TRAINED_MODELS_DIR / f"{model_name}.json"
    if model_path.exists():
        with open(model_path, 'r') as f:
            data = json.load(f)
            
        # V2 Model Support: Keys are strings like "60_0.25", no conversion needed
        if data.get("type") == "rhythm_markov_v2":
            return data["transitions"], data
            
        # Legacy V1 Support: Keys are strings of ints "60" -> convert to int
        transitions = {}
        for k, v in data["transitions"].items():
            try:
                transitions[int(k)] = {int(nk): nv for nk, nv in v.items()}
            except:
                pass # Skip partial bad data
        return transitions, data
    return None, None


def get_available_models():
    """
    List all available trained models.
    """
    models = []
    for model_file in TRAINED_MODELS_DIR.glob("*.json"):
        try:
            with open(model_file, 'r') as f:
                data = json.load(f)
            models.append({
                "name": data.get("name", model_file.stem),
                "trained_on": data.get("trained_on", 0),
                "total_notes": data.get("total_notes_analyzed", 0),
                "unique_notes": data.get("unique_notes", 0),
            })
        except Exception:
            pass
    return models


# ============================================================
# Markov Chain Melody Generator
# ============================================================
def build_markov_transitions(scale):
    """
    Build simple (note, duration) transitions from a scale for fallback.
    """
    transitions = {}
    durations = [0.25, 0.5, 0.5, 1.0] # Standard mix
    
    for note in scale:
        for dur in durations:
            state = f"{note}_{dur}"
            probs = {}
            for target in scale:
                # favor closer notes
                dist = abs(target - note)
                weight = 1.0 / (dist + 1)
                
                # Pick a random next duration
                next_dur = random.choice(durations)
                target_state = f"{target}_{next_dur}"
                
                probs[target_state] = weight
            
            # Normalize
            total = sum(probs.values())
            transitions[state] = {k: v/total for k,v in probs.items()}
            
    return transitions


def markov_melody(scale, num_notes=32, trained_model=None):
    """
    Generate a melody with (Note, Duration) tuples.
    Robustly handles both V2 (rhythm-aware) and V1 (legacy) models.
    """
    transitions = trained_model
    
    # Check model version
    is_v2_model = False
    if transitions:
        first_key = str(list(transitions.keys())[0])
        if "_" in first_key:
            is_v2_model = True
    
    # Fallback to scale-based transitions if no model provided
    if not transitions:
        transitions = build_markov_transitions(scale)
        is_v2_model = True # build_markov_transitions returns V2 format

    # Select start state
    if is_v2_model:
        possible_starts = list(transitions.keys())
        current_state = random.choice(possible_starts) if possible_starts else f"{scale[0]}_0.5"
    else:
        # Legacy V1: current_state is just the note number (int or str)
        current_state = random.choice(scale)

    melody = []
    
    # Generate initial note
    try:
        if is_v2_model and "_" in str(current_state):
            n_str, d_str = str(current_state).split('_')
            melody.append((int(n_str), float(d_str)))
        else:
            melody.append((int(current_state), 0.5))
    except (ValueError, AttributeError, IndexError):
        melody.append((random.choice(scale), 0.5))

    # Generate subsequent notes
    for _ in range(num_notes - 1):
        if current_state in transitions:
            probs = transitions[current_state]
            candidates = list(probs.keys())
            weights = list(probs.values())
            
            next_state = random.choices(candidates, weights=weights, k=1)[0]
            current_state = next_state
            
            # Parse next state into (note, duration)
            try:
                if is_v2_model and "_" in str(next_state):
                    n_str, d_str = str(next_state).split('_')
                    melody.append((int(n_str), float(d_str)))
                else:
                    # Legacy fallback: use fixed duration
                    melody.append((int(next_state), 0.5))
            except (ValueError, AttributeError, IndexError):
                # Fallback note if parsing fails
                n = random.choice(scale)
                melody.append((n, 0.5))
        else:
            # Dead end: pick accidental or random scale note
            note = random.choice(scale)
            dur = random.choice([0.25, 0.5, 1.0])
            current_state = f"{note}_{dur}" if is_v2_model else note
            melody.append((note, dur))
            
    return melody


# ============================================================
# Note Styling (Rhythm & Dynamics)
# ============================================================
def stylize_notes(melody_data, style, tempo, genre="pop", mood="happy"):
    """
    Apply genre-specific rhythm, dynamics, and phrasing.
    """
    styled = []
    if not melody_data: return []

    is_v2_data = isinstance(melody_data[0], tuple)
    mood_data = MOOD_SCALES.get(mood, MOOD_SCALES["happy"])
    dur_pool = mood_data.get("dur_pool", [0.5])

    for i, item in enumerate(melody_data):
        if is_v2_data:
            note, duration = item
        else:
            note, duration = item, random.choice(dur_pool)

        # 1. Accentuation / Velocity
        vel = random.randint(85, 110)
        if style == "complex" and i % 4 == 0: 
            vel = int(vel * 1.15) # Pulse accent
        
        # 2. Genre-Specific Phrasing
        if genre == "jazz":
            # Swing Feel: Use slightly different velocity for backbeats
            if i % 2 == 1: vel = int(vel * 0.85)
        
        elif genre == "electronic":
            # Gating: If note is long, chop it into rhythmic pulses
            if duration >= 0.75:
                pulses = 3 if duration == 0.75 else 4
                sub_dur = duration / pulses
                for _ in range(pulses):
                    styled.append((note, sub_dur, vel))
                continue 
        
        elif genre == "rock":
            # Doubling: Add octaves for thickness occasionally
            if random.random() > 0.8:
                styled.append((note - 12, duration, int(vel * 0.7)))

        styled.append((note, duration, min(127, vel)))
            
    return styled


# ============================================================
# Smart Model Auto-Selector
# ============================================================
def _auto_select_best_model():
    """
    Automatically select the best trained model based on the
    number of notes it was trained on (more data = better music).
    Returns model name or None.
    """
    models = get_available_models()
    if not models:
        return None
    # Pick the model with the most total analyzed notes
    best = max(models, key=lambda m: m.get("total_notes", 0))
    if best.get("total_notes", 0) > 0:
        print(f"🎯 Best model found: {best['name']} ({best['total_notes']} notes analyzed)")
        return best["name"]
    return None


# ============================================================
# Generate Notes Based on Mood
# ============================================================
def generate_notes(mood, style, tempo, duration="medium", trained_model_name=None, use_ai=False, genre="pop"):
    """
    Generate melody notes based on mood selection.
    - When Gemini AI is ON: Uses AI exclusively, ignoring trained models.
    - When Gemini AI is OFF: Auto-selects the best trained model for richest output.
    Returns: (styled_notes, engine_used) tuple
    """
    mood_data = MOOD_SCALES.get(mood, MOOD_SCALES["happy"])
    scale = mood_data["scale"]
    rhythm_feel = mood_data.get("rhythm_feel", "default")

    # Duration-based note count (scaled for ~4 notes per bar)
    duration_map = {
        "short": 24,       # ~6 bars
        "medium": 48,      # ~12 bars
        "long": 96,        # ~24 bars
        "extra_long": 144  # ~36 bars
    }
    num_notes = duration_map.get(duration, 48)
    if style == "complex":
        num_notes = int(num_notes * 1.5)

    melody = None
    engine_used = "Markov Chain"

    # ===== GEMINI AI MODE =====
    # When AI is ON, trained models are NOT used (Gemini handles everything)
    if use_ai:
        print(f"🚀 Gemini AI composing a {mood.upper()} melody...")
        melody = gemini_generate_melody(mood, style, tempo, scale, num_notes)
        if melody:
            print(f"✅ Gemini AI generated {len(melody)} notes with {mood} personality!")
            engine_used = "Gemini AI (Google)"
        else:
            print("⚠️ Gemini AI failed, falling back to Markov Chain")

    # ===== MARKOV CHAIN MODE (with smart model selection) =====
    if melody is None:
        trained_model = None

        # Smart Model Selection: auto-pick the best available model
        if trained_model_name and trained_model_name != "none":
            trained_model, model_info = load_trained_model(trained_model_name)
            if trained_model:
                print(f"🧠 Using selected model: {trained_model_name}")
                engine_used = f"Markov Chain + {trained_model_name} model"
        else:
            # Auto-pick: find the largest model (most musical knowledge)
            best_model_name = _auto_select_best_model()
            if best_model_name:
                trained_model, model_info = load_trained_model(best_model_name)
                if trained_model:
                    print(f"🎯 Auto-selected best model: {best_model_name}")
                    engine_used = f"Markov Chain + {best_model_name} (auto)"

        melody = markov_melody(scale, num_notes, trained_model)

    # NEW: Scale Mapping (Quantization)
    # Ensure generated notes fit the mood scale if not using specialized AI
    if engine_used != "Gemini AI":
        mapped_melody = []
        for note, duration in melody:
            # Find nearest note in the selected mood scale
            nearest = min(scale, key=lambda s_note: abs(s_note - note))
            mapped_melody.append((nearest, duration))
        melody = mapped_melody

    # Apply style (rhythm & dynamics) with mood-specific rhythm feel
    styled_notes = stylize_notes(melody, style, tempo, genre=genre, mood=mood)

    return styled_notes, engine_used


# ============================================================
# Gemini AI Melody Generator
# ============================================================
def gemini_generate_melody(mood, style, tempo, scale, num_notes):
    """
    Use Google Gemini AI to generate an intelligent melody.
    Each mood has a DEEP PERSONALITY that creates truly unique compositions.
    Returns a list of MIDI note numbers, or None if API fails.
    """
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    scale_note_names = [note_names[n % 12] + str(n // 12 - 1) for n in scale]

    # Deep Mood Personality System
    mood_personalities = {
        "happy": {
            "character": "You are a joyful street musician playing on a sunny day in Paris.",
            "theory": """
- Use MAJOR scale intervals exclusively (major 2nds, major 3rds, perfect 5ths)
- Create UPWARD melodic motion as the dominant direction (70% ascending)
- Add playful syncopation: short-short-LONG note groupings
- Use call-and-response patterns (ask a phrase, answer it higher)
- Include grace notes and ornamental runs between main notes
- End phrases on the tonic or major 3rd for brightness
- Tempo feel: Light, bouncy, dance-like
- Think: Wedding celebration, sunrise, children laughing""",
            "temperature": 0.9,
            "avoid": "Avoid minor intervals, descending sequences longer than 3 notes, and chromatic movement."
        },
        "sad": {
            "character": "You are a lonely pianist playing in an empty cathedral at midnight.",
            "theory": """
- Use NATURAL MINOR scale with emphasis on minor 3rds and minor 6ths
- Create predominantly DOWNWARD melodic motion (65% descending)
- Use LONG sustained notes followed by silence (rest) — breathing pauses
- Include repeated single notes (like crying, 2-3 repetitions)
- Create sighing motifs: step down, leap up small, step down further
- Use the flatted 7th degree for deep melancholy
- End phrases on the 5th or minor 3rd (unresolved, yearning)
- Tempo feel: Extremely slow, rubato-like, with sudden pauses
- Think: Rainy window, farewell, nostalgia, loss""",
            "temperature": 0.6,
            "avoid": "Avoid major 3rds, upbeat rhythmic patterns, and bright upper register jumps."
        },
        "calm": {
            "character": "You are a zen monk creating ambient soundscapes in a mountain monastery.",
            "theory": """
- Use PENTATONIC scale ONLY — no dissonant intervals at all
- Create MINIMAL movement between notes (steps of 1-2 scale degrees max)
- Use WIDE SPACING between phrases — lots of silence and breathing room
- Repeat motifs gently with tiny variations (like ripples in still water)
- Keep notes in the MIDDLE register (avoid extreme highs and lows)
- Use octave unisons for meditative resonance
- Create a sense of floating — no strong rhythmic pulse
- End every major phrase on the root note for deep peace
- Tempo feel: Weightless, suspended, like clouds drifting
- Think: Meditation, spa, gentle rainfall, deep breathing, nature""",
            "temperature": 0.5,
            "avoid": "Avoid large leaps, fast note sequences, rhythmic drive, and chromatic tension."
        },
        "energetic": {
            "character": "You are an EDM producer creating a festival anthem at a massive arena.",
            "theory": """
- Use DORIAN or MIXOLYDIAN mode for edgy power
- Create STRONG, DRIVING rhythmic patterns with repetitive motifs
- Use OCTAVE JUMPS and power 5ths aggressively
- Build tension with ascending sequences that climb relentlessly
- Use rapid repeated notes (machine-gun style) for intensity
- Create builds: start low, ascend through 2+ octaves
- Add syncopated off-beat accents for groove
- Use the b7 for bluesy power
- End phrases on strong beats with authority (root or 5th)
- Tempo feel: Pounding, relentless, adrenaline-fueled
- Think: Racing, fighting, victory, extreme sports, rave""",
            "temperature": 0.95,
            "avoid": "Avoid long sustained notes, gentle pentatonic movement, and soft dynamics."
        }
    }

    personality = mood_personalities.get(mood, mood_personalities["happy"])

    prompt = f"""{personality['character']}

Generate a melody as a JSON array of {num_notes} MIDI note numbers.

MUSICAL CONTEXT:
- Mood: {mood.upper()} — this is the SOUL of the piece
- Style Complexity: {style}
- Tempo: {tempo} BPM
- Available notes (MIDI): {scale}
- Note names: {', '.join(scale_note_names)}

MUSIC THEORY DIRECTIVES:
{personality['theory']}

STRICT CONSTRAINTS:
1. Output ONLY a JSON array of integers. No text, no explanation.
2. Use ONLY notes from: {scale}
3. {personality['avoid']}
4. The melody MUST have: intro (4-8 notes), development (middle section), climax (peak), resolution (ending).
5. {'Use repetitive patterns and motifs for catchiness.' if style == 'simple' else 'Add ornamental variations, countermelodies, and complex intervallic relationships.'}

Output exactly {num_notes} MIDI numbers. Format: [67, 69, 71, ...]"""

    try:
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": personality.get("temperature", 0.8),
                "maxOutputTokens": num_notes * 5,
                "responseMimeType": "application/json"
            }
        }

        response = http_requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=30)

        if response.status_code != 200:
            print(f"\u26a0\ufe0f Gemini API error: {response.status_code} - {response.text[:300]}")
            return None

        result = response.json()
        content = result["candidates"][0]["content"]["parts"][0]["text"].strip()

        # Clean up the response (remove markdown code blocks if present)
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            content = content.rsplit("```", 1)[0]
        content = content.strip()

        # Parse the JSON array
        melody = json.loads(content)

        # Handle case where Gemini wraps in an object
        if isinstance(melody, dict):
            # Try common key names
            for key in ['melody', 'notes', 'midi_notes', 'data']:
                if key in melody:
                    melody = melody[key]
                    break
            else:
                # Take first list value found
                for v in melody.values():
                    if isinstance(v, list):
                        melody = v
                        break

        if not isinstance(melody, list):
            print("\u26a0\ufe0f Gemini returned non-list response")
            return None

        # Validate and clamp notes
        valid_melody = []
        for note in melody:
            if isinstance(note, (int, float)):
                n = int(note)
                n = max(0, min(127, n))
                valid_melody.append(n)

        if len(valid_melody) < 4:
            print(f"\u26a0\ufe0f Gemini returned too few valid notes: {len(valid_melody)}")
            return None

        # Pad or trim to exact num_notes
        while len(valid_melody) < num_notes:
            valid_melody.extend(valid_melody[:num_notes - len(valid_melody)])
        valid_melody = valid_melody[:num_notes]

        return valid_melody

    except http_requests.exceptions.Timeout:
        print("\u26a0\ufe0f Gemini API timed out")
        return None
    except json.JSONDecodeError as e:
        print(f"\u26a0\ufe0f Failed to parse Gemini response as JSON: {e}")
        return None
    except Exception as e:
        print(f"\u26a0\ufe0f Gemini AI error: {e}")
        return None


# ============================================================
# Create Multi-Track MIDI File (Full BGM Style)
# ============================================================
def _build_chord(root, scale, chord_type="triad"):
    """Build a chord from a root note using the given scale."""
    # Find closest scale notes to form a chord
    scale_sorted = sorted(set(scale))
    # Find root index in scale
    root_idx = None
    for i, n in enumerate(scale_sorted):
        if n % 12 == root % 12:
            root_idx = i
            break
    if root_idx is None:
        root_idx = 0
        root = scale_sorted[0]

    chord = [root]
    # Add third (2 scale steps up)
    if root_idx + 2 < len(scale_sorted):
        chord.append(scale_sorted[root_idx + 2])
    elif len(scale_sorted) > 2:
        chord.append(scale_sorted[2])
    # Add fifth (4 scale steps up)
    if root_idx + 4 < len(scale_sorted):
        chord.append(scale_sorted[root_idx + 4])
    elif len(scale_sorted) > 4:
        chord.append(scale_sorted[4])
    # Optional seventh for jazz
    if chord_type == "seventh" and root_idx + 6 < len(scale_sorted):
        chord.append(scale_sorted[root_idx + 6])

    return [max(0, min(127, n)) for n in chord]


def _generate_chord_progression(genre, scale, total_duration):
    """Generate genre-specific chord progressions."""
    scale_sorted = sorted(set(scale))
    
    # Define genre-specific progressions (using scale degree indices)
    progressions = {
        "pop": [
            [0, 3, 5, 4], # I-IV-vi-V
            [0, 4, 5, 3], # I-V-vi-IV
        ],
        "rock": [
            [0, 4, 3, 0], # I-V-IV-I
            [0, 3, 4, 3], # I-IV-V-IV
        ],
        "jazz": [
            [1, 4, 0, 5], # ii-V-I-vi
            [0, 5, 1, 4], # I-vi-ii-V
        ],
        "electronic": [
            [0, 0, 5, 4], # I-I-vi-V
            [0, 4, 0, 3], # I-V-I-IV
        ]
    }
    
    genre_progs = progressions.get(genre, progressions["pop"])
    pattern = random.choice(genre_progs)
    
    chords = []
    for deg in pattern:
        if deg < len(scale_sorted):
            root = scale_sorted[deg]
        else:
            root = scale_sorted[deg % len(scale_sorted)]
        chords.append(root)
    return chords


def create_midi(notes, genre, tempo, output_path, mood_scale=None):
    """
    Create a multi-track BGM-style MIDI file:
    - Track 0: Lead Melody
    - Track 1: Chord Pad (sustained chords)
    - Track 2: Bassline (root notes, proper movement)
    - Track 3: Arpeggio (broken chord patterns)
    - Track 4: Drums (genre-specific patterns)
    """
    midi = MIDIFile(5)  # 5 tracks for full BGM
    genre_data = GENRE_INSTRUMENTS.get(genre, GENRE_INSTRUMENTS["pop"])

    total_duration = sum(d for _, d, _ in notes)

    # Determine scale from notes if not provided
    if mood_scale is None:
        unique_notes = sorted(set(n for n, _, _ in notes))
        mood_scale = unique_notes if unique_notes else [60, 62, 64, 65, 67, 69, 71, 72]

    # Generate chord progression
    chord_roots = _generate_chord_progression(genre, mood_scale, total_duration)

    # ─── Track 0: Lead Melody ───
    track, channel = 0, 0
    midi.addTrackName(track, 0, "Lead Melody")
    midi.addTempo(track, 0, tempo)
    midi.addProgramChange(track, channel, 0, genre_data["melody"])

    current_time = 0
    for note, duration, velocity in notes:
        # Subtle Velocity accents based on genre
        if genre == "electronic":
            velocity = int(velocity * 1.1) # Louder lead
        elif genre == "jazz":
            velocity = int(velocity * 0.9) # Softer phrasing
            
        note = max(0, min(127, note))
        velocity = max(1, min(127, velocity))
        midi.addNote(track, channel, note, current_time, duration, velocity)
        current_time += duration

    # ─── Track 1: Chord Pad ───
    track, channel = 1, 1
    midi.addTrackName(track, 0, "Chord Pad")
    midi.addTempo(track, 0, tempo)
    midi.addProgramChange(track, channel, 0, genre_data["pad"])

    chord_duration = max(2.5, total_duration / (len(chord_roots) * 2))
    current_time = 0
    chord_idx = 0
    while current_time < total_duration:
        root = chord_roots[chord_idx % len(chord_roots)]
        
        # Voicing Style
        if genre == "rock":
            # Power chords: Root + 5th + Octave (heavy sound)
            chord_notes = [root, root + 7, root + 12]
        elif genre == "jazz":
            chord_notes = _build_chord(root, mood_scale, "seventh")
        else:
            chord_notes = _build_chord(root, mood_scale, "triad")

        # Place chord in appropriate register
        for cn in chord_notes:
            pad_note = cn
            # Pull to mid register
            while pad_note > 68: pad_note -= 12
            while pad_note < 44: pad_note += 12
            pad_note = max(0, min(127, pad_note))
            pad_vel = random.randint(40, 65)
            dur = min(chord_duration, total_duration - current_time)
            if dur > 0:
                midi.addNote(track, channel, pad_note, current_time, dur, pad_vel)

        current_time += chord_duration
        chord_idx += 1

    # ─── Track 2: Bassline ───
    track, channel = 2, 2
    midi.addTrackName(track, 0, "Bassline")
    midi.addTempo(track, 0, tempo)
    midi.addProgramChange(track, channel, 0, genre_data["bass"])

    # Build a proper bassline following the chord roots
    # Bass Patterns
    bass_patterns = {
        "pop":        [1.0, 1.0, 1.0, 1.0],      # Solid quarters
        "rock":       [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], # Driving eighths
        "jazz":       [1.0, 1.0, 1.0, 1.0],      # Standard Walk
        "electronic": [0.75, 0.25, 0.75, 0.25],  # Syncopated/Gallop
    }
    bass_rhythm = bass_patterns.get(genre, [1.0] * 4)

    current_time = 0
    chord_idx = 0
    bar_duration = chord_duration
    while current_time < total_duration:
        root = chord_roots[chord_idx % len(chord_roots)]
        # Bass in low register (octave 2-3)
        bass_root = root
        while bass_root > 55:
            bass_root -= 12
        while bass_root < 36:
            bass_root += 12
        bass_root = max(0, min(127, bass_root))

        # Play bass pattern within this chord
        bar_time = 0
        for dur in bass_rhythm:
            if current_time + bar_time >= total_duration:
                break
            # Alternate between root and fifth for movement
            if random.random() < 0.7:
                bn = bass_root
            else:
                bn = max(0, min(127, bass_root + 7))  # Fifth
            vel = random.randint(70, 100)
            actual_dur = min(dur, total_duration - current_time - bar_time)
            if actual_dur > 0:
                midi.addNote(track, channel, bn, current_time + bar_time, actual_dur, vel)
            bar_time += dur

        current_time += bar_duration
        chord_idx += 1

    # ─── Track 3: Arpeggio ───
    track, channel = 3, 3
    midi.addTrackName(track, 0, "Arpeggio")
    midi.addTempo(track, 0, tempo)
    # Use a bright/bell-like instrument for arpeggios
    arp_instruments = {"pop": 88, "rock": 27, "jazz": 11, "electronic": 81}
    midi.addProgramChange(track, channel, 0, arp_instruments.get(genre, 88))

    # Genre-Specific Arpeggio Logic
    arp_settings = {
        "pop":        {"speed": 0.5,   "pattern": "up-down"},
        "rock":       {"speed": 0.5,   "pattern": "up"},
        "jazz":       {"speed": 0.66,  "pattern": "random"}, # Swing-ish
        "electronic": {"speed": 0.25,  "pattern": "gated"}
    }
    settings = arp_settings.get(genre, arp_settings["pop"])
    arp_dur = settings["speed"]

    current_time = 0
    chord_idx = 0
    while current_time < total_duration:
        root = chord_roots[chord_idx % len(chord_roots)]
        chord_notes = _build_chord(root, mood_scale)

        # Place arpeggio in mid-upper register (octave 4-5)
        arp_notes = []
        for cn in chord_notes:
            an = cn
            while an > 80: an -= 12
            while an < 64: an += 12
            arp_notes.append(max(0, min(127, an)))

        # Define patterns
        if settings["pattern"] == "up-down":
            pattern = arp_notes + list(reversed(arp_notes[1:-1]))
        elif settings["pattern"] == "gated":
            # Just Root and Fifth pulsing
            pattern = [arp_notes[0], arp_notes[0], arp_notes[min(2, len(arp_notes)-1)]]
        else:
            pattern = arp_notes
            
        if not pattern: pattern = [72]

        bar_time = 0
        p_idx = 0
        chord_step = 2.0 # Wait 2 beats per chord group
        while bar_time < chord_step:
            if current_time + bar_time >= total_duration:
                break
            
            an = pattern[p_idx % len(pattern)]
            vel = random.randint(60, 95)
            # Staccato feel
            midi.addNote(track, channel, an, current_time + bar_time, arp_dur * 0.8, vel)
            
            bar_time += arp_dur
            p_idx += 1

        current_time += chord_step
        chord_idx = int(current_time / chord_duration)

    # ─── Track 4: Drums (genre-specific) ───
    track, channel = 4, 9  # Channel 9 = percussion
    midi.addTrackName(track, 0, "Drums")
    midi.addTempo(track, 0, tempo)

    # General MIDI percussion
    KICK = 36
    SNARE = 38
    CLAP = 39
    HH_CLOSED = 42
    HH_OPEN = 46
    CRASH = 49
    RIDE = 51
    TOM_HIGH = 50
    TOM_LOW = 45
    TAMBOURINE = 54
    SHAKER = 70

    current_time = 0
    beat_count = 0
    beat_dur = 1.0

    while current_time < total_duration:
        measure_pos = beat_count % 4
        bar_num = beat_count // 4

        if genre == "pop":
            # Pop: Kick-Snare with steady hi-hats and occasional fills
            if measure_pos == 0:
                midi.addNote(track, channel, KICK, current_time, 0.5, 100)
            elif measure_pos == 1:
                midi.addNote(track, channel, SNARE, current_time, 0.5, 85)
            elif measure_pos == 2:
                midi.addNote(track, channel, KICK, current_time, 0.5, 90)
                midi.addNote(track, channel, KICK, current_time + 0.5, 0.25, 70)
            elif measure_pos == 3:
                midi.addNote(track, channel, SNARE, current_time, 0.5, 90)
            # Hi-hats on eighth notes
            midi.addNote(track, channel, HH_CLOSED, current_time, 0.25, 60)
            midi.addNote(track, channel, HH_CLOSED, current_time + 0.5, 0.25, 50)
            # Crash on bar 1
            if beat_count % 16 == 0:
                midi.addNote(track, channel, CRASH, current_time, 1.0, 80)

        elif genre == "rock":
            # Driving Rock Beat (Eighth notes on Hi-Hat, Kick/Snare interplay)
            # Kick on 1, 3 and variations
            if measure_pos == 0:
                midi.addNote(track, channel, KICK, current_time, 0.5, 115)
            elif measure_pos == 2.0:
                midi.addNote(track, channel, KICK, current_time, 0.5, 110)
            elif measure_pos == 2.5 and random.random() > 0.6: # Syncopated kick
                midi.addNote(track, channel, KICK, current_time, 0.25, 95)
            
            # Snare on 2 and 4 (Backbeat) with ghost notes
            if measure_pos in [1.0, 3.0]:
                midi.addNote(track, channel, SNARE, current_time, 0.5, 115)
            elif random.random() > 0.85: # Random ghost snare
                midi.addNote(track, channel, SNARE, current_time, 0.1, 45)

            # Driving Hi-Hats (Closed, accented on beats)
            midi.addNote(track, channel, HH_CLOSED, current_time, 0.25, 90 if measure_pos % 1 == 0 else 70)
            midi.addNote(track, channel, HH_CLOSED, current_time + 0.5, 0.25, 60)

            # Crash at start of phrase
            if beat_count % 16 == 0:
                midi.addNote(track, channel, CRASH, current_time, 1.0, 100)

        elif genre == "jazz":
            # Jazz Swing Pattern (Ride Cymbal focus)
            # Ride Pattern: Ding ... Ding-a-Ding
            midi.addNote(track, channel, RIDE, current_time, 0.25, 80 if measure_pos % 1 == 0 else 60)
            if measure_pos % 1 == 0.0: # On the beat
                 pass # Already hit
            else: # Off-beat swing
                 midi.addNote(track, channel, RIDE, current_time + 0.66, 0.15, 55) # Swing feel

            # Feathering Kick (very soft on all beats)
            if random.random() > 0.6:
                midi.addNote(track, channel, KICK, current_time, 0.25, 40)
            
            # Hi-Hat Pedal on 2 and 4 (The "Chick" sound)
            if measure_pos in [1.0, 3.0]:
                midi.addNote(track, channel, 44, current_time, 0.25, 60) # G#1 = Pedal Hi-Hat

            # Snare Comping (Random interplay)
            if random.random() > 0.8:
                offset = random.choice([0, 0.33, 0.66])
                midi.addNote(track, channel, SNARE, current_time + offset, 0.2, random.randint(30, 70))

        elif genre == "electronic":
            # Four-on-the-Floor (House/Techno style)
            midi.addNote(track, channel, KICK, current_time, 0.9, 120) # Heavy Kick on every beat
            
            # Off-beat Open Hi-Hat ("Untz-tss-Untz-tss")
            midi.addNote(track, channel, HH_OPEN, current_time + 0.5, 0.4, 95)
            
            # Clap on 2 and 4 (layered with kick)
            if measure_pos in [1.0, 3.0]:
                 midi.addNote(track, channel, CLAP, current_time, 0.5, 100)
            
            # Random percussion/fill elements
            if beat_count % 16 == 15: # End of phrase fill
                 midi.addNote(track, channel, SNARE, current_time, 0.15, 100)
                 midi.addNote(track, channel, SNARE, current_time + 0.25, 0.15, 90)
                 midi.addNote(track, channel, SNARE, current_time + 0.5, 0.15, 110)
                 midi.addNote(track, channel, SNARE, current_time + 0.75, 0.15, 110)

        else: # Pop / Default
            # The "Money Beat" (Simple, effective)
            if measure_pos in [0.0, 2.0]: 
                midi.addNote(track, channel, KICK, current_time, 0.5, 110)
            elif measure_pos == 2.5: # Pop syncopation
                midi.addNote(track, channel, KICK, current_time, 0.25, 100)

            if measure_pos in [1.0, 3.0]:
                midi.addNote(track, channel, SNARE, current_time, 0.5, 110)
            
            # Straight 8th note Hi-Hats
            midi.addNote(track, channel, HH_CLOSED, current_time, 0.25, 80)
            midi.addNote(track, channel, HH_CLOSED, current_time + 0.5, 0.25, 70)

        current_time += beat_dur
        beat_count += 1

    # Write MIDI file
    with open(str(output_path), "wb") as f:
        midi.writeFile(f)

    return output_path


# ============================================================
# NumPy-Powered MIDI-to-WAV Synthesizer (Fast!)
# ============================================================
import numpy as np


def midi_note_to_freq(note):
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def np_generate_tone(freq, duration, sample_rate=44100, volume=0.3, wave_type="sine"):
    """Generate a tone using numpy for speed. Returns numpy array."""
    num_samples = int(sample_rate * duration)
    if num_samples == 0:
        return np.array([], dtype=np.float64)

    t = np.arange(num_samples, dtype=np.float64) / sample_rate
    phase = 2 * np.pi * freq * t

    # Generate waveform
    if wave_type == "sine":
        samples = np.sin(phase)
    elif wave_type == "square":
        samples = np.sign(np.sin(phase)) * 0.5
    elif wave_type == "saw":
        samples = 2.0 * (freq * t - np.floor(0.5 + freq * t)) * 0.4
    elif wave_type == "triangle":
        samples = (2.0 * np.abs(2.0 * (freq * t - np.floor(freq * t + 0.5))) - 1.0) * 0.6
    else:
        samples = np.sin(phase)

    # ADSR envelope
    attack = min(int(0.01 * sample_rate), num_samples // 4)
    decay = min(int(0.05 * sample_rate), num_samples // 4)
    release = min(int(0.1 * sample_rate), num_samples // 3)

    envelope = np.full(num_samples, 0.7)
    if attack > 0:
        envelope[:attack] = np.linspace(0, 1, attack)
    if decay > 0:
        ad = attack + decay
        envelope[attack:ad] = np.linspace(1.0, 0.7, decay)
    if release > 0:
        envelope[-release:] = np.linspace(0.7, 0, release)

    return samples * envelope * volume


def np_generate_kick(duration=0.2, sample_rate=44100, volume=0.5):
    """Generate a kick drum sound using numpy."""
    num_samples = int(sample_rate * duration)
    t = np.arange(num_samples, dtype=np.float64) / sample_rate
    freq = 150 * np.exp(-t * 20) + 50
    # Phase = integral of freq
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    samples = np.sin(phase)
    envelope = np.maximum(0, 1.0 - t / duration) ** 2
    return samples * envelope * volume


def np_generate_snare(duration=0.15, sample_rate=44100, volume=0.4):
    """Generate a snare drum sound using numpy."""
    num_samples = int(sample_rate * duration)
    t = np.arange(num_samples, dtype=np.float64) / sample_rate
    tone = np.sin(2 * np.pi * 200 * t) * 0.5
    noise = np.random.uniform(-1, 1, num_samples) * 0.7
    samples = tone + noise
    envelope = np.maximum(0, 1.0 - t / duration) ** 1.5
    return samples * envelope * volume


def np_generate_hihat(duration=0.05, sample_rate=44100, volume=0.2):
    """Generate a hi-hat sound using numpy."""
    num_samples = int(sample_rate * duration)
    t = np.arange(num_samples, dtype=np.float64) / sample_rate
    samples = np.random.uniform(-1, 1, num_samples)
    envelope = np.maximum(0, 1.0 - t / duration) ** 4
    return samples * envelope * volume


def np_generate_noise(duration=0.1, sample_rate=44100, volume=0.3):
    """Generate a noise hit using numpy."""
    num_samples = int(sample_rate * duration)
    i = np.arange(num_samples, dtype=np.float64)
    samples = np.random.uniform(-1, 1, num_samples)
    envelope = np.maximum(0, 1.0 - i / num_samples) ** 3
    return samples * envelope * volume


def get_wave_type_for_genre(genre):
    """Select wave type based on genre for appropriate timbre."""
    wave_types = {
        "pop": "triangle",
        "rock": "saw",
        "jazz": "sine",
        "electronic": "square",
    }
    return wave_types.get(genre, "sine")


def synthesize_midi_to_wav(midi_path, wav_path, genre="pop"):
    """
    NumPy-powered MIDI-to-WAV synthesizer.
    Reads a MIDI file and synthesizes it to WAV using vectorized operations.
    Much faster than pure Python loops.
    """
    sample_rate = 44100
    mid = mido.MidiFile(str(midi_path))

    # Get total duration in seconds
    total_duration = mid.length + 1.0
    total_samples = int(sample_rate * total_duration)

    # Initialize output buffer as numpy array
    audio_buffer = np.zeros(total_samples, dtype=np.float64)

    wave_type = get_wave_type_for_genre(genre)
    bass_wave = "sine"

    for track_idx, track in enumerate(mid.tracks):
        current_time = 0.0
        active_notes = {}
        tempo_val = get_tempo_from_midi(mid)

        for msg in track:
            current_time += mido.tick2second(msg.time, mid.ticks_per_beat, tempo_val)

            if msg.type == 'note_on' and msg.velocity > 0:
                note = msg.note
                velocity = msg.velocity
                channel = msg.channel
                volume = (velocity / 127.0) * 0.25

                if channel == 9:
                    # Percussion - generate immediately
                    if note == 36:
                        tone = np_generate_kick(0.2, sample_rate, volume * 1.5)
                    elif note == 38:
                        tone = np_generate_snare(0.15, sample_rate, volume * 1.2)
                    elif note in (42, 44, 46):
                        tone = np_generate_hihat(0.05, sample_rate, volume * 0.8)
                    else:
                        tone = np_generate_noise(0.1, sample_rate, volume * 0.5)

                    start = int(current_time * sample_rate)
                    end = min(start + len(tone), total_samples)
                    if start < total_samples and start >= 0:
                        length = end - start
                        audio_buffer[start:end] += tone[:length]
                else:
                    active_notes[(channel, note)] = (current_time, volume)

            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in active_notes:
                    start_time, volume = active_notes.pop(key)
                    duration = max(0.05, current_time - start_time)
                    freq = midi_note_to_freq(msg.note)
                    
                    # Choose wave type based on track
                    if track_idx == 2:  # Bass
                        wt = "sine"
                    elif track_idx == 1: # Pad
                        wt = "triangle"
                    elif track_idx == 3: # Arpeggio
                        wt = "square" if genre == "electronic" else "sine"
                    else: # Lead Melody
                        wt = wave_type

                    tone = np_generate_tone(freq, duration, sample_rate, volume, wt)
                    start = int(start_time * sample_rate)
                    end = min(start + len(tone), total_samples)
                    if start < total_samples and start >= 0:
                        length = end - start
                        audio_buffer[start:end] += tone[:length]

        # Handle remaining active notes
        for key, (start_time, volume) in active_notes.items():
            channel, note = key
            duration = max(0.05, min(2.0, total_duration - start_time - 0.5))
            freq = midi_note_to_freq(note)
            
            if track_idx == 2: wt = "sine"
            elif track_idx == 1: wt = "triangle"
            elif track_idx == 3: wt = "square" if genre == "electronic" else "sine"
            else: wt = wave_type
            
            tone = np_generate_tone(freq, duration, sample_rate, volume, wt)
            start = int(start_time * sample_rate)
            end = min(start + len(tone), total_samples)
            if start < total_samples and start >= 0:
                length = end - start
                audio_buffer[start:end] += tone[:length]

    # Normalize
    max_val = np.max(np.abs(audio_buffer))
    if max_val > 0:
        audio_buffer = audio_buffer * (0.85 / max_val)

    # Clip and convert to 16-bit
    audio_buffer = np.clip(audio_buffer, -1.0, 1.0)
    int_samples = (audio_buffer * 32767).astype(np.int16)

    # Write WAV file
    with wave.open(str(wav_path), 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(int_samples.tobytes())

    return wav_path


def get_tempo_from_midi(mid):
    """Extract tempo from MIDI file (default 500000 = 120 BPM)."""
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                return msg.tempo
    return 500000  # Default 120 BPM


# ============================================================
# Convert MIDI to WAV (with FluidSynth or fallback)
# ============================================================
def convert_midi_to_wav(midi_path, wav_path, genre="pop"):
    """
    Convert MIDI to WAV. Tries FluidSynth first, falls back to
    pure Python synthesizer if not available.
    """
    if FLUIDSYNTH_AVAILABLE and SOUNDFONT_PATH.exists():
        try:
            from midi2audio import FluidSynth
            fs = FluidSynth(str(SOUNDFONT_PATH))
            fs.midi_to_audio(str(midi_path), str(wav_path))
            print(f"✅ Converted with FluidSynth: {wav_path}")
            return wav_path
        except Exception as e:
            print(f"⚠️ FluidSynth failed: {e}, using fallback synthesizer")

    # Fallback: Pure Python synthesizer
    print(f"🎹 Synthesizing with built-in engine...")
    synthesize_midi_to_wav(midi_path, wav_path, genre)
    print(f"✅ Synthesized: {wav_path}")
    return wav_path


# ============================================================
# API Routes
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    """Serve the landing page with music generation form."""
    models = get_available_models()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "trained_models": models
    })


@app.post("/api/generate")
async def api_generate_music(params: dict):
    """JSON API for Spring Boot backend to trigger generation."""
    mood = params.get("mood", "happy").lower()
    genre = params.get("genre", "pop").lower()
    tempo = int(params.get("tempo", 120))
    style = params.get("style", "simple").lower()
    duration = params.get("duration", "medium")
    model_name = params.get("trained_model") if params.get("trained_model") != "none" else None
    use_ai = params.get("use_ai") == "on" or params.get("use_ai") is True

    # Note: Logic duplicated from generate_track but returning JSON
    timestamp = int(time.time())
    base_name = f"{mood}_{genre}_{tempo}bpm_{style}_{timestamp}"
    midi_path = OUTPUT_FOLDER / f"{base_name}.mid"
    wav_path = OUTPUT_FOLDER / f"{base_name}.wav"

    notes, engine_used = generate_notes(mood, style, tempo, duration, model_name, use_ai=use_ai, genre=genre)
    mood_data = MOOD_SCALES.get(mood, MOOD_SCALES["happy"])
    scale = mood_data["scale"]

    create_midi(notes, genre, tempo, midi_path, mood_scale=scale)
    convert_midi_to_wav(midi_path, wav_path, genre)

    return {
        "success": True,
        "filename": wav_path.name,
        "mood": mood,
        "genre": genre,
        "tempo": tempo,
        "style": style,
        "ai_engine": engine_used,
        "size_kb": round(wav_path.stat().st_size / 1024, 1)
    }


@app.post("/", response_class=HTMLResponse)
async def generate_track(
    request: Request,
    mood: str = Form(...),
    genre: str = Form(...),
    tempo: int = Form(...),
    style: str = Form(...),
    duration: str = Form("medium"),
    trained_model: str = Form("none"),
    use_ai: str = Form("off")
):
    try:
        # Validate inputs
        mood = mood.lower()
        genre = genre.lower()
        style = style.lower()
        tempo = max(60, min(200, tempo))
        model_name = trained_model if trained_model != "none" else None
        ai_enabled = use_ai == "on"

        print(f"🎵 NEW GENERATION REQUEST: {mood}, {genre}, {tempo} BPM, {style}, {duration}, model={model_name}")

        # Generate unique filename with timestamp
        timestamp = int(time.time())
        base_name = f"{mood}_{genre}_{tempo}bpm_{style}_{timestamp}"
        midi_path = OUTPUT_FOLDER / f"{base_name}.mid"
        wav_path = OUTPUT_FOLDER / f"{base_name}.wav"

        # Step 1: Generate notes
        print("🛠️ Step 1: Generating notes...")
        notes, engine_used = generate_notes(mood, style, tempo, duration, model_name, use_ai=ai_enabled, genre=genre)
        print(f"✅ Generated {len(notes)} notes")

        # Get mood scale
        mood_data = MOOD_SCALES.get(mood, MOOD_SCALES["happy"])
        scale = mood_data["scale"]

        # Step 2: Create MIDI
        print("🛠️ Step 2: Creating multi-track MIDI...")
        create_midi(notes, genre, tempo, midi_path, mood_scale=scale)

        # Step 3: Convert MIDI to WAV
        print("🛠️ Step 3: Converting MIDI to WAV...")
        convert_midi_to_wav(midi_path, wav_path, genre)

        # Step 4: Build Markov visualization data
        print("🛠️ Step 4: Building visualization data...")
        mood_data = MOOD_SCALES.get(mood, MOOD_SCALES["happy"])
        scale = mood_data["scale"]
        transitions = build_markov_transitions(scale)
        
        # Determine current model data if any
        if model_name:
             t_model, _ = load_trained_model(model_name)
             if t_model:
                 transitions = t_model

        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        markov_viz = []
        for src_key, targets in list(transitions.items())[:15]:
            try:
                src_note = int(str(src_key).split('_')[0])
                src_name = note_names[src_note % 12] + str(src_note // 12 - 1)
                for tgt_key, prob in sorted(targets.items(), key=lambda x: -x[1])[:3]:
                    tgt_note = int(str(tgt_key).split('_')[0])
                    tgt_name = note_names[tgt_note % 12] + str(tgt_note // 12 - 1)
                    markov_viz.append({"from": src_name, "to": tgt_name, "prob": round(prob * 100, 1)})
            except:
                continue
                
        wav_filename = wav_path.name
        midi_filename = midi_path.name
        genre_display = GENRE_INSTRUMENTS.get(genre, {}).get("name", genre.capitalize())
        mood_display = mood.capitalize()
        scale_info = MOOD_SCALES.get(mood, {}).get("description", "Custom scale")

        # Step 5: Save to SQL Backend
        print("🛠️ Step 5: Updating track history...")
        try:
            file_size = round(wav_path.stat().st_size / 1024, 1) if wav_path.exists() else 0
            http_requests.post("http://127.0.0.1:8081/api/tracks/save", json={
                "filename": wav_filename,
                "mood": mood,
                "genre": genre,
                "tempo": tempo,
                "style": style,
                "aiEngine": engine_used,
                "fileSizeKb": file_size
            }, timeout=2)
        except Exception as e:
            print(f"⚠️ Warning: Could not save to SQL history backend: {e}")

        print("✨ Generation complete! Rendering results...")
        return templates.TemplateResponse("result.html", {
            "request": request,
            "mood": mood_display,
            "genre": genre_display,
            "tempo": tempo,
            "style": style.capitalize(),
            "duration": duration.capitalize(),
            "filename": wav_filename,
            "midi_filename": midi_filename,
            "scale_info": scale_info,
            "markov_data": json.dumps(markov_viz),
            "num_notes": len(notes),
            "model_used": model_name or "Default (scale-based)",
            "ai_engine": engine_used,
        })

    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(f"❌ CRITICAL ERROR in generate_track:\n{err_msg}")
        return HTMLResponse(content=f"""
            <div style="background:#111; color:#ff4444; padding:20px; font-family:monospace; border:2px solid #ff0000; border-radius:10px;">
                <h1 style="color:#ff0000; margin-top:0;">🛑 500 | Internal Server Error</h1>
                <p>The AI Music Engine encountered a critical error during generation.</p>
                <hr style="border-color:#333;">
                <pre style="background:#000; padding:15px; overflow-x:auto;">{err_msg}</pre>
                <a href="/" style="background:#444; color:#fff; text-decoration:none; padding:10px 20px; border-radius:5px; display:inline-block;">⬅️ Go Back</a>
            </div>
        """, status_code=500)


@app.get("/download/{filename}")
async def download_file(filename: str):
    """Serve generated WAV or MIDI files for download."""
    file_path = OUTPUT_FOLDER / filename
    if file_path.exists() and file_path.suffix in (".wav", ".mid"):
        media_type = "audio/wav" if file_path.suffix == ".wav" else "audio/midi"
        return FileResponse(
            str(file_path),
            media_type=media_type,
            filename=filename
        )
    return HTMLResponse("<h1>File not found</h1>", status_code=404)


# ============================================================
# Training Routes
# ============================================================
@app.get("/train", response_class=HTMLResponse)
async def training_page(request: Request):
    """Serve the model training page."""
    models = get_available_models()
    return templates.TemplateResponse("train.html", {
        "request": request,
        "models": models
    })


@app.post("/train", response_class=HTMLResponse)
async def train_model(
    request: Request,
    model_name: str = Form(...),
    midi_files: list[UploadFile] = File(...)
):
    """
    Train a Markov Chain model from uploaded MIDI files.
    Saves uploaded files and trains the model.
    """
    # Create subfolder for this training set
    train_folder = TRAINING_DIR / model_name
    train_folder.mkdir(exist_ok=True)

    # Save uploaded MIDI files
    saved_files = []
    for midi_file in midi_files:
        if midi_file.filename and (midi_file.filename.endswith('.mid') or midi_file.filename.endswith('.midi')):
            dest = train_folder / midi_file.filename
            content = await midi_file.read()
            with open(dest, 'wb') as f:
                f.write(content)
            saved_files.append(dest.name)

    if not saved_files:
        models = get_available_models()
        return templates.TemplateResponse("train.html", {
            "request": request,
            "models": models,
            "error": "No valid MIDI files uploaded. Please upload .mid or .midi files."
        })

    # Train the model
    transitions, num_files, total_notes = train_from_midi_folder(train_folder, model_name)

    models = get_available_models()
    return templates.TemplateResponse("train.html", {
        "request": request,
        "models": models,
        "success": f"Model '{model_name}' trained successfully on {num_files} files ({total_notes:,} note transitions learned)!",
        "trained_files": saved_files
    })


# ============================================================
# Track History API
# ============================================================
@app.get("/history", response_class=HTMLResponse)
async def track_history(request: Request):
    """Display previously generated tracks from the Spring Boot SQL database."""
    tracks = []
    
    # Try fetching from Spring Boot SQL Backend first
    try:
        response = http_requests.get("http://127.0.0.1:8081/api/tracks/history", timeout=2)
        if response.status_code == 200:
            db_tracks = response.json()
            for t in db_tracks:
                wav_file = OUTPUT_FOLDER / t['filename']
                if wav_file.exists():
                    midi_file = OUTPUT_FOLDER / t['filename'].replace('.wav', '.mid')
                    tracks.append({
                        "filename": t['filename'],
                        "mood": t['mood'].lower(), # Keep lowercase for template matching
                        "genre": t['genre'].lower(),
                        "tempo": t['tempo'],
                        "style": t['style'],
                        "size_kb": t['fileSizeKb'],
                        "created": t['createdAt'].replace('T', ' ').split('.')[0] if t.get('createdAt') else "Recent",
                        "ai_engine": t.get('aiEngine', 'Markov Chain'),
                        "has_midi": midi_file.exists(),
                        "midi_filename": midi_file.name if midi_file.exists() else None
                    })
            if tracks:
                print("📋 Loaded history from SQL Backend")
                return templates.TemplateResponse("history.html", {"request": request, "tracks": tracks})
    except Exception as e:
        print(f"⚠️ Spring Boot connection failed, falling back to local file scan: {e}")

    # Fallback: Traditional folder scanning logic
    wav_files = sorted(OUTPUT_FOLDER.glob("*.wav"), key=os.path.getmtime, reverse=True)
    for wav_file in wav_files[:20]:
        name_parts = wav_file.stem.split('_')
        if len(name_parts) >= 4:
            track_info = {
                "filename": wav_file.name,
                "mood": name_parts[0].capitalize(),
                "genre": name_parts[1].capitalize(),
                "tempo": name_parts[2].replace('bpm', ''),
                "style": name_parts[3].capitalize(),
                "size_kb": round(wav_file.stat().st_size / 1024, 1),
                "created": time.strftime('%Y-%m-%d %H:%M', time.localtime(wav_file.stat().st_mtime))
            }
            midi_file = OUTPUT_FOLDER / (wav_file.stem + ".mid")
            track_info["has_midi"] = midi_file.exists()
            track_info["midi_filename"] = midi_file.name if midi_file.exists() else None
            tracks.append(track_info)

    return templates.TemplateResponse("history.html", {
        "request": request,
        "tracks": tracks
    })


# ============================================================
# API Endpoints (JSON)
# ============================================================
@app.get("/api/models")
async def api_models():
    """Return available trained models as JSON."""
    return JSONResponse(content=get_available_models())


@app.get("/api/history")
async def api_history():
    """Return track history as JSON."""
    tracks = []
    wav_files = sorted(OUTPUT_FOLDER.glob("*.wav"), key=os.path.getmtime, reverse=True)
    for wav_file in wav_files[:20]:
        parts = wav_file.stem.split('_')
        if len(parts) >= 4:
            tracks.append({
                "filename": wav_file.name,
                "mood": parts[0],
                "genre": parts[1],
                "tempo": parts[2].replace('bpm', ''),
                "style": parts[3],
                "size_kb": round(wav_file.stat().st_size / 1024, 1)
            })
    return JSONResponse(content=tracks)


@app.post("/api/clear_history")
async def clear_history():
    """Delete all generated tracks (Files + SQL Database)."""
    try:
        # 1. Clear SQL Backend (Spring Boot)
        try:
            http_requests.delete("http://127.0.0.1:8081/api/tracks/clear", timeout=3)
            print("📜 SQL History cleared in Spring Boot")
        except Exception as e:
            print(f"⚠️ Could not clear SQL backend: {e}")

        # 2. Clear Local Files
        count = 0
        for f in OUTPUT_FOLDER.glob("*"):
            if f.is_file():
                f.unlink()
                count += 1
        return JSONResponse(content={"success": True, "message": f"History cleared (SQL + {count} files)."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": str(e)})


# ============================================================
# Run Server
# ============================================================
if __name__ == "__main__":
    import uvicorn
    print("🎵 CoverComposer - AI Music Generator")
    print("=" * 45)
    print(f"📂 Output folder: {OUTPUT_FOLDER}")
    print(f"🧠 Trained models: {len(get_available_models())}")
    print(f"🎹 FluidSynth: {'Available' if FLUIDSYNTH_AVAILABLE else 'Using built-in synth'}")
    print(f"🌐 Starting server at http://127.0.0.1:8000")
    print("=" * 45)
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
