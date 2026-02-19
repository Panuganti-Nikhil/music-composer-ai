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
        # G Major, bright upper register (G4-G6)
        "scale": [67, 69, 71, 72, 74, 76, 78, 79, 81, 83, 84, 86, 88, 90, 91],
        "description": "Bright G Major scale (upper register)",
        "rhythm_feel": "bouncy"  # Short + syncopated
    },
    "sad": {
        # A Minor, low-mid register (A3-A5)
        "scale": [57, 59, 60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81],
        "description": "Emotional A Natural Minor scale",
        "rhythm_feel": "slow"  # Long, drawn-out notes
    },
    "calm": {
        # Eb Pentatonic, spaced-out dreamy intervals (Eb4-Eb6)
        "scale": [63, 65, 68, 70, 72, 75, 77, 80, 82, 84, 87, 89],
        "description": "Dreamy Eb Pentatonic (wide intervals)",
        "rhythm_feel": "flowing"  # Smooth, sustained
    },
    "energetic": {
        # D Dorian, punchy mid range (D4-D6)
        "scale": [62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83, 84, 86],
        "description": "Driving D Dorian mode",
        "rhythm_feel": "driving"  # Fast, rhythmic
    },
}


# ============================================================
# MIDI Training System — Learn from Real Music
# ============================================================
def extract_notes_from_midi(midi_path):
    """
    Extract note sequences from a MIDI file.
    Returns a list of MIDI note numbers.
    """
    try:
        mid = mido.MidiFile(str(midi_path))
        notes = []
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    notes.append(msg.note)
        return notes
    except Exception as e:
        print(f"⚠️ Error reading {midi_path}: {e}")
        return []


def train_markov_from_notes(note_sequences):
    """
    Train a Markov Chain transition matrix from note sequences.
    Takes a list of note lists and builds transition probabilities.
    """
    transition_counts = defaultdict(lambda: defaultdict(int))
    total_notes = 0

    for notes in note_sequences:
        for i in range(len(notes) - 1):
            current = notes[i]
            next_note = notes[i + 1]
            transition_counts[current][next_note] += 1
            total_notes += 1

    # Convert counts to probabilities
    transitions = {}
    for note, targets in transition_counts.items():
        total = sum(targets.values())
        transitions[int(note)] = {
            int(target): count / total
            for target, count in targets.items()
        }

    return transitions, total_notes


def train_from_midi_folder(folder_path, model_name="custom"):
    """
    Train a Markov Chain model from all MIDI files in a folder.
    Saves the trained model as a JSON file.
    """
    midi_files = list(Path(folder_path).glob("*.mid")) + list(Path(folder_path).glob("*.midi"))

    if not midi_files:
        return None, 0, 0

    all_sequences = []
    for midi_file in midi_files:
        notes = extract_notes_from_midi(midi_file)
        if notes:
            all_sequences.append(notes)

    if not all_sequences:
        return None, 0, 0

    transitions, total_notes = train_markov_from_notes(all_sequences)

    # Save trained model
    model_data = {
        "name": model_name,
        "trained_on": len(midi_files),
        "total_notes_analyzed": total_notes,
        "unique_notes": len(transitions),
        "timestamp": time.time(),
        "transitions": transitions,
        "files_used": [f.name for f in midi_files]
    }

    model_path = TRAINED_MODELS_DIR / f"{model_name}.json"
    with open(model_path, 'w') as f:
        json.dump(model_data, f, indent=2)

    print(f"✅ Model '{model_name}' trained on {len(midi_files)} files ({total_notes} notes)")
    return transitions, len(midi_files), total_notes


def load_trained_model(model_name):
    """
    Load a previously trained Markov Chain model from JSON.
    Returns the transition matrix or None if not found.
    """
    model_path = TRAINED_MODELS_DIR / f"{model_name}.json"
    if model_path.exists():
        with open(model_path, 'r') as f:
            data = json.load(f)
        # Convert string keys back to ints
        transitions = {}
        for note, targets in data["transitions"].items():
            transitions[int(note)] = {
                int(t): p for t, p in targets.items()
            }
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
    Build transition probability matrix for a given musical scale.
    Each note has probabilities of transitioning to other notes,
    weighted by musical interval relationships.
    """
    transitions = {}
    n = len(scale)

    for i, note in enumerate(scale):
        probs = {}
        for j, target in enumerate(scale):
            distance = abs(i - j)
            if distance == 0:
                weight = 0.05  # Low chance of repeating same note
            elif distance == 1:
                weight = 0.35  # High chance of stepwise motion
            elif distance == 2:
                weight = 0.25  # Good chance of skipping one step
            elif distance == 3:
                weight = 0.15  # Medium chance of thirds
            elif distance == 4:
                weight = 0.10  # Lower chance of fourths
            else:
                weight = 0.05  # Small chance of larger leaps

            # Add slight randomness for variation
            weight += random.uniform(0, 0.05)
            probs[target] = weight

        # Normalize probabilities
        total = sum(probs.values())
        for key in probs:
            probs[key] /= total

        transitions[note] = probs

    return transitions


def markov_melody(scale, num_notes=32, trained_model=None):
    """
    Generate a melody using Markov Chain transitions.
    If a trained_model is provided, uses learned transitions.
    Otherwise, builds transitions from the scale.
    Returns a list of MIDI note numbers.
    """
    if trained_model:
        transitions = trained_model
        # Filter to only use notes within the given scale range
        available_notes = list(transitions.keys())
        if not available_notes:
            transitions = build_markov_transitions(scale)
            available_notes = scale
    else:
        transitions = build_markov_transitions(scale)
        available_notes = scale

    # Increase starting note variety
    start_notes = [scale[0], scale[2], scale[4], scale[7]] if len(scale) >= 8 else scale
    current_note = random.choice(start_notes)
    
    # Introduce random seed based on time for every unique generation
    random.seed(time.time() + random.random())

    melody = [current_note]

    for _ in range(num_notes - 1):
        if current_note in transitions:
            probs = transitions[current_note]
            notes = list(probs.keys())
            weights = list(probs.values())
            
            # Add a bit of "chaos" factor (5% chance to jump to any scale note)
            if random.random() < 0.05:
                current_note = random.choice(scale)
            else:
                current_note = random.choices(notes, weights=weights, k=1)[0]
        else:
            current_note = random.choice(scale)
        melody.append(current_note)

    return melody


# ============================================================
# Note Styling (Rhythm & Dynamics)
# ============================================================
def stylize_notes(melody, style, tempo, **kwargs):
    """
    Apply rhythm and dynamics based on style.
    Returns list of (note, duration, velocity) tuples.
    """
    styled = []
    
    # Get mood rhythm feel if available
    rhythm_feel = kwargs.get("rhythm_feel", "default")

    if style == "simple":
        if rhythm_feel == "bouncy":
            # Happy: mix of short and medium notes, upbeat
            dur_pool = [0.25, 0.5, 0.5, 0.75, 0.5, 0.25]
            vel_range = (90, 120)
        elif rhythm_feel == "slow":
            # Sad: long sustained notes, softer
            dur_pool = [1.0, 1.5, 1.0, 2.0, 1.5, 1.0]
            vel_range = (55, 85)
        elif rhythm_feel == "flowing":
            # Calm: medium-long, gentle, even
            dur_pool = [0.75, 1.0, 1.0, 1.25, 1.5, 1.0]
            vel_range = (50, 75)
        elif rhythm_feel == "driving":
            # Energetic: fast and punchy
            dur_pool = [0.25, 0.25, 0.5, 0.25, 0.5, 0.25]
            vel_range = (95, 127)
        else:
            dur_pool = [0.5, 0.75, 1.0, 1.0, 0.5, 0.75]
            vel_range = (80, 105)
        
        for i, note in enumerate(melody):
            duration = random.choice(dur_pool)
            velocity = random.randint(*vel_range)
            styled.append((note, duration, velocity))
    else:
        # Complex: varied rhythms and dynamics
        if rhythm_feel == "bouncy":
            durations = [0.125, 0.25, 0.25, 0.375, 0.5, 0.5, 0.75, 1.0]
        elif rhythm_feel == "slow":
            durations = [0.5, 0.75, 1.0, 1.0, 1.5, 2.0, 2.0, 3.0]
        elif rhythm_feel == "flowing":
            durations = [0.5, 0.75, 1.0, 1.0, 1.25, 1.5, 2.0, 1.0]
        elif rhythm_feel == "driving":
            durations = [0.125, 0.25, 0.25, 0.25, 0.375, 0.5, 0.5, 0.75]
        else:
            durations = [0.25, 0.25, 0.5, 0.5, 0.75, 1.0, 1.0, 1.25, 1.5, 2.0]
        
        for i, note in enumerate(melody):
            duration = random.choice(durations)
            # Create dynamic swells
            position_factor = math.sin(i / len(melody) * math.pi)
            base_velocity = random.randint(65, 115)
            velocity = int(base_velocity * (0.8 + 0.2 * position_factor))
            velocity = max(40, min(127, velocity))
            styled.append((note, duration, velocity))

    return styled


# ============================================================
# Generate Notes Based on Mood
# ============================================================
def generate_notes(mood, style, tempo, duration="medium", trained_model_name=None, use_ai=False):
    """
    Generate melody notes based on mood selection.
    Maps mood -> scale -> Markov Chain melody -> styled notes.
    Optionally uses Gemini AI for intelligent composition.
    Returns: (styled_notes, engine_used) tuple
    """
    mood_data = MOOD_SCALES.get(mood, MOOD_SCALES["happy"])
    scale = mood_data["scale"]
    rhythm_feel = mood_data.get("rhythm_feel", "default")

    # Duration-based note count
    duration_map = {
        "short": 16,
        "medium": 32,
        "long": 64,
        "extra_long": 96
    }
    base_notes = duration_map.get(duration, 32)
    num_notes = int(base_notes * 1.5) if style == "complex" else base_notes

    melody = None
    engine_used = "Markov Chain"

    # Try Gemini AI first if enabled
    if use_ai:
        print("\U0001f680 Attempting Gemini AI melody generation...")
        melody = gemini_generate_melody(mood, style, tempo, scale, num_notes)
        if melody:
            print(f"\u2705 Gemini AI generated {len(melody)} notes!")
            engine_used = "Gemini AI (Google)"
        else:
            print("\u26a0\ufe0f Gemini AI failed, falling back to Markov Chain")

    # Fallback: Markov Chain
    if melody is None:
        trained_model = None
        if trained_model_name:
            trained_model, _ = load_trained_model(trained_model_name)
            if trained_model:
                print(f"\U0001f9e0 Using trained model: {trained_model_name}")
        melody = markov_melody(scale, num_notes, trained_model)
        engine_used = "Markov Chain"

    # Apply style (rhythm & dynamics) with mood-specific rhythm feel
    styled_notes = stylize_notes(melody, style, tempo, rhythm_feel=rhythm_feel)

    return styled_notes, engine_used


# ============================================================
# Gemini AI Melody Generator
# ============================================================
def gemini_generate_melody(mood, style, tempo, scale, num_notes):
    """
    Use Google Gemini AI to generate an intelligent melody.
    The AI understands music theory and generates context-appropriate notes.
    Returns a list of MIDI note numbers, or None if API fails.
    """
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    scale_note_names = [note_names[n % 12] + str(n // 12 - 1) for n in scale]

    prompt = f"""You are a professional music composer AI. Generate a melody as a list of MIDI note numbers.

Context:
- Mood: {mood}
- Style: {style}
- Tempo: {tempo} BPM
- Available scale notes (MIDI numbers): {scale}
- Scale note names: {', '.join(scale_note_names)}
- Number of notes needed: {num_notes}

Rules:
1. Output ONLY a JSON array of integers (MIDI note numbers), nothing else.
2. Use ONLY notes from the provided scale: {scale}
3. Create a musically coherent melody that fits the {mood} mood.
4. For "{mood}" mood:
   - {'Use bright, uplifting intervals. Prefer stepwise motion with occasional leaps up.' if mood == 'happy' else ''}
   - {'Use descending motion, minor intervals. Include longer sustained note repetitions.' if mood == 'sad' else ''}
   - {'Use smooth, flowing motion. Avoid large leaps. Keep it gentle and peaceful.' if mood == 'calm' else ''}
   - {'Use strong rhythmic patterns, octave jumps, and driving repetitive motifs.' if mood == 'energetic' else ''}
5. Include musical patterns: repetition, call-and-response, and resolution to the tonic.
6. The melody should have a clear beginning, development, climax, and resolution.

Output exactly {num_notes} MIDI note numbers as a JSON array. Example format: [67, 69, 71, 72, 74]"""

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
                "temperature": 0.8,
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


def _generate_chord_progression(scale, total_duration):
    """Generate a chord progression that spans the total duration."""
    scale_sorted = sorted(set(scale))
    # Common progression patterns (scale degree indices)
    progressions = [
        [0, 3, 4, 3],     # I - IV - V - IV
        [0, 4, 5, 3],     # I - V - vi - IV
        [0, 3, 5, 4],     # I - IV - vi - V
        [0, 5, 3, 4],     # I - vi - IV - V
        [0, 2, 3, 4],     # I - iii - IV - V
    ]
    pattern = random.choice(progressions)
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
    chord_roots = _generate_chord_progression(mood_scale, total_duration)

    # ─── Track 0: Lead Melody ───
    track, channel = 0, 0
    midi.addTrackName(track, 0, "Lead Melody")
    midi.addTempo(track, 0, tempo)
    midi.addProgramChange(track, channel, 0, genre_data["melody"])

    current_time = 0
    for note, duration, velocity in notes:
        note = max(0, min(127, note))
        velocity = max(1, min(127, velocity))
        midi.addNote(track, channel, note, current_time, duration, velocity)
        current_time += duration

    # ─── Track 1: Chord Pad ───
    track, channel = 1, 1
    midi.addTrackName(track, 0, "Chord Pad")
    midi.addTempo(track, 0, tempo)
    midi.addProgramChange(track, channel, 0, genre_data["pad"])

    chord_duration = max(2.0, total_duration / (len(chord_roots) * 2))
    current_time = 0
    chord_idx = 0
    while current_time < total_duration:
        root = chord_roots[chord_idx % len(chord_roots)]
        chord_type = "seventh" if genre == "jazz" else "triad"
        chord_notes = _build_chord(root, mood_scale, chord_type)

        # Place chord in mid register (octave 3-4)
        for cn in chord_notes:
            pad_note = cn
            while pad_note > 72:
                pad_note -= 12
            while pad_note < 48:
                pad_note += 12
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
    bass_patterns = {
        "pop":        [1.0, 1.0, 1.0, 1.0],             # Steady quarter notes
        "rock":       [1.0, 0.5, 0.5, 1.0, 1.0],        # Driving eighths
        "jazz":       [1.5, 0.5, 1.0, 1.0],              # Walking bass
        "electronic": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # Pulsing
    }
    bass_rhythm = bass_patterns.get(genre, bass_patterns["pop"])

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

    arp_speed = {"pop": 0.5, "rock": 0.25, "jazz": 0.375, "electronic": 0.25}
    arp_dur = arp_speed.get(genre, 0.5)

    current_time = 0
    chord_idx = 0
    while current_time < total_duration:
        root = chord_roots[chord_idx % len(chord_roots)]
        chord_notes = _build_chord(root, mood_scale)

        # Place arpeggio in mid-upper register (octave 4-5)
        arp_notes = []
        for cn in chord_notes:
            an = cn
            while an > 84:
                an -= 12
            while an < 60:
                an += 12
            arp_notes.append(max(0, min(127, an)))

        # Arpeggio pattern: up then down
        pattern = arp_notes + list(reversed(arp_notes[1:-1])) if len(arp_notes) > 2 else arp_notes
        if not pattern:
            pattern = [60]

        bar_time = 0
        for i in range(int(chord_duration / arp_dur)):
            if current_time + bar_time >= total_duration:
                break
            note_to_play = pattern[i % len(pattern)]
            vel = random.randint(35, 60)
            actual_dur = min(arp_dur, total_duration - current_time - bar_time)
            if actual_dur > 0:
                midi.addNote(track, channel, note_to_play, current_time + bar_time, actual_dur, vel)
            bar_time += arp_dur

        current_time += chord_duration
        chord_idx += 1

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
            # Rock: Harder hits, ride cymbal, power fills
            if measure_pos == 0:
                midi.addNote(track, channel, KICK, current_time, 0.5, 110)
            elif measure_pos == 1:
                midi.addNote(track, channel, SNARE, current_time, 0.5, 100)
            elif measure_pos == 2:
                midi.addNote(track, channel, KICK, current_time, 0.25, 100)
                midi.addNote(track, channel, KICK, current_time + 0.25, 0.25, 90)
            elif measure_pos == 3:
                midi.addNote(track, channel, SNARE, current_time, 0.5, 105)
            # Ride cymbal
            midi.addNote(track, channel, RIDE, current_time, 0.25, 75)
            midi.addNote(track, channel, RIDE, current_time + 0.5, 0.25, 65)
            # Fill every 4 bars
            if beat_count % 16 == 15:
                midi.addNote(track, channel, TOM_HIGH, current_time, 0.25, 80)
                midi.addNote(track, channel, TOM_LOW, current_time + 0.25, 0.25, 80)
                midi.addNote(track, channel, CRASH, current_time + 0.5, 0.5, 90)

        elif genre == "jazz":
            # Jazz: Ride pattern, soft kick, brush snare on 2&4
            midi.addNote(track, channel, RIDE, current_time, 0.25, 65)
            if random.random() > 0.3:
                midi.addNote(track, channel, RIDE, current_time + 0.67, 0.25, 50)  # Swing feel
            if measure_pos in (1, 3):
                midi.addNote(track, channel, SNARE, current_time, 0.5, 55)  # Brush
            if measure_pos == 0:
                midi.addNote(track, channel, KICK, current_time, 0.5, 60)
            # Random kick comping
            if random.random() > 0.7:
                midi.addNote(track, channel, KICK, current_time + 0.5, 0.25, 50)

        elif genre == "electronic":
            # Electronic: 4-on-floor kick, off-beat hi-hats, claps
            midi.addNote(track, channel, KICK, current_time, 0.5, 110)
            if measure_pos in (1, 3):
                midi.addNote(track, channel, CLAP, current_time, 0.5, 90)
            # Off-beat hi-hats (the "tss-tss" between kicks)
            midi.addNote(track, channel, HH_OPEN, current_time + 0.5, 0.25, 70)
            # Extra 16th note hi-hats for energy
            if random.random() > 0.5:
                midi.addNote(track, channel, HH_CLOSED, current_time + 0.25, 0.12, 45)
                midi.addNote(track, channel, HH_CLOSED, current_time + 0.75, 0.12, 40)
            # Build-up crash every 8 bars
            if beat_count % 32 == 0:
                midi.addNote(track, channel, CRASH, current_time, 1.0, 100)

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

    notes, engine_used = generate_notes(mood, style, tempo, duration, model_name, use_ai=use_ai)
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
    """
    Generate a music track based on user inputs.
    Process: Note Generation → MIDI Creation → Audio Rendering
    """
    # Validate inputs
    mood = mood.lower()
    genre = genre.lower()
    style = style.lower()
    tempo = max(60, min(200, tempo))
    model_name = trained_model if trained_model != "none" else None
    ai_enabled = use_ai == "on"

    # Generate unique filename with timestamp
    timestamp = int(time.time())
    ai_tag = "gemini" if ai_enabled else "markov"
    base_name = f"{mood}_{genre}_{tempo}bpm_{style}_{timestamp}"
    midi_path = OUTPUT_FOLDER / f"{base_name}.mid"
    wav_path = OUTPUT_FOLDER / f"{base_name}.wav"

    # Step 1: Generate notes (Gemini AI or Markov Chain)
    notes, engine_used = generate_notes(mood, style, tempo, duration, model_name, use_ai=ai_enabled)

    # Get mood scale for harmonic layers (pads, bass, arpeggio)
    mood_data = MOOD_SCALES.get(mood, MOOD_SCALES["happy"])
    scale = mood_data["scale"]

    # Step 2: Create multi-track MIDI with full BGM layers (passing scale for chords)
    create_midi(notes, genre, tempo, midi_path, mood_scale=scale)

    # Step 3: Convert MIDI to WAV
    convert_midi_to_wav(midi_path, wav_path, genre)

    # Step 4: Build Markov visualization data
    mood_data = MOOD_SCALES.get(mood, MOOD_SCALES["happy"])
    scale = mood_data["scale"]
    transitions = build_markov_transitions(scale)
    # Simplify for JSON: note name mapping
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    markov_viz = []
    for src, targets in transitions.items():
        src_name = note_names[src % 12] + str(src // 12 - 1)
        for tgt, prob in sorted(targets.items(), key=lambda x: -x[1])[:4]:
            tgt_name = note_names[tgt % 12] + str(tgt // 12 - 1)
            markov_viz.append({
                "from": src_name,
                "to": tgt_name,
                "prob": round(prob * 100, 1)
            })

    # Prepare result data
    wav_filename = wav_path.name
    midi_filename = midi_path.name
    genre_display = GENRE_INSTRUMENTS.get(genre, {}).get("name", genre.capitalize())
    mood_display = mood.capitalize()
    scale_info = MOOD_SCALES.get(mood, {}).get("description", "Custom scale")

    # Step 5: Save to SQL Backend (Spring Boot)
    try:
        http_requests.post("http://127.0.0.1:8081/api/tracks/save", json={
            "filename": wav_filename,
            "mood": mood,
            "genre": genre,
            "tempo": tempo,
            "style": style,
            "aiEngine": engine_used,
            "fileSizeKb": round(wav_path.stat().st_size / 1024, 1)
        }, timeout=3)
        print(f"📁 Track metadata saved to Spring Boot SQL backend")
    except Exception as e:
        print(f"⚠️ Warning: Could not save to SQL backend: {e}")

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
                        "mood": t['mood'].capitalize(),
                        "genre": t['genre'].capitalize(),
                        "tempo": t['tempo'],
                        "style": t['style'].capitalize(),
                        "size_kb": t['fileSizeKb'],
                        "created": t['createdAt'].replace('T', ' ').split('.')[0] if t.get('createdAt') else "Recent",
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
