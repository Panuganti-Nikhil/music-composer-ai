"""
Generate a collection of MIDI files for training CoverComposer.
Creates multiple realistic-sized MIDI files in different keys and styles.
Much more effective for training than a single massive file!

Usage: python generate_large_midi.py
Output: training_data/mega_dataset/ (100 MIDI files, ~5-10MB total)
"""

import random
import os
from pathlib import Path
from midiutil import MIDIFile


# Musical scales across different keys
ALL_SCALES = {
    "C_major":     [48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 65, 67, 69, 71, 72],
    "C_minor":     [48, 50, 51, 53, 55, 56, 58, 60, 62, 63, 65, 67, 68, 70, 72],
    "D_major":     [50, 52, 54, 55, 57, 59, 61, 62, 64, 66, 67, 69, 71, 73, 74],
    "D_minor":     [50, 52, 53, 55, 57, 58, 60, 62, 64, 65, 67, 69, 70, 72, 74],
    "E_major":     [52, 54, 56, 57, 59, 61, 63, 64, 66, 68, 69, 71, 73, 75, 76],
    "E_minor":     [52, 54, 55, 57, 59, 60, 62, 64, 66, 67, 69, 71, 72, 74, 76],
    "F_major":     [53, 55, 57, 58, 60, 62, 64, 65, 67, 69, 70, 72, 74, 76, 77],
    "G_major":     [55, 57, 59, 60, 62, 64, 66, 67, 69, 71, 72, 74, 76, 78, 79],
    "G_minor":     [55, 57, 58, 60, 62, 63, 65, 67, 69, 70, 72, 74, 75, 77, 79],
    "A_major":     [57, 59, 61, 62, 64, 66, 68, 69, 71, 73, 74, 76, 78, 80, 81],
    "A_minor":     [57, 59, 60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81],
    "Bb_major":    [58, 60, 62, 63, 65, 67, 69, 70, 72, 74, 75, 77, 79, 81, 82],
    "B_minor":     [59, 61, 62, 64, 66, 67, 69, 71, 73, 74, 76, 78, 79, 81, 83],
    "Eb_major":    [51, 53, 55, 56, 58, 60, 62, 63, 65, 67, 68, 70, 72, 74, 75],
    "C_pentatonic": [48, 50, 52, 55, 57, 60, 62, 64, 67, 69, 72, 74, 76, 79, 81],
    "A_blues":     [57, 60, 62, 63, 64, 67, 69, 72, 74, 75, 76, 79, 81, 84, 86],
}

GENRE_PATTERNS = {
    "pop": {
        "tempos": [100, 110, 120, 128, 130],
        "durations": [0.25, 0.5, 0.5, 0.75, 1.0, 1.0],
        "instruments": [0, 4, 80],      # Piano, EP, Square
        "notes_range": (2000, 5000),
    },
    "rock": {
        "tempos": [120, 130, 140, 150, 160],
        "durations": [0.25, 0.25, 0.5, 0.5, 1.0],
        "instruments": [29, 30, 25],     # Overdriven, Distortion, Steel Guitar
        "notes_range": (2000, 5000),
    },
    "jazz": {
        "tempos": [80, 100, 120, 140],
        "durations": [0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0],
        "instruments": [0, 65, 32],      # Piano, Alto Sax, Acoustic Bass
        "notes_range": (3000, 6000),
    },
    "classical": {
        "tempos": [60, 72, 80, 100, 120],
        "durations": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0],
        "instruments": [40, 42, 48, 68],  # Violin, Cello, Strings, Oboe
        "notes_range": (4000, 8000),
    },
    "electronic": {
        "tempos": [120, 128, 135, 140, 150],
        "durations": [0.125, 0.25, 0.25, 0.5, 0.5],
        "instruments": [80, 81, 38, 88],  # Square, Sawtooth, SynthBass, Pad
        "notes_range": (3000, 6000),
    },
    "rnb": {
        "tempos": [70, 80, 90, 100],
        "durations": [0.25, 0.5, 0.5, 0.75, 1.0, 1.5],
        "instruments": [4, 0, 88],        # EP, Piano, Pad
        "notes_range": (2000, 4000),
    },
}


def generate_song_midi(output_path, scale, genre_config, tempo, song_name="song"):
    """Generate a single realistic MIDI song file with 3 tracks."""
    midi = MIDIFile(3)
    num_notes = random.randint(*genre_config["notes_range"])
    
    # Track 0: Melody
    midi.addTrackName(0, 0, "Melody")
    midi.addTempo(0, 0, tempo)
    midi.addProgramChange(0, 0, 0, random.choice(genre_config["instruments"]))
    
    current_time = 0.0
    current_note = random.choice(scale)
    
    for _ in range(num_notes):
        # Realistic Markov-like movement
        r = random.random()
        idx = scale.index(current_note) if current_note in scale else len(scale) // 2
        
        if r < 0.35:     # Stepwise
            idx = max(0, min(len(scale)-1, idx + random.choice([-1, 1])))
        elif r < 0.60:   # Small skip
            idx = max(0, min(len(scale)-1, idx + random.choice([-2, -3, 2, 3])))
        elif r < 0.80:   # Repeat
            pass
        elif r < 0.92:   # Larger leap
            idx = max(0, min(len(scale)-1, idx + random.choice([-4, -5, 4, 5])))
        else:            # Random
            idx = random.randint(0, len(scale)-1)
        
        current_note = scale[idx]
        duration = random.choice(genre_config["durations"])
        velocity = random.randint(55, 120)
        
        midi.addNote(0, 0, current_note, current_time, duration, velocity)
        current_time += duration
    
    # Track 1: Bass
    midi.addTrackName(1, 0, "Bass")
    midi.addTempo(1, 0, tempo)
    midi.addProgramChange(1, 1, 0, 33)  # Electric Bass
    
    bass_scale = [max(0, n - 12) for n in scale if n - 12 >= 28]
    if not bass_scale:
        bass_scale = scale[:5]
    
    current_time = 0.0
    bass_note = random.choice(bass_scale)
    
    for _ in range(num_notes // 3):
        idx = bass_scale.index(bass_note) if bass_note in bass_scale else 0
        step = random.choice([-1, 0, 0, 1, 2, -2])
        idx = max(0, min(len(bass_scale)-1, idx + step))
        bass_note = bass_scale[idx]
        
        duration = random.choice([0.5, 1.0, 1.0, 1.5, 2.0])
        velocity = random.randint(60, 100)
        
        midi.addNote(1, 1, bass_note, current_time, duration, velocity)
        current_time += duration
    
    # Track 2: Chords / Pad
    midi.addTrackName(2, 0, "Pad")
    midi.addTempo(2, 0, tempo)
    midi.addProgramChange(2, 2, 0, 88)  # Pad
    
    current_time = 0.0
    for _ in range(num_notes // 8):
        root_idx = random.randint(0, len(scale) - 4)
        chord = [scale[root_idx], scale[min(root_idx+2, len(scale)-1)], 
                 scale[min(root_idx+4, len(scale)-1)]]
        
        duration = random.choice([2.0, 4.0, 4.0, 8.0])
        velocity = random.randint(40, 80)
        
        for note in chord:
            midi.addNote(2, 2, note, current_time, duration, velocity)
        current_time += duration
    
    # Write
    with open(str(output_path), "wb") as f:
        midi.writeFile(f)
    
    return os.path.getsize(output_path)


def main():
    output_dir = Path("training_data") / "mega_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_files = 100
    total_bytes = 0
    total_notes_est = 0
    
    print("=" * 60)
    print("🎵 CoverComposer — Mega Training Dataset Generator")
    print("=" * 60)
    print(f"Generating {total_files} MIDI files...")
    print(f"Output: {output_dir}/")
    print()
    
    scale_names = list(ALL_SCALES.keys())
    genre_names = list(GENRE_PATTERNS.keys())
    
    for i in range(total_files):
        # Pick random scale and genre
        scale_name = random.choice(scale_names)
        genre_name = random.choice(genre_names)
        scale = ALL_SCALES[scale_name]
        genre_config = GENRE_PATTERNS[genre_name]
        tempo = random.choice(genre_config["tempos"])
        
        # Generate filename
        filename = f"{genre_name}_{scale_name}_{tempo}bpm_{i+1:03d}.mid"
        filepath = output_dir / filename
        
        # Generate the MIDI file
        size = generate_song_midi(filepath, scale, genre_config, tempo)
        total_bytes += size
        notes_est = random.randint(*genre_config["notes_range"])
        total_notes_est += notes_est
        
        pct = ((i + 1) / total_files) * 100
        print(f"  [{pct:5.1f}%] {filename} ({size/1024:.1f} KB)", end="\r")
    
    total_mb = total_bytes / (1024 * 1024)
    
    print(f"\n\n{'=' * 60}")
    print(f"✅ DATASET GENERATED SUCCESSFULLY!")
    print(f"{'=' * 60}")
    print(f"  📁 Location:     {output_dir}/")
    print(f"  📄 Files:        {total_files}")
    print(f"  💾 Total size:   {total_mb:.1f} MB ({total_bytes:,} bytes)")
    print(f"  🎵 Est. notes:   ~{total_notes_est:,}")
    print(f"  🧠 Transitions:  ~{total_notes_est:,}")
    print(f"{'=' * 60}")
    print()
    print("🚀 NEXT STEPS:")
    print(f"  1. Go to: http://127.0.0.1:8000/train")
    print(f"  2. Model name: 'mega_trained'")
    print(f"  3. Upload ALL files from: {output_dir}/")
    print(f"  4. Click 'Train Model'")
    print(f"  5. Use the trained model on the Generate page!")
    print()
    print("  💡 OR train from command line:")
    print(f'  python -c "from main import train_from_midi_folder; train_from_midi_folder(r\'{output_dir}\', \'mega_trained\')"')


if __name__ == "__main__":
    main()
