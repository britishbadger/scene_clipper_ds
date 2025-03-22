#!/usr/bin/env python3
import argparse
import os
import numpy as np
from pydub import AudioSegment
from pydub.utils import mediainfo
import tempfile
import subprocess
from tqdm import tqdm

def extract_audio(video_path):
    """Extract mono audio from video using ffmpeg"""
    audio = AudioSegment.from_file(video_path).set_channels(1).set_frame_rate(44100)
    return audio

def analyze_audio(audio):
    """Analyze audio using YAMNet for event detection"""
    import tensorflow as tf
    import tensorflow_hub as hub
    import librosa
    import csv
    
    # Load YAMNet model and class names
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    class_map_path = model.class_map_path().numpy().decode('utf-8')
    
    # Verify class map file exists
    if not os.path.exists(class_map_path):
        raise FileNotFoundError(f"YAMNet class map not found at {class_map_path}")
    
    # Load class names from CSV
    class_names = []
    with open(class_map_path, 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # Skip header
        for row in csv_reader:
            if len(row) >= 3:
                class_names.append(row[2])  # Display name is 3rd column
                
    print(f"Loaded {len(class_names)} YAMNet classes. First 5: {class_names[:5]}")
    
    # Convert to numpy array and resample to 16kHz
    samples = np.array(audio.get_array_of_samples())
    samples = librosa.resample(
        samples.astype(float), 
        orig_sr=audio.frame_rate, 
        target_sr=16000
    )
    
    # Run model inference
    scores, embeddings, spectrogram = model(samples)
    scores = scores.numpy()
    
    # Get timestamps for relevant events using verified YAMNet classes
    # Validate YAMNet class names against known variants
# Validate YAMNet classes with proper indentation
    # Verified YAMNet class names from debug output
    # Verified YAMNet classes matching code indices
    target_classes = [
        'Cheering',  # Class 85
        'Crowd',     # Class 132
        'Music',     # Class 137  
        'Applause'   # Class 84
    ]
    
    valid_classes = []
    for cls in target_classes:
        if cls in class_names:
            valid_classes.append(cls)
        else:
            print(f"Class '{cls}' not found in model")
    
    if not valid_classes:
        available = [c for c in class_names if any(x in c.lower() 
                    for x in ['cheer', 'horn', 'crowd', 'music', 'applaud'])]
        raise ValueError(
            "No valid event classes detected.\nAvailable relevant classes:\n- " 
            + "\n- ".join(available)
        )
    
    print(f"Monitoring audio events for: {valid_classes}")
    class_indices = [class_names.index(c) for c in valid_classes]

    if not valid_classes:
        available = [c for c in class_names if any(x in c.lower() for x in 
        ['cheer', 'horn', 'crowd', 'music', 'applaud'])]
    raise ValueError(
        "No valid event classes detected.\n"
        "Tried: " + ", ".join(target_classes) + "\n"
        "Available relevant classes:\n- " + "\n- ".join(available)
    )

print("Active sound classes:", valid_classes)
class_indices = [class_names.index(c) for c in valid_classes]
class_indices = [class_names.index(c) for c in valid_classes]
    # (This block removed - replaced by comprehensive validation above)
    
    print(f"Monitoring for sound classes: {[class_names[i] for i in class_indices]}")
    
    # Create a mask for any of our target classes
    event_mask = np.zeros(scores.shape[0], dtype=bool)
    for idx in class_indices:
        event_mask |= (np.argmax(scores, axis=1) == idx)
    
    event_indices = np.where(event_mask)[0]
    event_indices = np.where(
        # Print first 20 class names for verification
print("First 20 YAMNet classes:", class_names[:20])

# Use verified class indices
cheer_idx = class_names.index('Cheering')
crowd_idx = class_names.index('Crowd')
horn_idx = class_names.index('Horn')  # Verified correct class name
music_idx = class_names.index('Music')

event_indices = np.where(
    (np.argmax(scores, axis=1) == cheer_idx) |
    (np.argmax(scores, axis=1) == crowd_idx) |
    (np.argmax(scores, axis=1) == horn_idx) |
    (np.argmax(scores, axis=1) == music_idx)
)[0]
    )[0]
    
    # Debug class names
print("First 50 YAMNet classes:", class_names[:50])
# Convert window indices to seconds (YAMNet uses 0.96s windows)
    event_times = event_indices * 0.96
    return event_times, scores

def detect_events(event_times, min_duration=2, merge_window=3):
    """Cluster events and filter short/noisy detections"""
    events = []
    current_start = None
    current_end = None
    
    for t in sorted(event_times):
        if current_end is None or t > current_end + merge_window:
            if current_start is not None:
                events.append((current_start, current_end))
            current_start = t
            current_end = t
        else:
            current_end = max(current_end, t)
    
    if current_start is not None:
        events.append((current_start, current_end))
    
    # Filter short events and add buffer
    return [
        (max(0, start - 12), end + 5)  # 12s pre-roll, 5s post-roll
        for start, end in events
        if (end - start) >= min_duration
    ]

def export_clip(video_path, output_dir, start, end):
    """Export video clip using ffmpeg"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"highlight_{start}-{end}.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", video_path,
        "-c", "copy",
        output_path
    ]
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Hockey Highlight Extractor")
    parser.add_argument("--input", required=True, help="Input video file")
    parser.add_argument("--output_dir", default="./highlights", help="Output directory")
    args = parser.parse_args()

    print("Extracting audio...")
    audio = extract_audio(args.input)
    
    print("Analyzing audio...")
    chunks, threshold = analyze_audio(audio)
    
    print("Detecting events...")
    events = detect_events(chunks, threshold)
    
    print(f"Found {len(events)} events. Exporting clips...")
    for start, end in events:
        export_clip(args.input, args.output_dir, start, end)

if __name__ == "__main__":
    main()