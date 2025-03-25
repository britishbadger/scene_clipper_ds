#!/usr/bin/env python3
import argparse
import os
import numpy as np
from pydub import AudioSegment
from pydub.utils import mediainfo
import tempfile
import subprocess
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import csv

def extract_audio(video_path):
    """Extract mono audio from video using ffmpeg"""
    audio = AudioSegment.from_file(video_path).set_channels(1).set_frame_rate(44100)
    return audio


def analyze_audio(audio):
    """Analyze audio using YAMNet for event detection"""
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

    # Target classes to monitor
    target_classes = ['Cheering', 'Crowd', 'Music', 'Applause', 'Air horn, truck horn', 'Whistle']
        
    valid_classes = [cls for cls in target_classes if cls in class_names]


    class_indices = [class_names.index(c) for c in valid_classes]

    print(f"Monitoring for sound classes: {[class_names[i] for i in class_indices]}")

    # Create a mask for any of our target classes
    event_mask = np.zeros(scores.shape[0], dtype=bool)

    for idx in class_indices:
        event_mask |= (np.argmax(scores, axis=1) == idx)

    event_indices = np.where(event_mask)[0]

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
    event_times, scores = analyze_audio(audio)
    
    print("Detecting events...")
    events = detect_events(event_times)
    
    print(f"Found {len(events)} events. Exporting clips...")
    for start, end in events:
        export_clip(args.input, args.output_dir, start, end)

if __name__ == "__main__":
    main()