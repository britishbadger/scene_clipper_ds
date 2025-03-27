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
import matplotlib.pyplot as plt
import librosa
import numpy as np
import os
import matplotlib.ticker as ticker
from collections import defaultdict
from tabulate import tabulate

def print_event_summary(events):
    """Prints a table summarizing the detected event clusters."""
    if not events:
        print("No significant events detected.")
        return

    table_data = []
    for i, (start, end) in enumerate(events):
        duration = end - start
        table_data.append([i + 1, f"{int(start // 60):02d}:{int(start % 60):02d}", f"{int(end // 60):02d}:{int(end % 60):02d}", f"{duration:.2f} seconds"])

    headers = ["#", "Start (MM:SS)", "End (MM:SS)", "Duration"]
    print("\nDetected Event Clusters:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))



def format_time_axis(seconds, pos=None):
    """Formats seconds to MM:SS for the x-axis."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def visualize_events(video_path, event_times, output_dir):
    """Visualizes the detected events on the audio waveform with 5-minute MM:SS axis ticks and correct limits."""
    y, sr = librosa.load(video_path, sr=None)  # Load with original sample rate
    duration = librosa.get_duration(y=y, sr=sr)
    times = np.linspace(0, duration, num=len(y))

    plt.figure(figsize=(12, 4))
    plt.plot(times, y)  # Use basic plt.plot

    for start, end in event_times:
        plt.axvspan(start, end, color='red', alpha=0.2)

    plt.xlabel("Time (MM:SS)")
    plt.ylabel("Amplitude")
    plt.title("Detected Events")
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_time_axis))

    # Set x-axis limits
    plt.xlim(0, duration)

    # Set x-axis ticks every 5 minutes (300 seconds) up to the duration
    tick_interval = 300
    xticks = np.arange(0, duration + 1, tick_interval)  # Include the end
    plt.xticks(xticks)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "events_visualization.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved event visualization to {output_path}")


def get_video_duration(video_path):
    """Gets the duration of the video using ffmpeg."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
             video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8"
        )
        if result.returncode == 0:
            duration = float(result.stdout.strip())
            return duration
        else:
            print(f"Error getting video duration: {result.stderr}")
            return None
    except FileNotFoundError:
        print("Error: ffprobe not found. Make sure ffmpeg is installed.")
        return None

def extract_audio(video_path):
    """Extract mono audio from video using ffmpeg"""
    audio = AudioSegment.from_file(video_path).set_channels(1).set_frame_rate(44100)
    return audio

def analyze_audio_old(audio):
    """Analyze audio using YAMNet for event detection"""
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    class_map_path = model.class_map_path().numpy().decode('utf-8')
    if not os.path.exists(class_map_path):
        raise FileNotFoundError(f"YAMNet class map not found at {class_map_path}")
    class_names = []
    with open(class_map_path, 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        for row in csv_reader:
            if len(row) >= 3:
                class_names.append(row[2])
    print(f"Loaded {len(class_names)} YAMNet classes. First 5: {class_names[:5]}")
    samples = np.array(audio.get_array_of_samples())
    samples = librosa.resample(samples.astype(float), orig_sr=audio.frame_rate, target_sr=1600)

def analyze_audio(audio):
    """Analyze audio using YAMNet for event detection and class tracking"""
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    class_map_path = model.class_map_path().numpy().decode('utf-8')
    if not os.path.exists(class_map_path):
        raise FileNotFoundError(f"YAMNet class map not found at {class_map_path}")
    class_names = []
    with open(class_map_path, 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        for row in csv_reader:
            if len(row) >= 3:
                class_names.append(row[2])
    print(f"Loaded {len(class_names)} YAMNet classes. First 5: {class_names[:5]}")
    samples = np.array(audio.get_array_of_samples())
    samples = librosa.resample(samples.astype(float), orig_sr=audio.frame_rate, target_sr=16000)
    scores, embeddings, spectrogram = model(samples)
    scores = scores.numpy()
    target_classes = ['Cheering', 'Clapping','Crowd', 'Yell','Children shouting', 'Chatter','Music', 'Applause', 'Siren','Shout','Buzzer','Foghorn', 'Whistle']
    valid_classes = [cls for cls in target_classes if cls in class_names]
    class_indices = [class_names.index(c) for c in valid_classes]
    print(f"Monitoring for sound classes: {[class_names[i] for i in class_indices]}")
    event_mask = np.zeros(scores.shape[0], dtype=bool)
    detected_classes_per_frame = defaultdict(list)
    for i, score_row in enumerate(scores):
        top_index = np.argmax(score_row)
        if top_index in class_indices:
            event_mask[i] = True
            detected_classes_per_frame[i * 0.96].append(class_names[top_index])

    event_indices = np.where(event_mask)[0]
    event_times = event_indices * 0.96
    analyze_audio.class_names = class_names # Make class_names accessible
    return event_times, scores, detected_classes_per_frame


def detect_events(event_times, detected_classes, min_duration=3, merge_window=3.5):
    """Cluster events, filter short/noisy detections, and track classes."""
    events = []
    current_start = None
    current_end = None
    current_classes = set()

    for t in sorted(event_times):
        frame_index = int(round(t / 0.96))
        detected_class_at_t = [cls for time, classes in detected_classes.items() if abs(time - t) < 0.5 for cls in classes] # Find classes near this time

        if current_end is None or t > current_end + merge_window:
            if current_start is not None:
                events.append(((current_start, current_end), list(current_classes)))
            current_start = t
            current_end = t
            current_classes = set(detected_class_at_t)
        else:
            current_end = max(current_end, t)
            current_classes.update(detected_class_at_t)

    if current_start is not None:
        events.append(((current_start, current_end), list(current_classes)))

    # Filter short events and add buffer
    return [
        ((max(0, start - 12), end + 5), classes)
        for (start, end), classes in events
        if (end - start) >= min_duration
    ]

def print_event_summary(events_with_classes):
    """Prints a table summarizing the detected event clusters with included classes."""
    if not events_with_classes:
        print("No significant events detected.")
        return

    table_data = []
    for i, (((start, end), classes)) in enumerate(events_with_classes):
        duration = end - start
        class_str = ", ".join(sorted(list(set(classes)))) # Get unique classes and sort
        table_data.append([i + 1, f"{int(start // 60):02d}:{int(start % 60):02d}", f"{int(end // 60):02d}:{int(end % 60):02d}", f"{duration:.2f} seconds", class_str])

    headers = ["#", "Start (MM:SS)", "End (MM:SS)", "Duration", "Detected Classes"]
    print("\nDetected Event Clusters:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def print_all_detections(event_times, scores, class_names):
    """Prints all individual sound event detections with their timestamps and scores."""
    print("\nAll Individual Detections (Before Clustering and Filtering):")
    table_data = []
    for i, time in enumerate(event_times):
        frame_index = int(round(time / 0.96))
        if frame_index < scores.shape[0]:
            score_vector = scores[frame_index]
            top_index = np.argmax(score_vector)
            class_name = class_names[top_index]
            score = score_vector[top_index]
            table_data.append([f"{int(time // 60):02d}:{int(time % 60):02d}", class_name, f"{score:.4f}"])
        else:
            print(f"Warning: Time {time:.2f} exceeds score array length.")

    headers = ["Time (MM:SS)", "Detected Class", "Score"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def format_time_filename(seconds):
    """Converts seconds to 'MMmSSs' format for filenames."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m{secs}s"

def export_clip(video_path, output_dir, start, end, video_duration):
    """Export video clip with duration limits."""
    os.makedirs(output_dir, exist_ok=True)
    start = max(0, start)  # Ensure start is not negative
    end = min(video_duration, end)  # Ensure end does not exceed video duration

    start_str = format_time_filename(start)
    end_str = format_time_filename(end)
    output_path = os.path.join(output_dir, f"highlight_{start_str}-{end_str}.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", video_path,
        "-c", "copy",
        output_path
    ]
    if end > start:  # Only export if the end time is after the start time
        subprocess.run(cmd, check=True)
    else:
        print(f"Skipping export: end time ({end_str}) is not after start time ({start_str}).")

def main():
    parser = argparse.ArgumentParser(description="Hockey Highlight Extractor")
    parser.add_argument("--input", required=True, help="Input video file")
    parser.add_argument("--output_dir", default="./highlights", help="Output directory")
    parser.add_argument("--visualize_only", action="store_true", help="Only visualize events, do not export clips")
    args = parser.parse_args()

    print("Extracting audio...")
    audio = extract_audio(args.input)

    print("Analyzing audio...")
    event_times, scores, detected_classes = analyze_audio(audio)

    print("Detecting events...")
    events_with_classes = detect_events(event_times, detected_classes)
    events = [event_info[0] for event_info in events_with_classes] # Extract just the times for visualization

    # Visualize events
    visualize_events(args.input, events, args.output_dir)

    # Print all individual detections
    #print_all_detections(event_times, scores, analyze_audio.class_names) # Access class_names

    # Print event summary table with classes
    print_event_summary(events_with_classes)

    video_duration = get_video_duration(args.input)

    if video_duration is not None:
        if not args.visualize_only:
            print(f"Found {len(events_with_classes)} events. Exporting clips (max duration: {format_time_axis(video_duration)})...")
            for (start, end), classes in events_with_classes:
                export_clip(args.input, args.output_dir, start, end, video_duration)
        else:
            print("Skipping clip export as --visualize_only was specified.")
    else:
        print("Could not retrieve video duration. Skipping clip export.")

if __name__ == "__main__":
    main()
