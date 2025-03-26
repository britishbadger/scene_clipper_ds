✅ Objective
Develop a local Python-based command-line tool that automatically generates highlight clips from ice hockey game videos by detecting goal events and exciting moments using audio-based analysis. The tool will extract short clips (e.g., ~12–15 seconds) around detected events and save them for post-editing.

🧩 Key Features & Workflow
1. Input
Video format: .mp4 (HEVC or H.264 encoded)

Audio embedded in video (typically stereo with crowd, music, commentary)

2. Output
Folder of MP4 highlight clips (e.g., highlight_01.mp4, highlight_02.mp4, …)

Total highlight duration per input segment: ~10 minutes

No post-processing or concatenation; user will stitch later

3. Audio Analysis Logic
Extract audio using ffmpeg (WAV, mono, 44.1kHz)

Split into 1-second chunks

Calculate volume (e.g., RMS or dBFS) for each chunk

Use a dynamic threshold (e.g., 90th percentile of volume) to detect “loud” moments

Track sequences of loud chunks and mark them as potential highlight cues

4. Goal & Highlight Cue Strategy
Based on revised insight from prior failures

Music and goal horns are strong indicators of goals, but occur after the actual goal event.

Therefore, extract highlights from 12 seconds before the detected music/spike to ~5 seconds after.

This requires:

Buffering segments before and after each spike

Possibly ignoring the first 1–2 seconds of spike (where the horn/music begins) if refining further

Avoid duplicate/stoppage music by checking for:

Very short bursts

Events that do not follow crowd spikes

Clustering events that are too close in time

🧠 Additional Considerations
Music can play during stoppages (not goals). To differentiate:

Cross-reference with preceding crowd noise (5–10 seconds before music)

If volume rises before the music, it’s likely a goal

If volume is flat then suddenly music starts, likely a stoppage

Optional future enhancement:

Add basic ML classification using YAMNet or PANNs to confirm “cheer”, “horn”, “music” vs. “speech”

Keep detection fast and local

⚙️ Tools & Libraries
ffmpeg – Audio extraction and video clip trimming

pydub – Audio slicing and dBFS analysis

numpy – Volume analytics and percentile thresholding

Optional: librosa, torchaudio if moving to spectrograms/ML later

💻 Implementation Stack
Component	Technology
Platform	Linux (desktop/server)
Language	Python 3.8+
Audio Analysis	pydub + numpy
Video Clip Extraction	ffmpeg-python or subprocess
GPU/ML (optional)	PyTorch w/ CUDA (NVIDIA 3060 supported)
🧪 Testing Plan
Use a known 20-minute segment of hockey footage with 1–2 clear goals

Run extractor, check whether clips cover:

12 seconds before goal horn/music

5 seconds after

Crowd build-up and commentary

Ensure extracted clips are not triggered by intermission music or stoppages

📁 Deliverables
highlight_extractor.py (CLI script)

requirements.txt with dependencies

README.md with usage instructions:

bash
Copy
Edit
python highlight_extractor.py --input /path/to/video.mp4 --output_dir ./highlights

✨ Phase 2.
ML-powered cheer/music detection (YAMNet or PANNs)

Phase 3
UI frontend Tkinter
