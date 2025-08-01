import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load audio
y, sr = librosa.load("output.wav", sr=16000)

# Frame settings
frame_length = int(sr * 1.0)  # 1 second frames
hop_length = frame_length

# Compute RMS (volume)
rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

# Normalize RMS
rms_norm = (rms - rms.min()) / (rms.max() - rms.min())

# Thresholding
threshold = 0.4  # You can tune this
hype_frames = np.where(rms_norm > threshold)[0]

# Convert frame indices to time
hype_times = hype_frames * (hop_length / sr)

# Print timestamps
print("Potential hype moments:")
for t in hype_times:
    print(f"{int(t//60):02d}:{int(t%60):02d}")



plt.figure(figsize=(12, 4))
plt.plot(rms_norm, label="Normalized Volume")
plt.axhline(threshold, color='r', linestyle='--', label="Threshold")
plt.title("Audio Volume Over Time")
plt.xlabel("Seconds")
plt.legend()
plt.savefig("volume_plot.png")

from itertools import groupby

merged_hype = []
gap = 10  # seconds
last = -gap

for t in hype_times:
    if t - last > gap:
        merged_hype.append(t)
    last = t

print(merged_hype)