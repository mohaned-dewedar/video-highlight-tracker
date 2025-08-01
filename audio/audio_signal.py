# audio/audio_signal.py

import librosa
import numpy as np
from pathlib import Path

class AudioSignal:
    def __init__(self, file_path: str, sr: int = 16000):
        self.file_path = Path(file_path)
        self.sr = sr
        self.audio, self.sr = librosa.load(file_path, sr=sr)
        self.hype_timestamps = []

    def compute_volume(self, frame_sec: float = 1.0):
        frame_length = int(self.sr * frame_sec)
        hop_length = frame_length
        rms = librosa.feature.rms(y=self.audio, frame_length=frame_length, hop_length=hop_length)[0]
        norm_rms = (rms - np.min(rms)) / (np.max(rms) - np.min(rms))
        return norm_rms, hop_length

    def detect_hype_moments(self, threshold: float = 0.7, min_gap_sec: float = 10.0):
        norm_rms, hop_length = self.compute_volume()
        high_energy_frames = np.where(norm_rms > threshold)[0]
        times = high_energy_frames * (hop_length / self.sr)

        # Merge close timestamps
        merged = []
        last = -min_gap_sec
        for t in times:
            if t - last > min_gap_sec:
                merged.append(t)
            last = t

        self.hype_timestamps = merged
        return merged

    def print_timestamps(self):
        print("Detected Hype Moments (Audio Only):")
        for t in self.hype_timestamps:
            print(f"{int(t // 60):02d}:{int(t % 60):02d}")
