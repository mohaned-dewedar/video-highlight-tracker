# audio/audio_signal.py
from audio.audio_utils import (
    compute_pitch,
    compute_spectral_flatness,
    compute_onset_density,
    compute_energy_entropy
)
from pathlib import Path
import librosa
import numpy as np
import matplotlib.pyplot as plt

class AudioSignal:
    def __init__(self, file_path: str, sr: int = 16000):
        self.file_path = Path(file_path)
        self.sr = sr
        self.audio, self.sr = librosa.load(file_path, sr=sr)
        self.frame_length = int(self.sr * 1.0)
        self.hop_length = self.frame_length
        self.hype_timestamps = []
    def match_audio_template(self, template_path: str):
        """
        Matches a known audio template (e.g., kill sound, voice line) against the stream audio.

        Args:
            template_path (str): Path to the .wav file to match.

        Returns:
            list[float]: List of timestamps (in seconds) where the template likely appears.
        """
        raise NotImplementedError("Audio template matching not implemented yet.")


    def compute_volume(self):
        """
        Computes the normalized RMS volume of the audio signal.
        
        Returns:
            np.ndarray: Normalized RMS volume values.
        """
        rms = librosa.feature.rms(y=self.audio, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        return (rms - rms.min()) / (rms.max() - rms.min())

    def compute_pitch(self):
        """
        Computes the pitch of the audio signal.
        
        Returns:
            np.ndarray: Normalized pitch values.
        """
        return compute_pitch(self.audio, self.sr, self.frame_length, self.hop_length)

    def detect_hype_moments(self, threshold=0.6, min_gap_sec=10.0, method='or'):
        volume = self.compute_volume()
        pitch = self.compute_pitch()
        flatness = self.compute_spectral_complexity()
        onsets = self.compute_onset_density()
        entropy = self.compute_energy_entropy()

        if method == 'or':
            hype_score = np.maximum(volume, pitch)
        elif method == 'avg':
            hype_score = 0.2*volume + 0.2*pitch + 0.2*flatness + 0.2*onsets + 0.2*entropy
        elif method == 'euclidean':
            hype_score = np.sqrt(volume**2 + pitch**2 + flatness**2 + onsets**2 + entropy**2)
        else:
            raise ValueError("Unknown method")

        # Detect spikes
        high_energy_frames = np.where(hype_score > threshold)[0]
        times = high_energy_frames * (self.hop_length / self.sr)

        # Merge timestamps
        merged = []
        frame_indices = []
        last = -min_gap_sec
        for idx, t in zip(high_energy_frames, times):
            if t - last > min_gap_sec:
                merged.append(t)
                frame_indices.append(idx)
                last = t

        # Cache for print
        self.hype_timestamps = merged
        self._hype_frame_indices = frame_indices
        self._cached_features = {
            "volume": volume,
            "pitch": pitch,
            "flatness": flatness,
            "onsets": onsets,
            "entropy": entropy,
            "score": hype_score
        }

        return merged


    def compute_spectral_complexity(self):
        return compute_spectral_flatness(self.audio, self.sr, self.frame_length, self.hop_length)

    def compute_onset_density(self):
        return compute_onset_density(self.audio, self.sr, self.hop_length)

    def compute_energy_entropy(self):
        return compute_energy_entropy(self.audio, self.sr, self.frame_length, self.hop_length)

    def print_timestamps(self):
        print("Detected Hype Moments (Audio):")
        for t, idx in zip(self.hype_timestamps, self._hype_frame_indices):
            feature_vals = {name: self._cached_features[name][idx] for name in self._cached_features}
            timestamp_str = f"{int(t // 60):02d}:{int(t % 60):02d}"
            breakdown = ", ".join([f"{k.capitalize()}: {v:.2f}" for k, v in feature_vals.items()])
            print(f"{timestamp_str} - {breakdown}")



    def plot_hype_timeline(self):
        if not hasattr(self, '_last_hype_score'):
            raise RuntimeError("Run detect_hype_moments first.")

        plt.figure(figsize=(12, 4))
        plt.plot(self._last_volume, label='Volume')
        plt.plot(self._last_pitch, label='Pitch')
        plt.plot(self._last_hype_score, label='Hype Score', linestyle='--', alpha=0.8)
        plt.axhline(0.6, color='red', linestyle=':', label='Threshold')
        plt.title("Audio Features Over Time")
        plt.xlabel("Frame Index (1s steps)")
        plt.ylabel("Normalized Score")
        plt.legend()
        plt.tight_layout()
        plt.savefig("data/audio_hype_timeline.png")
    def plot_feature_timeline(self):
        volume = self.compute_volume()
        pitch = self.compute_pitch()
        flatness = self.compute_spectral_complexity()
        onsets = self.compute_onset_density()
        entropy = self.compute_energy_entropy()

        plt.figure(figsize=(14, 8))

        plt.subplot(5, 1, 1)
        plt.plot(volume, label="Volume")
        plt.legend()

        plt.subplot(5, 1, 2)
        plt.plot(pitch, label="Pitch")
        plt.legend()

        plt.subplot(5, 1, 3)
        plt.plot(flatness, label="Spectral Flatness")
        plt.legend()

        plt.subplot(5, 1, 4)
        plt.plot(onsets, label="Onset Density")
        plt.legend()

        plt.subplot(5, 1, 5)
        plt.plot(entropy, label="Energy Entropy")
        plt.legend()

        plt.tight_layout()
        plt.savefig("data/audio_feature_timeline.png")

    

