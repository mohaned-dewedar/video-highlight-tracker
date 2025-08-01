# audio/audio_utils.py

import librosa
import numpy as np

def compute_pitch(y, sr, frame_length, hop_length):
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)
    pitch_vals = []

    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        pitch_vals.append(pitch if pitch > 0 else 0)

    # Normalize to [0, 1]
    pitch_vals = np.array(pitch_vals)
    if pitch_vals.max() > 0:
        pitch_vals = (pitch_vals - pitch_vals.min()) / (pitch_vals.max() - pitch_vals.min())
    return pitch_vals
import librosa
import numpy as np

def compute_spectral_flatness(y, sr, frame_length, hop_length):
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
    return (flatness - flatness.min()) / (flatness.max() - flatness.min())

def compute_onset_density(y, sr, hop_length):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_env = np.clip(onset_env, 0, None)
    return (onset_env - onset_env.min()) / (onset_env.max() - onset_env.min())

def compute_energy_entropy(y, sr, frame_length, hop_length):
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    window_size = 5  # number of frames to consider for local entropy
    entropy = []

    for i in range(len(rms)):
        window = rms[max(0, i-window_size):i+1]
        prob = window / (np.sum(window) + 1e-6)
        ent = -np.sum(prob * np.log2(prob + 1e-6))
        entropy.append(ent)

    entropy = np.array(entropy)
    return (entropy - entropy.min()) / (entropy.max() - entropy.min())
