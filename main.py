from audio.audio_signal import AudioSignal

if __name__ == "__main__":
    path = "data/video.mp4"
    audio = AudioSignal(path)
    audio.detect_hype_moments(threshold=0.6, method='or')
    audio.print_timestamps()
    audio.plot_feature_timeline()
    audio.plot_score_histogram()