from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from src.voice_worker.emotion.detector import EmotionDetector, EmotionState


class TestEmotionDetector:
    def test_empty_audio(self):
        detector = EmotionDetector()
        result = detector.extract_and_classify(np.array([]), 16000)
        assert result == EmotionState.UNKNOWN

    def test_normal_audio(self):
        detector = EmotionDetector()
        audio = np.random.randn(16000).astype(np.float32) * 0.01
        result = detector.extract_and_classify(audio, 16000)
        assert result in EmotionState

    def test_high_energy_audio(self):
        detector = EmotionDetector()
        audio = np.random.randn(16000).astype(np.float32) * 0.2
        result = detector.extract_and_classify(audio, 16000)
        assert result in EmotionState

    def test_soundfile_roundtrip(self):
        detector = EmotionDetector()
        audio = np.random.randn(16000).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.wav"
            sf.write(str(path), audio, 16000)
            loaded, sr = sf.read(str(path))
            assert sr == 16000
            assert loaded.shape == audio.shape
            result = detector.extract_and_classify(loaded.astype(np.float32), sr)
            assert result in EmotionState
