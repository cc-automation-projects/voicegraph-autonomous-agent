import asyncio
import logging

import numpy as np
from livekit.agents import stt

from .detector import EmotionDetector, EmotionState

logger = logging.getLogger(__name__)


class EmotionAnalysisPipeline:
    def __init__(self, detector: EmotionDetector, window_duration_sec: float = 1.5):
        self.detector = detector
        self.window_duration_sec = window_duration_sec
        self.sample_rate = 16000
        self.window_bytes = int(window_duration_sec * self.sample_rate * 2)
        self.audio_buffer = bytearray()
        self.current_emotion = EmotionState.CALM
        self._task = None
        self._is_running = False

    async def start(self):
        self._is_running = True
        self._task = asyncio.create_task(self._processing_loop())
        logger.info("Emotion Analysis Pipeline запущен")

    async def stop(self):
        self._is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Emotion Analysis Pipeline остановлен")

    def push_audio_frame(self, frame: stt.AudioFrame):
        if not self._is_running:
            return

        self.audio_buffer.extend(frame.data.tobytes())

        if len(self.audio_buffer) > self.window_bytes:
            self.audio_buffer = self.audio_buffer[-self.window_bytes:]

    async def _processing_loop(self):
        check_interval = 0.5

        while self._is_running:
            await asyncio.sleep(check_interval)

            if len(self.audio_buffer) < self.window_bytes:
                continue

            buffer_copy = bytes(self.audio_buffer)

            try:
                audio_np = np.frombuffer(buffer_copy, dtype=np.int16).astype(np.float32) / 32768.0

                emotion = await self.detector.extract_and_classify_async(audio_np, self.sample_rate)

                if emotion != self.current_emotion:
                    logger.info(f"Изменение эмоции: {self.current_emotion.value} -> {emotion.value}")
                    self.current_emotion = emotion

            except Exception as e:
                logger.error(f"Ошибка в цикле обработки эмоций: {e}")

    def get_current_emotion(self) -> EmotionState:
        return self.current_emotion
