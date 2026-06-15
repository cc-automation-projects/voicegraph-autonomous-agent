import asyncio
import logging

import torch
from livekit.agents import tts, utils

logger = logging.getLogger(__name__)


class SileroTTS(tts.TTS):
    def __init__(self, *, device: str = "cpu"):
        super().__init__(streaming_supported=True, sample_rate=24000, num_channels=1)
        self._device = device
        self._model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models", model="silero_tts",
            language="ru", speaker="v3_ru"
        )
        self._model.to(device)
        self._executor = utils.aio.get_or_create_global_executor()

    async def synthesize_async(self, text: str) -> tts.SynthesizedAudio:
        loop = asyncio.get_running_loop()
        audio = await loop.run_in_executor(self._executor, self._model.apply_tts, text, "xenia", 24000)
        audio_tensor = torch.tensor(audio).unsqueeze(0)
        return tts.SynthesizedAudio(
            text=text,
            data=audio_tensor.numpy().tobytes(),
            sample_rate=24000,
        )

    def synthesize(self, text: str) -> tts.SynthesizedAudio:
        audio = self._model.apply_tts(text=text, speaker="xenia", sample_rate=24000)
        audio_tensor = torch.tensor(audio).unsqueeze(0)
        return tts.SynthesizedAudio(
            text=text,
            data=audio_tensor.numpy().tobytes(),
            sample_rate=24000,
        )

    async def stream(self) -> "SileroTTSStream":
        return SileroTTSStream(self)


class SileroTTSStream(tts.TTSStream):
    def __init__(self, tts_engine: SileroTTS):
        super().__init__()
        self._tts = tts_engine
        self._queue = utils.aio.Chan[tts.SynthesizedAudio]()
        self._buffer = ""
        self._synthesis_task = None

    def push_text(self, text: str):
        self._buffer += text
        if any(self._buffer.strip().endswith(p) for p in [".", "!", "?", "…"]):
            self._flush_buffer()

    def _flush_buffer(self):
        if not self._buffer.strip():
            return
        text_chunk = self._buffer
        self._buffer = ""
        if self._synthesis_task is None or self._synthesis_task.done():
            self._synthesis_task = asyncio.create_task(self._synthesize_and_enqueue(text_chunk))

    async def _synthesize_and_enqueue(self, text: str):
        try:
            audio = await self._tts.synthesize_async(text)
            self._queue.send_nowait(audio)
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")

    def flush(self):
        if self._buffer:
            self._flush_buffer()
        if self._synthesis_task:
            asyncio.create_task(self._synthesis_task)
        self._queue.close()

    async def aclose(self) -> None:
        self.flush()
