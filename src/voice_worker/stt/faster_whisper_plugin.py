import asyncio
import logging

from faster_whisper import WhisperModel
from livekit.agents import stt, utils
from livekit.agents.stt import SpeechEvent, SpeechEventType

logger = logging.getLogger(__name__)


class FasterWhisperSTT(stt.STT):
    def __init__(self, *, model_size: str = "distil-large-v3", device: str = "cuda"):
        super().__init__(streaming_supported=True, interim_results_supported=True)
        self._model = WhisperModel(model_size, device=device, compute_type="float16")

    async def stream(self) -> "FasterWhisperStream":
        return FasterWhisperStream(self._model)


class FasterWhisperStream(stt.SpeechStream):
    def __init__(self, model: WhisperModel):
        super().__init__()
        self._model = model
        self._queue = utils.aio.Chan[stt.SpeechEvent]()
        self._audio_buffer = bytearray()

    def push_frame(self, frame: stt.AudioFrame):
        self._audio_buffer.extend(frame.data.tobytes())
        # 400ms * 16kHz * 2 bytes = 12800 bytes
        if len(self._audio_buffer) >= 12800:
            asyncio.create_task(self._process_chunk_async(final=False))

    def end_input(self):
        asyncio.create_task(self._process_chunk_async(final=True))
        self._queue.close()

    async def _process_chunk_async(self, final: bool = False):
        if not self._audio_buffer:
            return

        data = bytes(self._audio_buffer)
        try:
            segments, info = await asyncio.to_thread(
                self._model.transcribe,
                data,
                beam_size=1,
                vad_filter=True,
                language="ru",
            )

            text = "".join([segment.text for segment in segments])
            if text.strip():
                event_type = SpeechEventType.FINAL_TRANSCRIPT if final else SpeechEventType.INTERIM_TRANSCRIPT
                self._queue.send_nowait(SpeechEvent(
                    type=event_type,
                    alternatives=[stt.SpeechData(text=text.strip(), language="ru", confidence=0.95)],
                ))

            if final:
                self._audio_buffer.clear()
            else:
                overlap_bytes = 3200 # 100ms overlap
                self._audio_buffer = self._audio_buffer[-overlap_bytes:]
        except Exception as e:
            logger.error(f"Ошибка транскрибации: {e}")

    async def aclose(self) -> None:
        self._queue.close()
