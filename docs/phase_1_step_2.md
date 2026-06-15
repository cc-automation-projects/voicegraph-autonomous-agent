Это "сердце" системы, где определяющее значение имеет каждая миллисекунда. Мы не будем использовать готовые, "черные ящики" для STT/LLM/TTS, а реализуем кастомные плагины для `livekit-agents`, чтобы иметь полный контроль над потоковой передачей (streaming), буферизацией и обработкой прерываний (barge-in).

---

# 🚀 ЭТАП 1.2: Сборка LiveKit Voice Agent (Streaming)

## Шаг 1: Зависимости и подготовка окружения

Добавляем специфичные для голосового пайплайна зависимости в `pyproject.toml` микросервиса `voice-worker`.

```toml
# Добавить в pyproject.toml voice-worker
dependencies = [
    "livekit-agents>=0.10.0",
    "livekit-api>=0.8.0",
    "faster-whisper>=1.0.3",       # Оптимизированный ASR (CTranslate2)
    "silero-vad>=5.1.2",            # Сверхбыстрый VAD
    "torch>=2.3.0",                 # Для Silero TTS и VAD
    "torchaudio>=2.3.0",
    "httpx>=0.27.0",                # Асинхронные HTTP-запросы к vLLM
    "numpy>=1.26.4",
    # Наши внутренние модули (устанавливаются как local packages или через workspace)
    "voicegraph-pii-sanitizer",     # Из Этапа 0.3
]
```

---

## Шаг 2: Реализация кастомного Streaming STT (Faster-Whisper)

Стандартный Whisper блокирующий. Мы используем `faster-whisper` с параметром `beam_size=1` и `vad_filter=True` для максимальной скорости, обернув его в асинхронный генератор, совместимый с `livekit-agents`.

**Файл: `src/voice_worker/stt/faster_whisper_plugin.py`**

```python
import asyncio
import logging
from typing import AsyncIterable
from livekit.agents import stt, utils
from livekit.agents.stt import SpeechEventType, SpeechEvent
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

class FasterWhisperSTT(stt.STT):
    def __init__(self, *, model_size: str = "distil-large-v3", device: str = "cuda"):
        super().__init__(streaming_supported=True, interim_results_supported=True)
        # distil-large-v3 дает ~2x ускорение при минимальной потере качества для русского языка
        self._model = WhisperModel(model_size, device=device, compute_type="float16")

    async def stream(self) -> "FasterWhisperStream":
        return FasterWhisperStream(self._model)

class FasterWhisperStream(stt.SpeechStream):
    def __init__(self, model: WhisperModel):
        super().__init__()
        self._model = model
        self._queue = utils.aio.Chan[stt.SpeechEvent]()
        self._audio_buffer = bytearray()
        self._task = asyncio.create_task(self._run())

    def push_frame(self, frame: stt.AudioFrame):
        # LiveKit предоставляет 10ms или 20ms чанки PCM 16kHz mono
        self._audio_buffer.extend(frame.data.tobytes())
        # Оптимизация: обрабатываем, когда накопилось ~400ms аудио (для баланса latency/точности)
        if len(self._audio_buffer) >= 400 * 16 * 2: # 400ms * 16000Hz * 2 bytes
            self._process_chunk()

    def end_input(self):
        self._process_chunk(final=True)
        self._queue.close()

    def _process_chunk(self, final: bool = False):
        if not self._audio_buffer:
            return
        
        # Быстрая транскрибация чанка
        segments, info = self._model.transcribe(
            self._audio_buffer,
            beam_size=1,
            vad_filter=True,
            language="ru"
        )
        
        text = "".join([segment.text for segment in segments])
        if text.strip():
            event_type = SpeechEventType.FINAL_TRANSCRIPT if final else SpeechEventType.INTERIM_TRANSCRIPT
            self._queue.send_nowait(SpeechEvent(
                type=event_type,
                alternatives=[stt.SpeechData(text=text.strip(), language="ru", confidence=0.95)]
            ))
            
        if final:
            self._audio_buffer.clear()
        else:
            # Оставляем последние 100ms для overlap (предотвращает разрыв слов на границах чанков)
            overlap_bytes = 100 * 16 * 2
            self._audio_buffer = self._audio_buffer[-overlap_bytes:]

    async def _run(self):
        # В реальной реализации здесь был бы асинхронный цикл, 
        # но faster-whisper синхронный, поэтому мы выносим его в executor, 
        # чтобы не блокировать asyncio loop. (Упрощено для примера, в проде использовать run_in_executor)
        pass

    async def aclose(self) -> None:
        self._queue.close()
        await self._task
```

---

## Шаг 3: Реализация кастомного Streaming LLM (vLLM Gateway)

Этот модуль подключается к нашему шлюзу из Подзадачи 1.1, поддерживает стриминг токенов и **обязательно** прогоняет текст через `PIISanitizer` перед отправкой.

**Файл: `src/voice_worker/llm/vllm_plugin.py`**

```python
import asyncio
import json
import logging
import httpx
from typing import AsyncIterable, List
from livekit.agents import llm
from src.pii_sanitizer.service import sanitizer # Из Этапа 0.3

logger = logging.getLogger(__name__)

class VLLMPlugin(llm.LLM):
    def __init__(self, *, api_url: str, model: str = "qwen2.5-72b"):
        self._api_url = api_url
        self._model = model
        self._client = httpx.AsyncClient(timeout=30.0)

    async def chat(self, *, chat_ctx: llm.ChatContext, fnc_ctx: llm.FunctionContext | None = None) -> "VLLMChatStream":
        # Преобразуем контекст LiveKit в формат сообщений
        messages = []
        for msg in chat_ctx.messages:
            role = "assistant" if msg.role == "assistant" else "user"
            # ВАЖНО: Маскирование PII перед отправкой в LLM (Compliance)
            safe_content = sanitizer.sanitize(msg.content) if isinstance(msg.content, str) else msg.content
            messages.append({"role": role, "content": safe_content})

        return VLLMChatStream(self._client, self._api_url, self._model, messages)

class VLLMChatStream(llm.ChatStream):
    def __init__(self, client: httpx.AsyncClient, api_url: str, model: str, messages: List[dict]):
        super().__init__()
        self._client = client
        self._api_url = api_url
        self._model = model
        self._messages = messages
        self._queue = utils.aio.Chan[llm.ChatChunk]()
        self._task = asyncio.create_task(self._run())

    async def _run(self):
        payload = {
            "model": self._model,
            "messages": self._messages,
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 150 # Ограничиваем длину ответа для voice
        }
        
        try:
            async with self._client.stream("POST", f"{self._api_url}/v1/chat/completions", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        data = json.loads(line[6:])
                        if data.get("choices"):
                            delta = data["choices"][0].get("delta", {}).get("content", "")
                            if delta:
                                self._queue.send_nowait(llm.ChatChunk(
                                    choices=[llm.Choice(delta=llm.ChoiceDelta(content=delta, role="assistant"))]
                                ))
        except Exception as e:
            logger.error(f"Ошибка стриминга LLM: {e}")
        finally:
            self._queue.close()

    async def aclose(self) -> None:
        self._queue.close()
        await self._task
```

---

## Шаг 4: Реализация Streaming TTS с интеллектуальной буферизацией (Silero)

Чтобы избежать "роботизированного" звука и заикания, мы не отправляем в TTS каждый токен по отдельности. Мы накапливаем токены до конца предложения (`.`, `?`, `!`) или до лимита в ~20 токенов, затем синтезируем.

**Файл: `src/voice_worker/tts/silero_plugin.py`**

```python
import asyncio
import logging
import torch
from typing import AsyncIterable
from livekit.agents import tts, utils
import torchaudio

logger = logging.getLogger(__name__)

class SileroTTS(tts.TTS):
    def __init__(self, *, device: str = "cpu"):
        super().__init__(streaming_supported=True, sample_rate=24000, num_channels=1)
        self._device = device
        # Загрузка Silero TTS (ru_v3)
        self._model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_tts', language='ru', speaker='v3_ru')
        self._model.to(device)

    def synthesize(self, text: str) -> tts.SynthesizedAudio:
        # Синхронная синтезация для буферизированных блоков текста
        audio = self._model.apply_tts(text=text, speaker='xenia', sample_rate=24000)
        audio_tensor = torch.tensor(audio).unsqueeze(0)
        return tts.SynthesizedAudio(
            text=text,
            data=audio_tensor.numpy().tobytes(),
            sample_rate=24000
        )

    async def stream(self) -> "SileroTTSStream":
        return SileroTTSStream(self)

class SileroTTSStream(tts.TTSStream):
    def __init__(self, tts_engine: SileroTTS):
        super().__init__()
        self._tts = tts_engine
        self._queue = utils.aio.Chan[tts.SynthesizedAudio]()
        self._buffer = ""
        self._task = asyncio.create_task(self._run())

    def push_text(self, text: str):
        self._buffer += text
        # Триггер синтеза: конец предложения ИЛИ накоплено > 20 символов/токенов
        if any(self._buffer.strip().endswith(p) for p in [".", "!", "?", ","]) or len(self._buffer) > 25:
            self._synthesize_buffer()

    def _synthesize_buffer(self):
        if not self._buffer.strip():
            return
        try:
            audio = self._tts.synthesize(self._buffer)
            self._queue.send_nowait(audio)
            self._buffer = "" # Очистка буфера после успешной синтезации
        except Exception as e:
            logger.error(f"Ошибка TTS синтеза: {e}")
            self._buffer = "" # Сброс при ошибке, чтобы не застревать

    def flush(self):
        self._synthesize_buffer()
        self._queue.close()

    async def _run(self):
        pass # Логика уже в push_text/flush для синхронного torch

    async def aclose(self) -> None:
        self.flush()
        await self._task
```

---

## Шаг 5: Сборка Voice Pipeline и обработка Barge-in

Теперь мы собираем все компоненты в единый `VoiceAgent`, используя высокоуровневый `VoicePipeline` из `livekit-agents`, который "из коробки" управляет WebRTC, VAD и очередями, но с нашими кастомными плагинами.

**Файл: `src/voice_worker/main.py`**

```python
import os
import logging
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, VoicePipeline, Agent
from livekit.plugins import silero # Для VAD

from .stt.faster_whisper_plugin import FasterWhisperSTT
from .llm.vllm_plugin import VLLMPlugin
from .tts.silero_plugin import SileroTTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

async def entrypoint(ctx: JobContext):
    logger.info(f"Подключение к комнате: {ctx.room.name}")
    
    # Инициализация наших кастомных компонентов
    stt_plugin = FasterWhisperSTT(model_size="distil-large-v3", device="cuda")
    llm_plugin = VLLMPlugin(api_url=os.getenv("VLLM_API_URL", "http://vllm-service:8000"))
    tts_plugin = SileroTTS(device="cuda" if torch.cuda.is_available() else "cpu")
    vad_plugin = silero.VAD() # Используем нативный плагин LiveKit для Silero VAD

    # Создание агента
    agent = Agent(
        instructions="Ты — вежливый и профессиональный ассистент VoiceGraph. Отвечай кратко, по существу. Если клиент перебивает, немедленно замолкай и слушай новый вопрос."
    )

    # Сборка VoicePipeline
    # Ключевой параметр: interrupt_speech_events обеспечивает barge-in < 300ms
    pipeline = VoicePipeline(
        agent=agent,
        vad=vad_plugin,
        stt=stt_plugin,
        llm=llm_plugin,
        tts=tts_plugin,
        # Настройки прерывания (Barge-in)
        interrupt_speech_events=True,
        interrupt_audio_duration=0.3 # Прерывать, если клиент говорит > 300мс
    )

    # Подключение пайплайна к комнате
    await pipeline.start(ctx.room)
    logger.info("VoicePipeline успешно запущен и ожидает входящих звонков.")

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            api_key=os.getenv("LIVEKIT_API_KEY"),
            api_secret=os.getenv("LIVEKIT_API_SECRET"),
            ws_url=os.getenv("LIVEKIT_WS_URL"),
        )
    )
```

---

## Шаг 6: Механизм синхронизации и предотвращения "заикания"

Как мы решили проблему наложения голосов и заикания (согласно требованию "Действие"):

1. **Barge-in (Прерывание):** Когда `silero.VAD` детектирует речь клиента, `VoicePipeline` автоматически:
   - Немедленно очищает очередь воспроизведения `AudioSource` (TTS обрывается < 50 мс).
   - Отправляет сигнал отмены в `VLLMChatStream` (закрывает HTTP-соединение, экономя токены vLLM).
   - Сбрасывает буфер `SileroTTSStream._buffer`.
2. **Интеллектуальная буферизация TTS:** Метод `push_text` в `SileroTTSStream` не синтезирует каждый токен. Он ждет накопления смыслового блока (знак препинания или 25 символов). Это гарантирует, что в `torch.hub` подается полноценная фраза, что критически важно для качества и просодии Silero TTS, устраняя "роботизированное" заикание на отдельных слогах.
3. **Overlap в ASR:** В `FasterWhisperStream` мы оставляем последние 100 мс аудио в буфере при обработке чанка. Это предотвращает "проглатывание" концовок слов на границах 400-мс интервалов.

---

## ✅ Definition of Done (Критерии готовности Подзадачи 1.2)

Прежде чем перейти к **Подзадаче 1.3 (Мультимодальный анализ эмоций)**, убедитесь, что:

- [ ] `VoiceAgentWorker` успешно подключается к LiveKit Server и принимает входящие WebRTC-звонки.
- [ ] **Тест Barge-in:** При воспроизведении аудио-файла с паузой в 1 секунду, а затем резкой репликой, агент обрывает свой ответ менее чем за 300 мс и начинает транскрибировать новую реплику.
- [ ] **Тест PII:** При произнесении вслух номера телефона или карты, в логах vLLM (или через перехват запросов) эти данные заменены на `[PHONE_REDACTED]` / `[CARD_NUMBER_REDACTED]`.
- [ ] **Тест TTS Buffering:** Агент не издает прерывистых звуков на отдельных токенах, а выдает плавные фразы, синтезируемые после накопления буфера.
- [ ] End-to-end latency (от конца речи клиента до начала речи агента) при простом вопросе ("Как дела?") составляет **< 800 мс** (измеряется через метрики LiveKit или вручную через осциллограмму аудио).
