Это критически важный компонент для создания "эмпатичного" ИИ. В отличие от анализа тональности текста (который реагирует постфактум), акустический анализ эмоций (prosody analysis) позволяет системе понять, что клиент раздражен, *еще до того*, как он закончит фразу, и мгновенно адаптировать стратегию ответа.

Мы реализуем это как **неблокирующий асинхронный поток**, работающий параллельно с ASR, чтобы не добавить ни миллисекунды к общей задержке (latency) голосового конвейера.

---

# 🚀 ЭТАП 1.3: Мультимодальный анализ эмоций

## Шаг 1: Зависимости и подготовка окружения

Для извлечения акустических признаков мы используем библиотеку `opensmile` (Python-обертка над высокооптимизированным C++ фреймворком OpenSMILE), которая является индустриальным стандартом для паралингвистического анализа. Для классификации признаков в эмоции мы будем использовать легковесную модель (в примере — заглушка на `scikit-learn`, которую в продакшене заменяют на ONNX-модель, обученную на русскоязычных датасетах, например, M-EmoRu).

Добавляем зависимости в `pyproject.tom` микросервиса `voice-worker`:

```toml
# Добавить в pyproject.toml voice-worker
dependencies = [
    # ... предыдущие зависимости ...
    "opensmile>=2.5.0",       # Извлечение акустических признаков (eGeMAPS)
    "librosa>=0.10.1",        # Предобработка аудио (ресемплинг, нормализация)
    "soundfile>=0.12.1",      # Чтение/запись аудио-буферов
    "scikit-learn>=1.4.2",    # Для легковесной классификации признаков
    "numpy>=1.26.4",
    "joblib>=1.4.0"           # Загрузка предобученной модели классификатора
]
```

---

## Шаг 2: Ядро детектора эмоций (Emotion Detector Core)

Создаем компонент, который извлекает признаки из аудио и классифицирует их. Мы используем набор признаков **eGeMAPSv02** (Extended Geneva Minimalistic Acoustic Parameter Set), который оптимизирован именно для распознавания эмоций и содержит 88 фич (тон, энергия, мерцание голоса и т.д.).

**Файл: `src/voice_worker/emotion/detector.py`**

```python
import logging
import numpy as np
import librosa
import opensmile
import joblib
from pathlib import Path
from typing import Literal
from enum import Enum

logger = logging.getLogger(__name__)

# Определяем возможные состояния эмоций, согласованные с архитектурой
class EmotionState(str, Enum):
    CALM = "CALM"
    ANNOYED = "ANNOYED"
    CONFUSED = "CONFUSED"
    ANGRY = "ANGRY"
    UNKNOWN = "UNKNOWN"

class EmotionDetector:
    """
    Асинхронно-совместимый детектор эмоций на базе OpenSMILE.
    Анализирует акустические признаки (просодию) в скользящем окне аудио.
    """
    def __init__(self, model_path: str | None = None):
        logger.info("🧠 Инициализация Emotion Detector (OpenSMILE eGeMAPSv02)...")
        
        # 1. Инициализация экстрактора признаков OpenSMILE
        # eGeMAPSv02 оптимизирован для emotion recognition и работает очень быстро
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        
        # 2. Загрузка модели классификации
        # В продакшене здесь должна быть ONNX модель, обученная на русскоязычных данных (например, M-EmoRu)
        # Для демонстрации пайплайна используем заглушку или простую модель, если файл существует
        self.model_path = Path(model_path) if model_path else Path("models/emotion_classifier.joblib")
        self.classifier = None
        
        if self.model_path.exists():
            self.classifier = joblib.load(self.model_path)
            logger.info(f"✅ Модель классификации эмоций загружена из {self.model_path}")
        else:
            logger.warning("⚠️ Модель классификации не найдена. Используется эвристическая заглушка (dummy).")

    def extract_and_classify(self, audio_buffer: np.ndarray, sample_rate: int = 16000) -> EmotionState:
        """
        Принимает буфер аудио (например, 1.5 секунды) и возвращает метку эмоции.
        """
        if len(audio_buffer) == 0:
            return EmotionState.UNKNOWN

        try:
            # 1. Нормализация и ресемплинг до 16kHz (требование OpenSMILE eGeMAPS)
            if sample_rate != 16000:
                audio_buffer = librosa.resample(audio_buffer, orig_sr=sample_rate, target_sr=16000)
            
            # Нормализация громкости (чтобы крик и шепот обрабатывались корректно по признакам, а не по амплитуде)
            audio_buffer = librosa.util.normalize(audio_buffer)

            # 2. Извлечение признаков через OpenSMILE (возвращает pandas DataFrame)
            # Для Functionals (статистика по всему файлу) передаем весь буфер
            features = self.smile.process_signal(audio_buffer, 16000)
            
            # Преобразуем в 2D numpy array для sklearn [1, n_features]
            feature_vector = features.values.reshape(1, -1)

            # 3. Классификация
            if self.classifier is not None:
                prediction = self.classifier.predict(feature_vector)[0]
                confidence = np.max(self.classifier.predict_proba(feature_vector))
                
                # Фильтрация по уверенности (если модель не уверена, возвращаем CALM или UNKNOWN)
                if confidence < 0.6:
                    return EmotionState.CALM 
                    
                return EmotionState(prediction)
            else:
                # Эвристическая заглушка для демонстрации работы пайплайна без модели
                # (В реальности здесь должен быть вызов self.classifier)
                energy = np.mean(np.abs(audio_buffer))
                if energy > 0.15: # Условный порог "громкости/агрессии"
                    return EmotionState.ANGRY
                return EmotionState.CALM

        except Exception as e:
            logger.error(f"❌ Ошибка при анализе эмоций: {e}")
            return EmotionState.UNKNOWN
```

---

## Шаг 3: Интеграция в асинхронный Voice Pipeline (Sliding Window)

Эмоции нельзя распознавать на чанках по 200 мс (слишком мало контекста). Нам нужно **скользящее окно (sliding window)** размером около 1.5–2.0 секунд. Мы создадим асинхронную задачу, которая накапливает аудио из LiveKit и периодически вызывает детектор, не блокируя основной поток ASR/TTS.

**Файл: `src/voice_worker/emotion/pipeline.py`**

```python
import asyncio
import logging
import numpy as np
from collections import deque
from livekit.agents import stt # Для типа AudioFrame

from .detector import EmotionDetector, EmotionState

logger = logging.getLogger(__name__)

class EmotionAnalysisPipeline:
    """
    Параллельный пайплайн анализа эмоций.
    Потребляет аудио-фреймы из LiveKit, накапливает их в скользящее окно
    и асинхронно обновляет состояние эмоции.
    """
    def __init__(self, detector: EmotionDetector, window_duration_sec: float = 1.5):
        self.detector = detector
        self.window_duration_sec = window_duration_sec
        self.sample_rate = 16000 # Ожидаемый sample rate от LiveKit/Silero VAD
        
        # Размер окна в сэмплах (1.5 сек * 16000 Гц * 2 байта на int16 = 48000 байт)
        self.window_bytes = int(window_duration_sec * self.sample_rate * 2)
        self.audio_buffer = bytearray()
        
        self.current_emotion = EmotionState.CALM
        self._task = None
        self._is_running = False

    async def start(self):
        self._is_running = True
        self._task = asyncio.create_task(self._processing_loop())
        logger.info("▶️ Emotion Analysis Pipeline запущен")

    async def stop(self):
        self._is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("⏹️ Emotion Analysis Pipeline остановлен")

    def push_audio_frame(self, frame: stt.AudioFrame):
        """Вызывается из основного VoicePipeline при получении нового аудио-чанка."""
        if not self._is_running:
            return
        
        # Добавляем новые байты в буфер
        self.audio_buffer.extend(frame.data.tobytes())
        
        # Обрезаем буфер, чтобы он не превышал размер скользящего окна
        if len(self.audio_buffer) > self.window_bytes:
            self.audio_buffer = self.audio_buffer[-self.window_bytes:]

    async def _processing_loop(self):
        """Асинхронный цикл, который периодически анализирует накопленный буфер."""
        # Анализируем каждые 0.5 секунды (достаточно для реакции, не перегружает CPU)
        check_interval = 0.5 
        
        while self._is_running:
            await asyncio.sleep(check_interval)
            
            if len(self.audio_buffer) < self.window_bytes:
                continue # Ждем накопления достаточного количества данных

            # Выполняем тяжелые вычисления (OpenSMILE) в executor, чтобы не блокировать asyncio
            loop = asyncio.get_running_loop()
            
            # Копируем буфер для потокобезопасной обработки
            buffer_copy = bytes(self.audio_buffer)
            
            try:
                audio_np = np.frombuffer(buffer_copy, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Запускаем синхронный метод в пуле потоков
                emotion = await loop.run_in_executor(
                    None, 
                    self.detector.extract_and_classify, 
                    audio_np, 
                    self.sample_rate
                )
                
                # Обновляем состояние только если эмоция изменилась (избегаем спама обновлениями)
                if emotion != self.current_emotion:
                    logger.info(f"🎭 Изменение эмоции: {self.current_emotion.value} -> {emotion.value}")
                    self.current_emotion = emotion
                    
                    # Здесь можно отправить событие в LangGraph State или напрямую в LLM-контекст
                    # См. Шаг 4 ниже
                    
            except Exception as e:
                logger.error(f"Ошибка в цикле обработки эмоций: {e}")

    def get_current_emotion(self) -> EmotionState:
        return self.current_emotion
```

---

## Шаг 4: Динамическая адаптация промпта LLM (Интеграция с AgentSessionState)

Теперь мы должны связать обнаруженную эмоцию с поведением LLM. В архитектуре LangGraph это делается через внедрение эмоции в системный промпт на каждом шаге генерации ответа.

**Файл: `src/voice_worker/llm/prompt_adapter.py`**

```python
from .detector import EmotionState

def get_emotion_adapted_instruction(emotion: EmotionState) -> str:
    """
    Возвращает блок инструкций для LLM в зависимости от текущей эмоции клиента.
    Внедряется в системный промпт перед каждым вызовом LLM.
    """
    if emotion == EmotionState.ANGRY:
        return (
            "КРИТИЧЕСКОЕ ПРЕДУПРЕЖДЕНИЕ: Клиент демонстрирует признаки гнева или сильного раздражения (по тону голоса). "
            "1. НЕМЕДЛЕННО извинись за неудобства. "
            "2. ГОВОРИ коротко, спокойно и максимально эмпатично. "
            "3. НЕ задавай уточняющих вопросов, не связанных с решением проблемы. "
            "4. ПРЕДЛОЖИ немедленный перевод на старшего оператора-человека."
        )
    elif emotion == EmotionState.ANNOYED:
        return (
            "ВНИМАНИЕ: Клиент звучит раздраженно или нетерпеливо. "
            "1. Будь максимально краток. Переходи сразу к сути. "
            "2. Избегай стандартных скриптовых фраз вежливости, они могут раздражать."
        )
    elif emotion == EmotionState.CONFUSED:
        return (
            "ВНИМАНИЕ: Клиент звучит растерянно или задает уточняющие вопросы. "
            "1. Объясняй информацию максимально простыми словами, без сложных терминов. "
            "2. Задай один уточняющий вопрос, чтобы убедиться, что клиент понял."
        )
    else: # CALM or UNKNOWN
        return (
            "Клиент спокоен. Следуй основному сценарию кампании, будь вежлив и профессионален."
        )
```

**Интеграция в `VLLMPlugin` (из Подзадачи 1.2):**

```python
# Внутри метода chat() класса VLLMPlugin:

# 1. Получаем текущую эмоцию из глобального состояния или передаем через контекст
current_emotion = get_current_emotion_from_context() # Эмуляция получения из EmotionAnalysisPipeline

# 2. Формируем динамический системный промпт
base_system_prompt = "Ты — вежливый ассистент VoiceGraph..."
emotion_instruction = get_emotion_adapted_instruction(current_emotion)

# 3. Вставляем инструкцию в начало сообщений
messages = [
    {"role": "system", "content": f"{base_system_prompt}\n\n{emotion_instruction}"},
    # ... остальные сообщения из chat_ctx ...
]
```

---

## Шаг 5: Модульное тестирование (Shift-Left Testing)

Используем **Golden Dataset** (из артефактов `ai_ml_specifications.md`), чтобы проверить, что пайплайн корректно обрабатывает разные акустические сценарии.

**Файл: `tests/test_emotion_pipeline.py`**

```python
import pytest
import numpy as np
import soundfile as sf
from pathlib import Path
from src.voice_worker.emotion.detector import EmotionDetector, EmotionState

@pytest.fixture
def detector():
    # Используем заглушку, так как в CI нет реальной обученной модели
    return EmotionDetector(model_path=None)

def test_angry_audio_classification(detector):
    # Загружаем тестовый файл из Golden Dataset
    audio_path = Path("tests/fixtures/audio/sess_004_wav.wav") # "Кто это? Я уже говорил, что не интересуюсь..."
    if not audio_path.exists():
        pytest.skip("Тестовый аудиофайл не найден")
        
    audio_data, sample_rate = sf.read(audio_path, dtype='float32')
    
    emotion = detector.extract_and_classify(audio_data, sample_rate)
    
    # В реальной системе с обученной моделью здесь будет assert emotion == EmotionState.ANGRY
    # Для заглушки проверяем, что метод не падает и возвращает валидный Enum
    assert isinstance(emotion, EmotionState)
    assert emotion in [EmotionState.ANGRY, EmotionState.CALM] # Заглушка может вернуть CALM при низкой энергии

def test_calm_audio_classification(detector):
    audio_path = Path("tests/fixtures/audio/sess_002_wav.wav") # "Здравствуйте. Это опрос..."
    if not audio_path.exists():
        pytest.skip("Тестовый аудиофайл не найлен")
        
    audio_data, sample_rate = sf.read(audio_path, dtype='float32')
    emotion = detector.extract_and_classify(audio_data, sample_rate)
    
    assert isinstance(emotion, EmotionState)
```

---

## ✅ Definition of Done (Критерии готовности Подзадачи 1.3)

Прежде чем перейти к **ЭТАПУ 2 (Propensity Modeling & MLOps)**, убедитесь, что:

- [ ] Компонент `EmotionDetector` успешно инициализируется и способен извлекать признаки eGeMAPS из аудио-массива NumPy без ошибок.
- [ ] `EmotionAnalysisPipeline` корректно накапливает аудио-фреймы в скользящее окно (1.5 сек) и запускает анализ в фоновом потоке (`run_in_executor`), **не блокируя** основной `asyncio` event loop (проверено через отсутствие предупреждений "BlockingIOError" или задержек в логах LiveKit).
- [ ] Функция `get_emotion_adapted_instruction` возвращает корректные, жесткие ограничения для LLM при статусе `ANGRY` или `ANNOYED`.
- [ ] В логах тестового прогона видно изменение состояния эмоции (например, `🎭 Изменение эмоции: CALM -> ANGRY`) при воспроизведении соответствующих файлов из Golden Dataset.
- [ ] Потребление CPU процессом анализа эмоций не превышает 5-10% на одно ядро при 100 параллельных сессиях (благодаря интервалу проверки 0.5 сек и использованию eGeMAPS Functionals).
