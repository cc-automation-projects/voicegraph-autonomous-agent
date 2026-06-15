import asyncio
import logging

import numpy as np
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, JobProcess, WorkerOptions, cli, llm, stt
from silero_vad import get_speech_timestamps, load_silero_vad

from src.memory.manager import MemoryManager
from src.pii_sanitizer.service import sanitizer
from src.voice_worker.emotion.detector import EmotionDetector
from src.voice_worker.emotion.pipeline import EmotionAnalysisPipeline
from src.voice_worker.llm.vllm_plugin import VLLMPlugin
from src.voice_worker.stt.faster_whisper_plugin import FasterWhisperSTT
from src.voice_worker.tts.silero_plugin import SileroTTS

logger = logging.getLogger(__name__)

_stt = None
_tts = None
_llm = None
_emotion_detector = None
_memory_manager = None
_vad_model = None

def prewarm(proc: JobProcess):
    global _stt, _tts, _llm, _emotion_detector, _vad_model
    _stt = FasterWhisperSTT(model_size="distil-large-v3", device="cuda")
    _tts = SileroTTS(device="cuda")
    _llm = VLLMPlugin(api_url="http://vllm-service:8000")
    _emotion_detector = EmotionDetector(model_path=None)
    _vad_model = load_silero_vad()

    proc.userdata["stt"] = _stt
    proc.userdata["tts"] = _tts
    proc.userdata["llm"] = _llm
    proc.userdata["emotion_detector"] = _emotion_detector
    proc.userdata["vad_model"] = _vad_model

async def entrypoint(job_ctx: JobContext):
    global _memory_manager
    _memory_manager = MemoryManager()
    await _memory_manager.init()

    stt_engine = job_ctx.proc.userdata["stt"]
    tts_engine = job_ctx.proc.userdata["tts"]
    llm_engine = job_ctx.proc.userdata["llm"]
    emotion_detector = job_ctx.proc.userdata["emotion_detector"]
    vad_model = job_ctx.proc.userdata["vad_model"]

    emotion_pipeline = EmotionAnalysisPipeline(detector=emotion_detector, window_duration_sec=1.5)
    await emotion_pipeline.start()

    user_id = job_ctx.room.name or "unknown_user"
    memories = await _memory_manager.retrieve_relevant(
        user_id=user_id,
        query="предпочтения, жалобы, прошлые взаимодействия",
        top_k=5,
    )
    memory_context = "\n".join([f"- {m.fact}" for m in memories])
    job_ctx.proc.userdata["memory_context"] = memory_context

    room = job_ctx.room
    logger.info(f"Connecting to room {room.name}...")
    await job_ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    @room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.Participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.create_task(
                handle_audio_track(track, user_id, stt_engine, tts_engine,
                                   llm_engine, memory_context, vad_model,
                                   emotion_pipeline)
            )

    try:
        await asyncio.Event().wait()
    finally:
        await emotion_pipeline.stop()
        if _memory_manager:
            await _memory_manager.shutdown()

async def handle_audio_track(track: rtc.Track, user_id: str, stt_engine,
                              tts_engine, llm_engine, memory_context: str,
                              vad_model, emotion_pipeline):
    audio_stream = rtc.AudioStream(track)
    stt_stream = stt_engine.stream()
    tts_stream = tts_engine.stream()

    source = rtc.AudioSource(tts_engine.sample_rate, tts_engine.num_channels)
    publication = await track.local_participant.publish_track(source)
    logger.info(f"TTS source published for user {user_id}")

    first_transcript = True
    current_tts_task = None
    audio_buffer = bytearray()
    vad_window_ms = 200
    bytes_per_ms = int(tts_engine.sample_rate * 2 / 1000)
    vad_window_bytes = vad_window_ms * bytes_per_ms

    async def generate_response(user_text: str):
        nonlocal current_tts_task
        chat_ctx = llm.ChatContext()
        chat_ctx.messages.append(llm.ChatMessage(role="user", content=user_text))
        if memory_context:
            chat_ctx.messages.insert(
                0, llm.ChatMessage(role="system",
                                   content=f"Контекст пользователя: {memory_context}")
            )

        tts_buffer = []
        async for chunk in await llm_engine.chat(chat_ctx=chat_ctx):
            if chunk.choices and chunk.choices[0].delta.content:
                text_chunk = chunk.choices[0].delta.content
                tts_buffer.append(text_chunk)
                tts_stream.push_text(text_chunk)
        tts_stream.flush()
        async for audio_data in tts_stream:
            frame = rtc.AudioFrame(
                data=audio_data.data.tobytes(),
                sample_rate=tts_engine.sample_rate,
                num_channels=tts_engine.num_channels,
                samples_per_channel=len(audio_data.data) // tts_engine.num_channels,
            )
            source.capture_frame(frame)

    async def barge_in():
        nonlocal current_tts_task
        if current_tts_task and not current_tts_task.done():
            current_tts_task.cancel()
            try:
                await current_tts_task
            except asyncio.CancelledError:
                pass
            logger.info("Barge-in: cancelled TTS generation")
        tts_stream.flush()
        stt_stream.interrupt()

    async for frame_event in audio_stream:
        audio_buffer.extend(frame_event.frame.data.tobytes())
        if len(audio_buffer) >= vad_window_bytes:
            audio_np = np.frombuffer(audio_buffer[:vad_window_bytes], dtype=np.int16).astype(np.float32) / 32768.0
            speech_ts = get_speech_timestamps(audio_np, vad_model, threshold=0.5, return_seconds=True)
            if speech_ts:
                await barge_in()
            audio_buffer = audio_buffer[vad_window_bytes//2:]

        stt_stream.push_frame(frame_event.frame)

        async for event in stt_stream:
            if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                user_text = event.alternatives[0].text
                safe_text = sanitizer.sanitize(user_text)

                if first_transcript:
                    greeting = "Этот разговор может быть записан в целях контроля качества. "
                    if memory_context:
                        greeting += "Я вижу, у вас были вопросы ранее. Давайте уточним, всё ли в порядке?"
                    else:
                        greeting += "Здравствуйте! Меня зовут Елена, я ваш персональный менеджер."
                    tts_stream.push_text(greeting)
                    tts_stream.flush()
                    first_transcript = False
                else:
                    current_tts_task = asyncio.create_task(generate_response(safe_text))

    await stt_stream.aclose()
    tts_stream.flush()
    await publication.unpublish()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
