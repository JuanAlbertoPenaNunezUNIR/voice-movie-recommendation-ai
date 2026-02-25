import base64
import tempfile
import subprocess
import os
import time
import logging
import wave
import io

logger = logging.getLogger("VoiceStreamService")

class VoiceStreamService:
    """
    Servicio de pseudo-streaming de audio con Whisper.

    Whisper NO es streaming nativo.
    Este servicio simula streaming acumulando audio,
    convirtiéndolo a WAV 16kHz y transcribiendo parcialmente.
    """

    MAX_AUDIO_SECONDS = 30          # límite por sesión
    SAMPLE_RATE = 16000
    BYTES_PER_SECOND = SAMPLE_RATE * 2  # PCM 16-bit mono

    def __init__(self, voice_processor):
        self.voice_processor = voice_processor

        # session_id → buffer
        self.audio_buffers: Dict[str, bytes] = {}

        # session_id → último texto enviado
        self.last_text: Dict[str, str] = {}

        # session_id → timestamp
        self.last_activity: Dict[str, float] = {}

    def _webm_to_pcm(self, audio_bytes: bytes) -> bytes:
        """
        Convierte webm/opus → PCM 16-bit mono 16kHz (RAW)
        """

        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            f.write(audio_bytes)
            webm_path = f.name

        pcm_path = webm_path.replace(".webm", ".pcm")

        cmd = [
            "ffmpeg",
            "-y",
            "-i", webm_path,
            "-ac", "1",
            "-ar", "16000",
            "-f", "s16le",
            pcm_path
        ]

        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )

            with open(pcm_path, "rb") as f:
                pcm_bytes = f.read()

        finally:
            if os.path.exists(webm_path):
                os.remove(webm_path)
            if os.path.exists(pcm_path):
                os.remove(pcm_path)

        return pcm_bytes

    def append_audio(self, session_id: str, audio_b64: str):
        """
        Recibe chunk desde frontend (webm/opus base64)
        """

        try:
            raw_bytes = base64.b64decode(audio_b64)
            pcm_bytes = self._webm_to_pcm(raw_bytes)

            if session_id not in self.audio_buffers:
                self.audio_buffers[session_id] = b""
                self.last_text[session_id] = ""
                self.last_activity[session_id] = time.time()

            self.audio_buffers[session_id] += pcm_bytes
            self.last_activity[session_id] = time.time()

            # 🔒 limitar memoria
            max_bytes = self.MAX_AUDIO_SECONDS * self.BYTES_PER_SECOND
            if len(self.audio_buffers[session_id]) > max_bytes:
                self.audio_buffers[session_id] = \
                    self.audio_buffers[session_id][-max_bytes:]

        except Exception as e:
            logger.error(f"❌ Error procesando chunk: {e}")
            raise

    async def process_partial(self, session_id: str) -> str:
        """
        Transcribe parcialmente el audio acumulado.
        Devuelve solo texto nuevo (no repetido).
        """

        pcm_audio = self.audio_buffers.get(session_id)
        wav_audio = self._pcm_to_wav_bytes(pcm_audio)

        if not pcm_audio or len(pcm_audio) < self.BYTES_PER_SECOND:
            return ""

        try:
            result = await self.voice_processor.transcribe(wav_audio)

            text = result.get("text", "").strip()
            confidence = result.get("confidence", 0.0)

            previous = self.last_text.get(session_id, "")

            # evitar repetir texto
            if text.startswith(previous):
                new_text = text[len(previous):].strip()
            else:
                new_text = text

            if new_text:
                self.last_text[session_id] = text
                logger.debug(
                    f"🎤 parcial='{new_text}' conf={confidence:.2f}"
                )

            return new_text

        except Exception as e:
            logger.error(f"❌ Error pseudo-streaming: {e}")
            return ""

    def _pcm_to_wav_bytes(self, pcm_bytes: bytes) -> bytes:
        wav_buffer = io.BytesIO()

        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16 bits
            wf.setframerate(16000)
            wf.writeframes(pcm_bytes)

        wav_buffer.seek(0)
        return wav_buffer.read()

    
    def clear_session(self, session_id: str):
        self.audio_buffers.pop(session_id, None)
        self.last_text.pop(session_id, None)
        self.last_activity.pop(session_id, None)

    def cleanup_inactive_sessions(self, timeout_seconds: int = 60):
        """
        Limpia sesiones inactivas
        """
        now = time.time()
        to_delete = [
            sid for sid, ts in self.last_activity.items()
            if now - ts > timeout_seconds
        ]

        for sid in to_delete:
            self.clear_session(sid)
            logger.info(f"🧹 Sesión {sid} limpiada por inactividad")