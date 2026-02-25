# Procesador de voz con Whisper y transcripción interactiva en tiempo real

import whisper
import numpy as np
import base64
import io
import asyncio
import logging
import soundfile as sf
from typing import Optional, Dict
from utils.device_manager import device_manager
from TTS.api import TTS

logger = logging.getLogger(__name__)

class VoiceProcessor:
    """
    Whisper OpenAI — máxima precisión en español
    """

    def __init__(self, model_size: str = "medium"):
        self.model_size = model_size
        self.model: Optional[whisper.Whisper] = None
        self.tts = None

    def get_tts(self):
        if self.tts is None:
            self.tts = TTS(
                "tts_models/multilingual/multi-dataset/xtts_v2",
                gpu=True
            )
        return self.tts

    async def initialize(self):
        device = device_manager.get_device_str()
        logger.info(f"🎙️ Cargando Whisper {self.model_size} en {device.upper()}")

        def _load():
            return whisper.load_model(self.model_size, device=device)

        self.model = await asyncio.to_thread(_load)
        logger.info("✅ Whisper cargado")

    async def transcribe(
        self,
        audio_b64: str,
        language: str = "es"
    ) -> Dict:
        if not self.model:
            await self.initialize()

        audio_bytes = base64.b64decode(audio_b64)

        with io.BytesIO(audio_bytes) as bio:
            audio, sr = sf.read(bio, dtype="float32")

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        result = await asyncio.to_thread(
            self.model.transcribe,
            audio,
            language="es",
            task="transcribe",
            temperature=0.0,
            fp16=(device_manager.get_device_str() == "cuda"),
            condition_on_previous_text=False,
            initial_prompt=(
                "El usuario habla español. "
                "No traduzcas. No uses inglés."
            )
        )

        segments = result.get("segments", [])

        confidence = self._calculate_confidence(segments)

        return {
            "text": result["text"].strip(),
            "segments": segments,
            "language": result.get("language", "es"),
            "confidence": confidence
        }

    def reload(self):
        self.model = None

    def is_available(self) -> bool:
        return self.model is not None

    def _calculate_confidence(self, segments):
        """
        Confidence REAL basado en avg_logprob de Whisper
        """
        import math

        if not segments:
            return 0.0

        log_probs = [
            s["avg_logprob"]
            for s in segments
            if "avg_logprob" in s
        ]

        if not log_probs:
            return 0.0

        avg = sum(log_probs) / len(log_probs)
        return round(math.exp(avg), 3)
