import asyncio
from pathlib import Path
import logging
import base64
import json
import shutil
import os
import torch
from typing import Optional, List, Dict
from utils.device_manager import device_manager

logger = logging.getLogger(__name__)

class TTSProcessor:
    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"):
        self.model_name = model_name
        self.model = None
        self.temp_dir = Path("temp_audio")
        self.temp_dir.mkdir(exist_ok=True)
        # Carpeta persistente para voces clonadas
        self.clones_dir = Path("voice_clones")
        self.clones_dir.mkdir(exist_ok=True)

    async def initialize(self):
        device = device_manager.get_device_str()
        logger.info(f"🗣️ Cargando TTS (XTTS v2) en {device.upper()}...")
        
        try:
            if self.model:
                del self.model
                device_manager.clear_cache()

            def _load():
                from TTS.api import TTS
                # Inicializar TTS y mover al dispositivo correcto
                tts = TTS(self.model_name)
                return tts.to(device)

            self.model = await asyncio.to_thread(_load)
            logger.info("✅ TTS Cargado correctamente.")
        except Exception as e:
            logger.error(f"❌ Error cargando TTS: {e}")

    def is_available(self) -> bool:
        return self.model is not None

    def get_available_clones(self) -> List[Dict]:
        """Lista las voces clonadas existentes."""
        voices = []
        if self.clones_dir.exists():
            for d in self.clones_dir.iterdir():
                meta_path = d / "metadata.json"
                if meta_path.exists():
                    try:
                        with open(meta_path) as f:
                            data = json.load(f)
                            voices.append({"id": data["id"], "name": data["name"]})
                    except: pass
        return voices

    async def clone_voice(self, user_id: str, voice_name: str, reference_audios: List[bytes]) -> str:
        """Procesa los audios para crear una entrada de voz clonada."""
        try:
            user_dir = self.clones_dir / user_id
            user_dir.mkdir(parents=True, exist_ok=True)
            
            # Guardamos el primer audio como referencia principal (XTTS usa one-shot reference)
            ref_path = user_dir / "reference.wav"
            with open(ref_path, "wb") as f:
                f.write(reference_audios[0])
            
            # Guardar metadatos
            metadata = {
                "id": user_id, 
                "name": voice_name, 
                "ref_path": str(ref_path.absolute())
            }
            with open(user_dir / "metadata.json", "w") as f:
                json.dump(metadata, f)
            
            logger.info(f"✅ Voz clonada guardada: {voice_name}")
            return str(ref_path)
        except Exception as e:
            logger.error(f"❌ Error clonando voz: {e}")
            raise

    def delete_voice(self, voice_id: str) -> bool:
        """Elimina una voz clonada y libera el espacio en disco."""
        try:
            # Sanitización básica del ID para evitar path traversal
            safe_id = Path(voice_id).name
            voice_dir = self.clones_dir / safe_id
            
            # Si no existe tal cual, probamos quitando la extensión (ej: "juan.wav" -> "juan")
            if not voice_dir.exists():
                voice_dir = self.clones_dir / Path(safe_id).stem

            if voice_dir.exists() and voice_dir.is_dir():
                shutil.rmtree(voice_dir)
                logger.info(f"🗑️ Voz eliminada correctamente: {safe_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Error eliminando voz {voice_id}: {e}")
            return False

    async def synthesize(self, text: str, voice_id: str = "default", language: str = "es") -> bytes:
        if not self.model: await self.initialize()
        
        output_path = self.temp_dir / f"output_{os.getpid()}_{hash(text)}.wav"
        
        try:
            speaker_wav = None
            fallback_speaker = None

            # Verificar si es una voz clonada explícita
            if voice_id != "default":
                clone_path = self.clones_dir / voice_id / "reference.wav"
                if clone_path.exists():
                    speaker_wav = str(clone_path.absolute())
                    logger.info(f"Sintetizando con voz clonada seleccionada: {voice_id}")
                else:
                    logger.error(f"No se encontró la referencia de la voz clonada seleccionada: {voice_id}")
                    return None
            
            # Si no hay speaker_wav pero hay clones disponibles, usar el primero
            if not speaker_wav and voice_id == "default":
                available_clones = self.get_available_clones()
                if available_clones:
                    first_clone_id = available_clones[0]["id"]
                    clone_path = self.clones_dir / first_clone_id / "reference.wav"
                    if clone_path.exists():
                        speaker_wav = str(clone_path.absolute())
                        logger.info(f"Usando primera voz clonada disponible: {first_clone_id}")

            # Como último recurso (solo si no se pidió clon explícito), si el modelo expone speakers, usa el primero
            if voice_id == "default" and not speaker_wav and hasattr(self.model, "speakers"):
                try:
                    speakers = getattr(self.model, "speakers", [])
                    if speakers:
                        fallback_speaker = speakers[0]
                        logger.info(f"Usando speaker interno por defecto: {fallback_speaker}")
                except Exception:
                    pass
            
            if not speaker_wav and not fallback_speaker:
                logger.warning("No hay voz clonada disponible. XTTS requiere un speaker_wav o speaker interno.")
                # Si no hay referencia ni speaker interno, avisamos y abortamos con None
                return None
            
            # Ajustes para mayor naturalidad/similitud con la referencia.
            # Más iteraciones y menor temperatura sacrifican velocidad por calidad.
            # Parámetros soportados por XTTS v2 para mayor similitud.
            # (Quitamos decoder_iterations/gpt_cond_len que no son aceptados aquí.)
            tts_kwargs = dict(
                temperature=0.75, # Un poco más alto para más expresividad
                length_penalty=1.0,
                repetition_penalty=2.0, # Aumentado drásticamente para evitar tartamudeo/bucles
                top_k=50,
                top_p=0.85,
                speed=1.0,
                split_sentences=True, # Crucial para pausas naturales en textos largos
            )

            def _run():
                if speaker_wav:
                    self.model.tts_to_file(
                        text=text, 
                        file_path=str(output_path), 
                        speaker_wav=speaker_wav, 
                        language=language,
                        **tts_kwargs
                    )
                elif fallback_speaker:
                    self.model.tts_to_file(
                        text=text,
                        file_path=str(output_path),
                        speaker=fallback_speaker,
                        language=language,
                        **tts_kwargs
                    )
                else:
                    # Este branch no debería darse por el return anterior, pero lo dejamos por seguridad
                    raise ValueError("Se requiere un speaker_wav o speaker interno para XTTS.")

            await asyncio.to_thread(_run)
            
            if output_path.exists():
                with open(output_path, "rb") as f:
                    data = f.read()
                output_path.unlink()
                return data
            return None
            
        except Exception as e:
            logger.error(f"❌ Error síntesis: {e}")
            return None

    def reload(self):
        self.model = None