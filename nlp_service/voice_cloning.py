# Módulo de clonación de voz usando Coqui TTS XTTS v2
# Movido al servicio NLP como sugiere el usuario

import asyncio
from pathlib import Path
import logging
import sys
import base64
import json
import os
import tempfile
from typing import Optional, List, Dict
import torch
from utils.device_manager import DeviceManager

logger = logging.getLogger(__name__)

class VoiceCloningService:
    """Servicio de clonación de voz usando Coqui TTS XTTS v2."""
    
    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2", device: str = None):
        self.model_name = model_name
        self.model = None
        self.logger = logger
        self.temp_dir = Path(tempfile.gettempdir()) / "tts_audio"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Directorio para almacenar clones de voz
        # Usar ruta absoluta para evitar problemas con rutas relativas
        base_dir = Path(__file__).parent.absolute()
        self.clones_dir = base_dir / "voice_clones"
        self.clones_dir.mkdir(parents=True, exist_ok=True)
        
        device_manager = DeviceManager()
        
        # Detectar dispositivo
        if device is None:
            self.device = device_manager.get_device_str()
        else:
            self.device = device
        
        self.logger.info(f"🎤 VoiceCloningService inicializado en {self.device}")

    async def initialize(self):
        """Inicializar el modelo TTS de forma asíncrona."""
        if self.model:
            return
        
        try:
            self.logger.info(f"🎯 Cargando modelo TTS: {self.model_name} en {self.device}")
            
            def _load():
                from TTS.api import TTS
                tts = TTS(self.model_name)
                return tts.to(self.device)
            
            self.model = await asyncio.to_thread(_load)
            self.logger.info("✅ Modelo TTS cargado correctamente")
            
        except Exception as e:
            self.logger.error(f"❌ Error cargando modelo TTS: {e}")
            raise

    def is_available(self) -> bool:
        """Verificar si el servicio está disponible."""
        return self.model is not None

    def get_available_clones(self) -> List[Dict[str, str]]:
        """Obtener lista de voces clonadas disponibles."""
        if not self.clones_dir.exists():
            return []
        
        voices = []
        for clone_dir in self.clones_dir.iterdir():
            if clone_dir.is_dir():
                # Buscar archivo de referencia
                ref_file = next(clone_dir.glob("reference.*"), None)
                if ref_file:
                    clone_id = clone_dir.name
                    # Intentar leer metadata
                    metadata_file = clone_dir / "metadata.json"
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, "r", encoding="utf-8") as f:
                                metadata = json.load(f)
                                voice_name = metadata.get("name", clone_id)
                        except:
                            voice_name = clone_id
                    else:
                        voice_name = clone_id
                    
                    voices.append({"id": clone_id, "name": voice_name})
        
        return sorted(voices, key=lambda x: x['name'])

    async def clone_voice(
        self, 
        user_id: str, 
        voice_name: str, 
        reference_audios: List[bytes]
    ) -> str:
        """
        Clonar voz del usuario usando audios de referencia.
        
        Args:
            user_id: ID único del usuario
            voice_name: Nombre descriptivo de la voz
            reference_audios: Lista de audios de referencia en bytes
            
        Returns:
            Ruta al directorio del clone de voz
        """
        try:
            if not self.model:
                await self.initialize()
            
            # Crear directorio para este clone
            clone_dir = self.clones_dir / user_id
            clone_dir.mkdir(parents=True, exist_ok=True)
            
            # Guardar metadatos
            metadata = {
                "id": user_id,
                "name": voice_name,
                "created_at": asyncio.get_event_loop().time(),
                "reference_count": len(reference_audios)
            }
            
            with open(clone_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            
            # Guardar audio de referencia (usar el primero si hay múltiples)
            if reference_audios:
                ref_path = clone_dir / "reference.wav"
                with open(ref_path, "wb") as f:
                    f.write(reference_audios[0])
                
                self.logger.info(f"✅ Voz clonada guardada: {voice_name} (ID: {user_id})")
                return str(clone_dir)
            else:
                raise ValueError("No se proporcionaron audios de referencia")
                
        except Exception as e:
            self.logger.error(f"❌ Error clonando voz: {e}")
            raise

    async def synthesize(
        self, 
        text: str, 
        speaker_wav_path: Optional[str] = None,
        language: str = "es"
    ) -> bytes:
        """
        Sintetizar texto a voz.
        
        Args:
            text: Texto a sintetizar
            speaker_wav_path: Ruta al archivo WAV del speaker (para clonación)
            language: Idioma del texto
            
        Returns:
            Audio en bytes (formato WAV)
        """
        if not self.model:
            await self.initialize()
        
        # Crear archivo temporal para el output
        output_file = self.temp_dir / f"{os.getpid()}_{hash(text)}.wav"
        
        try:
            def _run():
                if "xtts" in self.model_name.lower() and speaker_wav_path:
                    # Usar voz clonada
                    self.model.tts_to_file(
                        text=text,
                        file_path=str(output_file),
                        speaker_wav=speaker_wav_path,
                        language=language
                    )
                else:
                    # Usar voz por defecto
                    self.model.tts_to_file(
                        text=text,
                        file_path=str(output_file),
                        language=language
                    )
            
            await asyncio.to_thread(_run)
            
            # Leer el archivo generado
            with open(output_file, "rb") as f:
                audio_data = f.read()
            
            # Limpiar archivo temporal
            output_file.unlink(missing_ok=True)
            
            return audio_data
            
        except Exception as e:
            self.logger.error(f"❌ Error en síntesis: {e}")
            if output_file.exists():
                output_file.unlink(missing_ok=True)
            raise

    async def synthesize_with_cloned_voice(
        self, 
        text: str, 
        user_id: str,
        language: str = "es"
    ) -> bytes:
        """
        Sintetizar texto usando una voz clonada específica.
        
        Args:
            text: Texto a sintetizar
            user_id: ID del usuario (clave del clone)
            language: Idioma del texto
            
        Returns:
            Audio en bytes
        """
        clone_dir = self.clones_dir / user_id
        if not clone_dir.exists():
            raise ValueError(f"Voz clonada no encontrada para usuario: {user_id}")
        
        # Buscar archivo de referencia
        ref_file = next(clone_dir.glob("reference.*"), None)
        if not ref_file:
            raise ValueError(f"Archivo de referencia no encontrado para usuario: {user_id}")
        
        return await self.synthesize(text, str(ref_file), language)

# Instancia global del servicio
voice_cloning_service = VoiceCloningService()

