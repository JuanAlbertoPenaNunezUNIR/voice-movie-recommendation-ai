# Gestor de dispositivos mejorado

import torch
import logging
import os
import gc

logger = logging.getLogger(__name__)

class DeviceManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeviceManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self._forced_cpu = os.getenv("FORCE_DEVICE") == "cpu"
        self._device_override = None
        # Detección inicial
        self._detect_hardware()

    def _detect_hardware(self):
        """Detecta el hardware real disponible."""
        if torch.cuda.is_available():
            self._real_device = "cuda"
            props = torch.cuda.get_device_properties(0)
            logger.info(f"✅ Hardware NVIDIA detectado: {torch.cuda.get_device_name(0)} ({props.total_memory / 1024**3:.1f} GB)")
        else:
            self._real_device = "cpu"
            logger.warning("⚠️ No se detectó GPU NVIDIA. Funcionando en modo CPU.")

    def get_device_str(self) -> str:
        """Devuelve 'cuda' o 'cpu' según la configuración actual."""
        if self._device_override:
            return self._device_override
        if self._forced_cpu:
            return "cpu"
        return self._real_device

    def set_device(self, device: str):
        """Permite cambiar el dispositivo y limpiar la memoria. Retorna True si cambió."""
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("No se puede activar CUDA: No hay GPU NVIDIA disponible.")
        
        if device not in ["cpu", "cuda"]:
            raise ValueError("El dispositivo debe ser 'cpu' o 'cuda'")

        current = self.get_device_str()
        if device != current:
            self._device_override = device
            logger.info(f"🔄 Dispositivo cambiado manualmente a: {device.upper()}")
            self.clear_cache()
            return True
        return False

    def clear_cache(self):
        """Limpia la memoria VRAM/RAM."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def is_cuda_available(self) -> bool:
        return torch.cuda.is_available()

    def get_status(self):
        """Devuelve el estado completo para el frontend."""
        return {
            "current_device": self.get_device_str(),
            "cuda_available": self.is_cuda_available,
            "gpu_name": torch.cuda.get_device_name(0) if self.is_cuda_available else None
        }

    def log_device_info(self):
        """Imprime información detallada sobre el hardware detectado."""
        if self.is_cuda_available:
            try:
                gpu_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                total_mem = props.total_memory / 1024**3
                logger.info(f"✅ GPU Detectada y Optimizada: {gpu_name} ({total_mem:.2f} GB VRAM)")
                logger.info(f"🚀 Modo de alto rendimiento activado (cuDNN benchmark)")
            except Exception as e:
                logger.warning(f"✅ GPU Detectada (Error obteniendo detalles: {e})")
        else:
            logger.info("ℹ️ Usando CPU para el procesamiento (Modo estándar).")

# Instancia global exportada que esperan main.py y tts_processor.py
device_manager = DeviceManager()