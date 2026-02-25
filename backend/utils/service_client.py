# Cliente HTTP reutilizable para comunicación entre microservicios
# Patrón Singleton para reutilizar conexiones HTTP

import httpx
import logging
from typing import Optional, Dict, Any
import asyncio
from functools import lru_cache

logger = logging.getLogger(__name__)


class ServiceClient:
    """
    Cliente HTTP asíncrono reutilizable para comunicación entre microservicios.
    Implementa patrón Singleton y pool de conexiones.
    """
    
    _instance: Optional['ServiceClient'] = None
    _client: Optional[httpx.AsyncClient] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ServiceClient, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._client is None:
            # Timeout configurable para diferentes tipos de operaciones
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=300.0,  # Modelos de IA pueden tardar
                    write=10.0,
                    pool=5.0
                ),
                limits=httpx.Limits(
                    max_keepalive_connections=10,
                    max_connections=20
                ),
                follow_redirects=True
            )
            logger.info("✅ ServiceClient inicializado")
    
    async def get(self, url: str, **kwargs) -> Dict[str, Any]:
        """GET request con manejo de errores."""
        try:
            response = await self._client.get(url, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ Error HTTP {e.response.status_code} en GET {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Error en GET {url}: {e}")
            raise
    
    async def post(self, url: str, json: Optional[Dict] = None, data: Optional[Dict] = None, 
                   files: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        """POST request con manejo de errores."""
        try:
            if files:
                response = await self._client.post(url, files=files, data=data, **kwargs)
            else:
                response = await self._client.post(url, json=json, data=data, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ Error HTTP {e.response.status_code} en POST {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Error en POST {url}: {e}")
            raise
    
    async def close(self):
        """Cerrar cliente HTTP."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("🔌 ServiceClient cerrado")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Instancia global
service_client = ServiceClient()

