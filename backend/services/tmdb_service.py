# Servicio de integración con The Movie Database (TMDB) para obtener datos de películas

import os
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
import aiohttp
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class TMDBService:
    """
    Servicio para obtener datos de películas desde TMDB.
    """

    def __init__(self):
        self.api_key = os.getenv("TMDB_API_KEY")
        self.base_url = "https://api.themoviedb.org/3"
        self.image_base_url = "https://image.tmdb.org/t/p"
        self.session: Optional[aiohttp.ClientSession] = None
        self.request_timeout = 10
        self.cache = None # Cliente Redis

    async def initialize(self):
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.request_timeout))

    async def close(self):
        if self.session:
            await self.session.close()
            
    def set_cache(self, redis_client):
        self.cache = redis_client

    async def get_person_id(self, name: str) -> Optional[int]:
        await self.initialize()
        url = f"{self.base_url}/search/person"
        params = {"api_key": self.api_key, "query": name, "language": "es-ES"}
        async with self.session.get(url, params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                results = data.get("results", [])
                if results:
                    return results[0]["id"]
        return None

    async def get_keyword_id(self, query: str) -> Optional[int]:
        """Busca el ID de una palabra clave (ej: 'space')."""
        await self.initialize()
        url = f"{self.base_url}/search/keyword"
        params = {"api_key": self.api_key, "query": query, "page": 1}
        try:
            async with self.session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results = data.get("results", [])
                    if results:
                        return results[0]["id"]
        except Exception as e:
            logger.warning(f"Error buscando keyword '{query}': {e}")
        return None

    def _parse_years(self, years: List[str]) -> Tuple[Optional[str], Optional[str]]:
        """Convierte entradas de años/décadas en fechas válidas para TMDB."""
        if not years:
            return None, None
        
        start_year = None
        end_year = None
        
        # Aplanar y limpiar
        valid_years = []
        is_retro = False

        for y in years:
            y_str = str(y).lower().strip()
            
            # Detectar conceptos abstractos
            if y_str in ["antiguas", "antigua", "clásicas", "clasicas", "old", "retro"]:
                is_retro = True
                continue
            
            # Detectar décadas (1980s, 80s)
            if "s" in y_str:
                decade = y_str.replace("s", "")
                if len(decade) == 2: decade = "19" + decade # Asumir siglo XX para 80s, 90s
                if decade.isdigit():
                    valid_years.append(int(decade))
                    valid_years.append(int(decade) + 9) # Final de la década
            # Años exactos
            elif y_str.isdigit():
                valid_years.append(int(y_str))

        if is_retro and not valid_years:
            # Si solo dijo "antiguas", definimos un rango arbitrario hasta el año 2000
            return "1900-01-01", "1999-12-31"

        if valid_years:
            valid_years.sort()
            start_year = f"{valid_years[0]}-01-01"
            end_year = f"{valid_years[-1]}-12-31"
        
        return start_year, end_year

    async def search_movies_by_query(self, query: str) -> Dict[str, Any]:
        """Búsqueda textual directa para encontrar títulos o tramas específicas."""
        await self.initialize()
        
        # Limpieza básica de la query para quitar "ruido" conversacional
        clean_query = query.lower()
        stop_phrases = ["recomiéndame", "recomienda", "quiero ver", "una película de", "películas de", "algo de", "sobre", "que trate de", "encontrame", "encuéntrame", "de la historia", "las mejores", "mejores", "dime", "una película", "con", "película", "una", "top 3", "top 5", "top 10"]
        for phrase in stop_phrases:
            clean_query = clean_query.replace(phrase, "")
        clean_query = clean_query.strip()

        if len(clean_query) < 2:
            return {"results": []}

        # --- CACHE CHECK ---
        cache_key = f"tmdb:search:{clean_query}"
        if self.cache:
            cached = await self.cache.get(cache_key)
            if cached:
                logger.info(f"⚡ Cache hit: {clean_query}")
                return json.loads(cached)

        url = f"{self.base_url}/search/movie"
        params = {"api_key": self.api_key, "language": "es-ES", "query": clean_query, "page": 1, "include_adult": "false"}
        
        async with self.session.get(url, params=params) as resp:
            if resp.status != 200: return {"results": []}
            data = await resp.json()
            
        # Filtrado estricto: Nada de futuras ni basura sin votos
        today = datetime.now().strftime("%Y-%m-%d")
        valid_results = []
        for m in data.get("results", []):
            r_date = m.get("release_date")
            if not r_date or r_date > today: continue # Filtrado estricto: Si no tiene fecha o es futura, fuera.
            if m.get("vote_count", 0) < 2: continue
            valid_results.append(m)

        enriched = []
        for movie in valid_results[:10]:
            detail = await self.get_movie_details(movie["id"])
            if detail: enriched.append(detail)
            
        result = {"results": enriched}
        
        # --- CACHE SET (24 horas) ---
        if self.cache and enriched:
            await self.cache.setex(cache_key, 86400, json.dumps(result))
            
        return result

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def search_movies_advanced(
        self,
        genres: List[str] = None,
        actors: List[str] = None,
        directors: List[str] = None,
        writers: List[str] = None,
        years: List[str] = None,
        keywords: List[str] = None,
        sort_by: str = "popularity.desc",
        page: int = 1
    ) -> Dict[str, Any]:

        await self.initialize()

        params = {
            "api_key": self.api_key,
            "language": "es-ES",
            "include_adult": "false",
            "page": page,
            # No mostrar películas futuras
            "primary_release_date.lte": datetime.now().strftime("%Y-%m-%d"),
        }

        # Lógica de Ordenación Inteligente
        # Si piden antiguas o clásicas, es mejor ordenar por nota que por popularidad actual
        # Prioridad: 1. LLM explícito, 2. Retro logic, 3. Default
        final_sort_by = "popularity.desc"
        if sort_by and sort_by != "popularity.desc":
            final_sort_by = sort_by
        else:
            is_retro_search = years and any(y in ["antiguas", "clásicas"] for y in years)
            if is_retro_search:
                final_sort_by = "vote_average.desc"
        
        params["sort_by"] = final_sort_by

        # Ajuste dinámico de votos mínimos: Si buscamos "las mejores", subimos el umbral
        # para evitar películas desconocidas con pocas valoraciones (ej: Gabriel's Inferno).
        if "vote_average" in final_sort_by:
            # Si filtramos por actor/director, relajamos el umbral de votos para no ser tan estrictos
            if actors or directors:
                params["vote_count.gte"] = 100
            elif genres or years:
                # Si hay géneros o años específicos, umbral intermedio para no ser tan restrictivos
                params["vote_count.gte"] = 300
            else:
                # "Mejores de la historia" genérico -> Umbral MUY ALTO para garantizar clásicos universales
                # Bajamos a 1000 para permitir combinaciones de géneros (ej: Sci-Fi + Terror) que tengan clásicos con menos votos masivos
                params["vote_count.gte"] = 1000
        else:
            params["vote_count.gte"] = 200

        # Géneros
        if genres:
            genre_list = await self.get_genres_list()
            genre_ids = [str(genre_list[g.lower()]) for g in genres if g.lower() in genre_list]
            if genre_ids:
                params["with_genres"] = ",".join(genre_ids)

        # Años
        start_date, end_date = self._parse_years(years)
        if start_date: params["primary_release_date.gte"] = start_date
        
        # Lógica de fecha tope: Nunca mostrar futuras (Fix Zootrópolis 2)
        today = datetime.now().strftime("%Y-%m-%d")
        if end_date:
            params["primary_release_date.lte"] = min(end_date, today)
        else:
            params["primary_release_date.lte"] = today
            
        # Actores
        if actors:
            actor_ids = [str(await self.get_person_id(a)) for a in actors if await self.get_person_id(a)]
            if actor_ids:
                params["with_cast"] = ",".join(actor_ids)

        crew_ids = []
        if directors:
            crew_ids += [str(await self.get_person_id(d)) for d in directors if await self.get_person_id(d)]
        if writers:
            crew_ids += [str(await self.get_person_id(w)) for w in writers if await self.get_person_id(w)]
        if crew_ids:
            params["with_crew"] = ",".join(crew_ids)

        # Palabras clave
        if keywords:
            # Intentar buscar IDs para keywords (limitado a los 3 primeros para no saturar)
            k_ids = []
            for k in keywords[:3]:
                # Traducir conceptos comunes si es necesario, o confiar en que TMDB tiene keywords en español/inglés
                k_id = await self.get_keyword_id(k) 
                if k_id: k_ids.append(str(k_id))

            if k_ids:
                # Usamos OR (|) para ser más flexibles, o AND (,) si queremos precisión
                params["with_keywords"] = "|".join(k_ids)

        # --- CACHE CHECK (Advanced) ---
        # Creamos una key única basada en los parámetros ordenados
        param_str = json.dumps(params, sort_keys=True)
        cache_key = f"tmdb:discover:{hash(param_str)}"
        if self.cache:
            cached = await self.cache.get(cache_key)
            if cached: return json.loads(cached)

        logger.info(f"🔍 TMDB Discover Params: {params}")
        discover_url = f"{self.base_url}/discover/movie"

        async with self.session.get(discover_url, params=params) as resp:
            if resp.status != 200:
                logger.error(f"Error TMDB: {resp.status}")
                return {"results": []}
            data = await resp.json()

        # Enriquecer resultados
        results = data.get("results", [])
        enriched = []
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Traemos un poco más de detalle para el ranking semántico posterior
        for movie in results[:15]: 
            # Filtro manual de seguridad: Eliminar películas futuras (Fix Zootrópolis 2, Avatar, etc.)
            r_date = movie.get("release_date")
            if not r_date or r_date > today:
                continue
                
            detail = await self.get_movie_details(movie["id"])
            if detail:
                enriched.append(detail)
        
        result = {"results": enriched}
        
        # --- CACHE SET (6 horas para discovery) ---
        if self.cache and enriched:
            await self.cache.setex(cache_key, 21600, json.dumps(result))
            
        return result

    async def get_movie_details(self, movie_id: int) -> Optional[Dict[str, Any]]:
        await self.initialize()
        url = f"{self.base_url}/movie/{movie_id}"
        params = {"api_key": self.api_key, "language": "es-ES", "append_to_response": "credits,keywords"}
        async with self.session.get(url, params=params) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            return self._enrich_movie_data(data)

    def _enrich_movie_data(self, movie_data: Dict) -> Dict[str, Any]:
        enriched = movie_data.copy()
        enriched["tmdb_id"] = movie_data.get("id")
        enriched["title"] = movie_data.get("title", "Título no disponible")
        enriched["overview"] = movie_data.get("overview", "Sin sinopsis disponible.")
        release_date = movie_data.get("release_date", "")
        enriched["release_year"] = release_date[:4] if release_date else "N/A"
        
        # Construir Poster URL
        if movie_data.get("poster_path"):
            enriched["poster_url"] = f"{self.image_base_url}/w500{movie_data['poster_path']}"
        else:
            enriched["poster_url"] = "https://via.placeholder.com/500x750?text=No+Poster"

        # Info extra
        enriched["vote_average"] = movie_data.get("vote_average", 0)
        enriched["popularity"] = movie_data.get("popularity", 0)

        # Actores principales
        credits = movie_data.get("credits", {})
        cast = credits.get("cast", [])
        enriched["main_cast"] = [{"name": c["name"]} for c in cast[:3]]
        
        # Director (Nuevo)
        crew = credits.get("crew", [])
        enriched["directors"] = [c["name"] for c in crew if c["job"] == "Director"][:1]
        
        # Géneros (Nuevo - nombres reales)
        enriched["genres"] = [g["name"] for g in movie_data.get("genres", [])]
        
        return enriched

    async def get_genres_list(self) -> Dict[str, int]:
        await self.initialize()
        url = f"{self.base_url}/genre/movie/list"
        params = {"api_key": self.api_key, "language": "es-ES"}
        async with self.session.get(url, params=params) as resp:
            if resp.status != 200:
                return {}
            data = await resp.json()
            return {g["name"].lower(): g["id"] for g in data.get("genres", [])}