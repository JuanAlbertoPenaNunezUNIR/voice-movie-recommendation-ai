from typing import List, Dict, Any
import logging
from datetime import datetime
from semantic.synopsis_matcher import SynopsisMatcher
from utils.device_manager import device_manager

class RecommendationService:
    """
    Orquestador de Recomendación Híbrida (Hybrid Retrieval & Re-ranking).
    Combina filtrado de metadatos (TMDB) con similitud semántica (SBERT).
    """

    def __init__(self, recommendation_engine, tmdb_service):
        self.recommendation_engine = recommendation_engine
        self.tmdb_service = tmdb_service
        self.semantic_matcher = SynopsisMatcher()
        self.semantic_matcher.initialize()
        self.logger = logging.getLogger(__name__)

    def reload_model(self):
        """Fuerza la recarga del modelo semántico en el nuevo hardware."""
        self.semantic_matcher.initialize()

    def _filter_results(self, movies: List[Dict], filters: Dict) -> List[Dict]:
        """Aplica filtros duros en memoria para limpiar resultados de búsqueda textual/híbrida."""
        filtered = []
        
        # Preparar filtros
        f_genre_ids = filters.get("genre_ids", [])
        f_year_min = filters.get("year_min")
        f_year_max = filters.get("year_max")
        f_director = filters.get("director")
        
        for m in movies:
            # 1. Filtro de Año
            if f_year_min or f_year_max:
                try:
                    y = int(m.get("release_year", 0))
                    if f_year_min and y < int(f_year_min): continue
                    if f_year_max and y > int(f_year_max): continue
                except:
                    continue # Si no tiene año y pedimos año, fuera.

            # 2. Filtro de Género por ID (STRICT match: si la peli tiene TODOS los géneros pedidos)
            if f_genre_ids:
                m_genre_ids = m.get("genre_ids", [])
                if not m_genre_ids: continue
                if not all(gid in m_genre_ids for gid in f_genre_ids):
                    continue

            # 3. Filtro de Director (Si tenemos el dato en la respuesta de TMDB)
            if f_director:
                m_directors = [d.lower() for d in m.get("directors", [])]
                if not any(f_director.lower() in d for d in m_directors):
                    continue

            filtered.append(m)
        return filtered

    async def get_enriched_recommendations(
        self,
        user_id: str,
        preferences: Dict,
        conversation_context: Dict = None,
        limit: int = 5,
        user_query: str = ""
    ) -> Dict[str, Any]:

        try:
            # ==========================
            # 🧠 ADAPTACIÓN LLM → ENGINE
            # ==========================
            genres = preferences.get("genres", [])
            director = preferences.get("director")

            # Adaptar rango de años si viene del LLM
            years = None
            if preferences.get("year_min") or preferences.get("year_max"):
                years = [
                    preferences.get("year_min"),
                    preferences.get("year_max")
                ]

            actors = preferences.get("actors", [])
            keywords = preferences.get("keywords", [])
            limit = preferences.get("limit", limit)
            sort_by = preferences.get("sort_by", "popularity.desc")

            # Query semántica prioritaria
            semantic_query = (
                preferences.get("similar_to")
                or user_query
            )

            # Detección más flexible de intención de "calidad"
            q_lower = semantic_query.lower()
            is_best_query = any(k in q_lower for k in ["mejor", "best", "top", "maestra", "buena", "historia"]) or (sort_by == "vote_average.desc")

            # ==========================
            # 1️⃣ RETRIEVAL
            # ==========================
            retrieval_limit = max(40, limit * 8)
            movies = []

            # Comprobamos si hay filtros específicos que TMDB pueda manejar directamente
            has_specific_filters = (
                genres or 
                actors or 
                keywords or 
                years or 
                director or
                (sort_by and sort_by != "popularity.desc")
            )

            # Solo hacemos búsqueda semántica abierta (traer populares y re-rankear)
            # si NO tenemos filtros duros que acoten la búsqueda.
            # open_semantic_search = bool(semantic_query) and not has_specific_filters # YA NO SE USA ESTA BANDERA EXCLUSIVA

            match_type = "exact"

            # --- ESTRATEGIA HÍBRIDA DE RECUPERACIÓN ---
            
            # A. Búsqueda Textual (para encontrar la película de referencia o por título)
            text_movies = []
            should_text_search = bool(preferences.get("similar_to")) or not has_specific_filters
            
            if should_text_search and semantic_query:
                text_results_data = await self.tmdb_service.search_movies_by_query(semantic_query)
                text_movies = text_results_data.get("results", [])

            # B. Búsqueda por Similitud (TMDB Similar API) - ¡NUEVO Y PRIORITARIO!
            # Si el usuario pidió "tipo X" y encontramos X, buscamos sus similares.
            similar_movies = []
            if preferences.get("similar_to") and text_movies:
                # 1. Usar el ID de la película de referencia encontrada en la búsqueda textual
                ref_movie_id = text_movies[0].get("tmdb_id")
                ref_movie_title_found = text_movies[0].get("title")
                
                # 2. Si tenemos ID, buscar similares
                if ref_movie_id:
                    self.logger.info(f"🔎 Buscando películas similares a '{ref_movie_title_found}' (ID: {ref_movie_id})")
                    similar_data = await self.tmdb_service.get_similar_movies(ref_movie_id)
                    similar_movies = similar_data.get("results", [])

            # C. Búsqueda por Filtros/Descubrimiento (Discover API)
            discovery_movies = []
            if has_specific_filters:
                # Si hay filtros explícitos (género, año, actor...), los respetamos
                data = await self.tmdb_service.search_movies_advanced(
                    genres=genres, actors=actors, directors=[director] if director else None, years=years, keywords=keywords, sort_by=sort_by
                )
                discovery_movies = data.get("results", [])
            else:
                # Si es búsqueda abierta, traemos populares/mejores valoradas para rellenar
                for page in range(1, 3):
                    data = await self.tmdb_service.search_movies_advanced(
                        page=page, sort_by=sort_by
                    )
                    discovery_movies.extend(data.get("results", []))
                    if len(discovery_movies) >= retrieval_limit:
                        break

            # D. FUSIÓN INTELIGENTE
            # Si el usuario busca "Mejores películas" (is_best_query), confiamos más en Discovery (ordenado por nota).
            # Si busca algo específico ("Payasos asesinos"), confiamos más en Text Search.
            # Si busca "tipo X", la máxima prioridad son los similares.
            
            final_list = []
            seen_ids = set()

            # Definir orden de prioridad
            sources = [similar_movies, text_movies, discovery_movies] if not is_best_query else [discovery_movies, similar_movies, text_movies]
            
            for source in sources:
                for m in source:
                    if m["tmdb_id"] not in seen_ids:
                        final_list.append(m)
                        seen_ids.add(m["tmdb_id"])

            # --- FILTRADO ESTRICTO ---
            # Aplicamos los filtros duros a la lista combinada para eliminar ruido
            # (ej: Blade Runner 2049 en búsqueda de los 80, o documentales en búsqueda de director)
            # Convertimos nombres de género a IDs para un filtrado robusto e independiente del idioma.
            genre_map_en = await self.tmdb_service.get_genres_list(lang="en-US")
            genre_ids_to_filter = [genre_map_en.get(g.lower()) for g in preferences.get("genres", []) if genre_map_en.get(g.lower())]
            filters_by_id = preferences.copy()
            filters_by_id["genre_ids"] = [gid for gid in genre_ids_to_filter if gid is not None]

            movies = self._filter_results(final_list, filters_by_id)

            if text_movies:
                match_type = "exact"
            elif discovery_movies:
                match_type = "partial" if not has_specific_filters else "exact"
            else:
                match_type = "none"

            if not movies:
                # Estrategia de Fallback Inteligente
                if has_specific_filters and (years or sort_by == "vote_average.desc"):
                    self.logger.info("⚠️ Fallback: Relajando filtros temporales/ordenación para encontrar resultados de la entidad.")
                    # Si falló "Brad Pitt + 2024 + Mejores", probamos "Brad Pitt" a secas (por popularidad)
                    data = await self.tmdb_service.search_movies_advanced(
                        genres=genres,
                        actors=actors,
                        directors=[director] if director else None,
                        keywords=keywords,
                        sort_by="popularity.desc" # Quitamos years y forzamos popularidad
                    )
                    movies = data.get("results", [])
                    if movies:
                        match_type = "partial"

            # ==========================
            # 2️⃣ RE-RANKING
            # ==========================
            if semantic_query and movies:
                # Si es una búsqueda explícita de "mejores" (Top X), el criterio principal debe ser la nota (TMDB),
                # no la similitud semántica con la frase "mejores películas".
                if is_best_query:
                    self.logger.info("🏆 Detectado 'Top/Mejores': Priorizando orden por nota (Vote Average).")
                    # Ordenamos puramente por nota para respetar el ranking histórico (El Padrino, Cadena Perpetua...)
                    movies.sort(key=lambda x: x.get("vote_average", 0), reverse=True)
                else:
                    self.logger.info("🧠 Re-ranking híbrido semántico.")
                    movies = await self.semantic_matcher.calculate_scores(semantic_query, movies)
                    
                    alpha = 0.75
                    beta = 0.25
                    
                    for movie in movies:
                        sem = movie.get("_semantic_score", 0)
                        vote = min((movie.get("vote_average") or 0) / 10, 1)
                        movie["_final_score"] = (sem * alpha) + (vote * beta)

                    movies.sort(key=lambda x: x["_final_score"], reverse=True)

            elif genres:
                movies.sort(
                    key=lambda x: x.get("vote_average", 0),
                    reverse=True
                )

            # --- FILTRO FINAL DE SEGURIDAD ---
            # A pesar de los filtros anteriores, como salvaguarda final,
            # nos aseguramos de no devolver películas futuras.
            today_year = datetime.now().year
            final_movies = []
            for m in movies:
                try:
                    release_year = int(m.get("release_year", "0"))
                    if release_year > 0 and release_year <= today_year:
                        final_movies.append(m)
                except (ValueError, TypeError):
                    continue # Ignorar películas con año inválido

            return {
                "results": final_movies[:limit],
                "match_type": match_type
            }

        except Exception as e:
            self.logger.error(
                f"❌ Error crítico en recomendación: {e}",
                exc_info=True
            )
            return {"results": [], "match_type": "none"}
