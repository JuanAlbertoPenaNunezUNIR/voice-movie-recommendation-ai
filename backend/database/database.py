# Gestor de base de datos completo con optimizaciones

import sqlite3
import json
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from contextlib import contextmanager
import pandas as pd
import numpy as np
from pathlib import Path

class DatabaseManager:
    """
    Gestor avanzado de base de datos con optimizaciones para el sistema de recomendación.
    Incluye caching, análisis de datos y funciones de mantenimiento.
    """
    
    def __init__(self, db_path: str = "recommendation.db"):
        """
        Inicializar gestor de base de datos con configuración avanzada.
        
        Args:
            db_path: Ruta al archivo de base de datos
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.cache = {}  # Cache en memoria para consultas frecuentes
        self.cache_ttl = 300  # 5 minutos en segundos
        self._init_database()
        
    def _init_database(self):
        """
        Inicializar esquema de base de datos con índices y optimizaciones.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Enable WAL mode for better concurrency
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA cache_size=-2000")  # 2MB cache
                
                # Tabla de usuarios con índices
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_interaction TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        total_interactions INTEGER DEFAULT 0,
                        preferences_json TEXT,
                        metadata_json TEXT,
                        user_segment TEXT DEFAULT 'new'
                    )
                ''')
                
                # Tabla de interacciones detallada
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS interactions (
                        interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        session_id TEXT,
                        interaction_type TEXT NOT NULL,
                        input_text TEXT,
                        transcription_text TEXT,
                        preferences_json TEXT,
                        recommendations_json TEXT,
                        feedback_json TEXT,
                        processing_time REAL,
                        confidence_score REAL,
                        error_message TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (user_id)
                    )
                ''')
                
                # Tabla de correcciones para aprendizaje
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS transcription_corrections (
                        correction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        session_id TEXT,
                        original_text TEXT NOT NULL,
                        corrected_text TEXT NOT NULL,
                        correction_type TEXT,
                        word_error_rate REAL,
                        pattern_analysis_json TEXT,
                        learned_lesson TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (user_id)
                    )
                ''')
                
                # Tabla de feedback de recomendaciones
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS recommendation_feedback (
                        feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        session_id TEXT,
                        movie_id INTEGER NOT NULL,
                        movie_title TEXT,
                        feedback_type TEXT NOT NULL,
                        rating INTEGER,
                        reason_text TEXT,
                        context_json TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (user_id)
                    )
                ''')
                
                # Tabla de sesiones de usuario
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        session_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        end_time TIMESTAMP,
                        total_interactions INTEGER DEFAULT 0,
                        session_duration REAL,
                        device_info TEXT,
                        location_info TEXT,
                        metadata_json TEXT,
                        FOREIGN KEY (user_id) REFERENCES users (user_id)
                    )
                ''')
                
                # Tabla de métricas del sistema
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        metric_type TEXT NOT NULL,
                        metric_subtype TEXT,
                        metric_value REAL NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata_json TEXT
                    )
                ''')
                
                # Tabla de modelos de IA
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS ai_models (
                        model_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        model_version TEXT,
                        training_date TIMESTAMP,
                        performance_metrics_json TEXT,
                        is_active BOOLEAN DEFAULT 1,
                        last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Crear índices para optimización
                self._create_indexes(cursor)
                
                conn.commit()
                self.logger.info("✅ Base de datos inicializada con esquema avanzado")
                
        except Exception as e:
            self.logger.error(f"❌ Error inicializando base de datos: {e}")
            raise
    
    def _create_indexes(self, cursor):
        """Crear índices para optimizar consultas frecuentes."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_interactions_user_timestamp ON interactions(user_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_interactions_type ON interactions(interaction_type)",
            "CREATE INDEX IF NOT EXISTS idx_feedback_user_movie ON recommendation_feedback(user_id, movie_id)",
            "CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON recommendation_feedback(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_corrections_user ON transcription_corrections(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_type_timestamp ON system_metrics(metric_type, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_user ON user_sessions(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_time ON user_sessions(start_time)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
    
    @contextmanager
    def _get_connection(self):
        """
        Context manager para manejar conexiones a la base de datos con retry.
        """
        conn = None
        retries = 3
        for attempt in range(retries):
            try:
                conn = sqlite3.connect(self.db_path, timeout=30)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA foreign_keys = ON")
                try:
                    yield conn
                    # Si llegamos aquí, la operación fue exitosa
                    conn.commit()
                    break
                except Exception as inner_e:
                    # Error dentro del bloque with, hacer rollback
                    try:
                        conn.rollback()
                    except:
                        pass
                    raise inner_e
                finally:
                    # Cerrar conexión al salir del bloque with
                    if conn:
                        conn.close()
                        conn = None
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < retries - 1:
                    self.logger.warning(f"⚠️ Base de datos bloqueada, reintento {attempt + 1}")
                    import time
                    time.sleep(0.1 * (attempt + 1))
                    if conn:
                        try:
                            conn.close()
                        except:
                            pass
                        conn = None
                else:
                    self.logger.error(f"❌ Error de conexión a BD: {e}")
                    if conn:
                        try:
                            conn.close()
                        except:
                            pass
                    raise
            except Exception as e:
                self.logger.error(f"❌ Error inesperado de conexión: {e}")
                if conn:
                    try:
                        conn.rollback()
                        conn.close()
                    except:
                        pass
                raise
    
    def _cache_get(self, key: str):
        """Obtener valor del cache con validación de TTL."""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if (datetime.now() - timestamp).seconds < self.cache_ttl:
                return value
            else:
                del self.cache[key]
        return None
    
    def _cache_set(self, key: str, value):
        """Guardar valor en cache."""
        self.cache[key] = (value, datetime.now())
    
    # ========== GESTIÓN DE USUARIOS ==========
    
    def create_or_update_user(self, user_id: str, metadata: Dict = None) -> bool:
        """
        Crear o actualizar usuario en el sistema.
        
        Args:
            user_id: Identificador único del usuario
            metadata: Metadatos adicionales del usuario
            
        Returns:
            bool: True si la operación fue exitosa
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
                
                cursor.execute('''
                    INSERT OR REPLACE INTO users 
                    (user_id, last_interaction, metadata_json)
                    VALUES (?, CURRENT_TIMESTAMP, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                    last_interaction = CURRENT_TIMESTAMP,
                    total_interactions = total_interactions + 1,
                    metadata_json = COALESCE(excluded.metadata_json, metadata_json)
                ''', (user_id, metadata_json))
                
                conn.commit()
                
                # Invalidar cache relacionado
                cache_key = f"user_prefs_{user_id}"
                if cache_key in self.cache:
                    del self.cache[cache_key]
                
                self.logger.debug(f"👤 Usuario {user_id} actualizado")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Error creando/actualizando usuario: {e}")
            return False
    
    def get_user(self, user_id: str) -> Optional[Dict]:
        """
        Obtener información completa de un usuario.
        
        Args:
            user_id: ID del usuario
            
        Returns:
            Optional[Dict]: Datos del usuario o None si no existe
        """
        cache_key = f"user_full_{user_id}"
        cached = self._cache_get(cache_key)
        if cached:
            return cached
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT user_id, created_at, last_interaction, total_interactions,
                           preferences_json, metadata_json, user_segment
                    FROM users 
                    WHERE user_id = ?
                ''', (user_id,))
                
                row = cursor.fetchone()
                if row:
                    user_data = dict(row)
                    # Parsear JSON fields
                    if user_data['preferences_json']:
                        user_data['preferences'] = json.loads(user_data['preferences_json'])
                    if user_data['metadata_json']:
                        user_data['metadata'] = json.loads(user_data['metadata_json'])
                    
                    # Remover campos JSON originales
                    user_data.pop('preferences_json', None)
                    user_data.pop('metadata_json', None)
                    
                    self._cache_set(cache_key, user_data)
                    return user_data
                
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error obteniendo usuario: {e}")
            return None
    
    def save_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """
        Guardar preferencias del usuario con versionado.
        
        Args:
            user_id: ID del usuario
            preferences: Diccionario de preferencias
            
        Returns:
            bool: True si se guardó correctamente
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Obtener preferencias anteriores para comparación
                cursor.execute(
                    'SELECT preferences_json FROM users WHERE user_id = ?',
                    (user_id,)
                )
                row = cursor.fetchone()
                
                previous_prefs = {}
                if row and row[0]:
                    previous_prefs = json.loads(row[0])
                
                # Añadir metadata a las preferencias
                enhanced_preferences = {
                    **preferences,
                    '_metadata': {
                        'last_updated': datetime.now().isoformat(),
                        'version': len(previous_prefs.get('_metadata', {}).get('history', [])) + 1,
                        'history': previous_prefs.get('_metadata', {}).get('history', []) + [{
                            'timestamp': datetime.now().isoformat(),
                            'changes': self._compare_preferences(previous_prefs, preferences)
                        }]
                    }
                }
                
                preferences_json = json.dumps(enhanced_preferences, ensure_ascii=False)
                
                cursor.execute('''
                    UPDATE users 
                    SET preferences_json = ?, last_interaction = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                ''', (preferences_json, user_id))
                
                # Registrar métrica de actualización
                self.log_system_metric(
                    'user_preferences_updated',
                    len(preferences),
                    {'user_id': user_id, 'change_count': len(enhanced_preferences['_metadata']['history'])}
                )
                
                conn.commit()
                
                # Invalidar cache
                cache_key = f"user_prefs_{user_id}"
                if cache_key in self.cache:
                    del self.cache[cache_key]
                
                self.logger.debug(f"💾 Preferencias guardadas para usuario {user_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Error guardando preferencias: {e}")
            return False
    
    def _compare_preferences(self, old_prefs: Dict, new_prefs: Dict) -> List[Dict]:
        """Comparar preferencias antiguas y nuevas para tracking de cambios."""
        changes = []
        
        # Comparar géneros
        old_genres = set(old_prefs.get('genres', []))
        new_genres = set(new_prefs.get('genres', []))
        
        if old_genres != new_genres:
            added = list(new_genres - old_genres)
            removed = list(old_genres - new_genres)
            if added or removed:
                changes.append({
                    'field': 'genres',
                    'added': added,
                    'removed': removed
                })
        
        # Podrían añadirse más comparaciones aquí
        
        return changes
    
    # ========== GESTIÓN DE INTERACCIONES ==========
    
    def log_interaction(self, interaction_data: Dict[str, Any]) -> int:
        """
        Registrar una interacción detallada del usuario.
        
        Args:
            interaction_data: Diccionario con datos de la interacción
            
        Returns:
            int: ID de la interacción registrada
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Preparar datos
                user_id = interaction_data.get('user_id', 'anonymous')
                session_id = interaction_data.get('session_id', f"session_{user_id}_{datetime.now().timestamp()}")
                
                # Crear usuario si no existe
                self.create_or_update_user(user_id)
                
                # Insertar interacción
                cursor.execute('''
                    INSERT INTO interactions 
                    (user_id, session_id, interaction_type, input_text, transcription_text,
                     preferences_json, recommendations_json, feedback_json,
                     processing_time, confidence_score, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_id,
                    session_id,
                    interaction_data.get('interaction_type', 'unknown'),
                    interaction_data.get('input_text'),
                    interaction_data.get('transcription_text'),
                    json.dumps(interaction_data.get('preferences'), ensure_ascii=False) if interaction_data.get('preferences') else None,
                    json.dumps(interaction_data.get('recommendations'), ensure_ascii=False) if interaction_data.get('recommendations') else None,
                    json.dumps(interaction_data.get('feedback'), ensure_ascii=False) if interaction_data.get('feedback') else None,
                    interaction_data.get('processing_time'),
                    interaction_data.get('confidence_score'),
                    interaction_data.get('error_message')
                ))
                
                interaction_id = cursor.lastrowid
                
                # Actualizar contador de interacciones del usuario
                cursor.execute('''
                    UPDATE users 
                    SET total_interactions = total_interactions + 1,
                        last_interaction = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                ''', (user_id,))
                
                # Actualizar o crear sesión
                self._update_user_session(user_id, session_id, cursor)
                
                conn.commit()
                
                # Registrar métrica
                self.log_system_metric(
                    'interaction_logged',
                    1,
                    {'interaction_type': interaction_data.get('interaction_type', 'unknown')}
                )
                
                self.logger.debug(f"📝 Interacción {interaction_id} registrada para usuario {user_id}")
                return interaction_id
                
        except Exception as e:
            self.logger.error(f"❌ Error registrando interacción: {e}")
            return -1
    
    def _update_user_session(self, user_id: str, session_id: str, cursor):
        """Actualizar o crear sesión de usuario."""
        # Verificar si la sesión existe
        cursor.execute(
            'SELECT session_id FROM user_sessions WHERE session_id = ?',
            (session_id,)
        )
        
        if cursor.fetchone():
            # Actualizar sesión existente
            cursor.execute('''
                UPDATE user_sessions 
                SET total_interactions = total_interactions + 1,
                    session_duration = julianday(CURRENT_TIMESTAMP) - julianday(start_time)
                WHERE session_id = ?
            ''', (session_id,))
        else:
            # Crear nueva sesión
            cursor.execute('''
                INSERT INTO user_sessions 
                (session_id, user_id, start_time)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (session_id, user_id))
    
    def get_user_interactions(self, user_id: str, limit: int = 50, 
                            interaction_type: str = None) -> List[Dict]:
        """
        Obtener historial de interacciones del usuario.
        
        Args:
            user_id: ID del usuario
            limit: Límite de interacciones a recuperar
            interaction_type: Filtrar por tipo de interacción (opcional)
            
        Returns:
            List[Dict]: Historial de interacciones
        """
        cache_key = f"interactions_{user_id}_{limit}_{interaction_type}"
        cached = self._cache_get(cache_key)
        if cached:
            return cached
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                query = '''
                    SELECT interaction_id, session_id, interaction_type, input_text,
                           transcription_text, preferences_json, recommendations_json,
                           feedback_json, processing_time, confidence_score,
                           error_message, timestamp
                    FROM interactions 
                    WHERE user_id = ?
                '''
                params = [user_id]
                
                if interaction_type:
                    query += " AND interaction_type = ?"
                    params.append(interaction_type)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                
                interactions = []
                for row in cursor.fetchall():
                    interaction = dict(row)
                    
                    # Parsear JSON fields
                    for field in ['preferences_json', 'recommendations_json', 'feedback_json']:
                        if interaction.get(field):
                            key = field.replace('_json', '')
                            interaction[key] = json.loads(interaction[field])
                            del interaction[field]
                    
                    interactions.append(interaction)
                
                self._cache_set(cache_key, interactions)
                return interactions
                
        except Exception as e:
            self.logger.error(f"❌ Error obteniendo interacciones: {e}")
            return []
    
    # ========== GESTIÓN DE FEEDBACK ==========
    
    def log_feedback(self, feedback_data: Dict[str, Any]) -> bool:
        """
        Registrar feedback del usuario sobre recomendaciones.
        
        Args:
            feedback_data: Datos del feedback
            
        Returns:
            bool: True si se registró correctamente
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO recommendation_feedback 
                    (user_id, session_id, movie_id, movie_title, feedback_type,
                     rating, reason_text, context_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    feedback_data['user_id'],
                    feedback_data.get('session_id'),
                    feedback_data['movie_id'],
                    feedback_data.get('movie_title'),
                    feedback_data['feedback_type'],
                    feedback_data.get('rating'),
                    feedback_data.get('reason_text'),
                    json.dumps(feedback_data.get('context'), ensure_ascii=False) if feedback_data.get('context') else None
                ))
                
                # Registrar métrica de feedback
                self.log_system_metric(
                    'feedback_received',
                    1,
                    {
                        'feedback_type': feedback_data['feedback_type'],
                        'user_id': feedback_data['user_id'],
                        'movie_id': feedback_data['movie_id']
                    }
                )
                
                conn.commit()
                
                # Invalidar cache de feedback del usuario
                cache_key = f"feedback_{feedback_data['user_id']}"
                if cache_key in self.cache:
                    del self.cache[cache_key]
                
                self.logger.debug(f"📊 Feedback registrado: usuario {feedback_data['user_id']}")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Error registrando feedback: {e}")
            return False
    
    def get_user_feedback(self, user_id: str, limit: int = 100) -> List[Dict]:
        """
        Obtener historial de feedback del usuario.
        
        Args:
            user_id: ID del usuario
            limit: Límite de feedback a recuperar
            
        Returns:
            List[Dict]: Historial de feedback
        """
        cache_key = f"user_feedback_{user_id}_{limit}"
        cached = self._cache_get(cache_key)
        if cached:
            return cached
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT movie_id, movie_title, feedback_type, rating, reason_text,
                           context_json, timestamp
                    FROM recommendation_feedback 
                    WHERE user_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (user_id, limit))
                
                feedback_history = []
                for row in cursor.fetchall():
                    feedback = dict(row)
                    if feedback.get('context_json'):
                        feedback['context'] = json.loads(feedback['context_json'])
                        del feedback['context_json']
                    
                    feedback_history.append(feedback)
                
                self._cache_set(cache_key, feedback_history)
                return feedback_history
                
        except Exception as e:
            self.logger.error(f"❌ Error obteniendo feedback: {e}")
            return []
    
    def get_feedback_analytics(self, user_id: str = None) -> Dict[str, Any]:
        """
        Obtener análisis de feedback para insights.
        
        Args:
            user_id: ID específico del usuario (opcional)
            
        Returns:
            Dict: Análisis de feedback
        """
        cache_key = f"feedback_analytics_{user_id if user_id else 'global'}"
        cached = self._cache_get(cache_key)
        if cached:
            return cached
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                analytics = {}
                
                if user_id:
                    # Análisis por usuario
                    cursor.execute('''
                        SELECT 
                            COUNT(*) as total_feedback,
                            SUM(CASE WHEN feedback_type = 'like' THEN 1 ELSE 0 END) as likes,
                            SUM(CASE WHEN feedback_type = 'dislike' THEN 1 ELSE 0 END) as dislikes,
                            AVG(rating) as avg_rating
                        FROM recommendation_feedback 
                        WHERE user_id = ?
                    ''', (user_id,))
                else:
                    # Análisis global
                    cursor.execute('''
                        SELECT 
                            COUNT(*) as total_feedback,
                            COUNT(DISTINCT user_id) as unique_users,
                            SUM(CASE WHEN feedback_type = 'like' THEN 1 ELSE 0 END) as likes,
                            SUM(CASE WHEN feedback_type = 'dislike' THEN 1 ELSE 0 END) as dislikes,
                            AVG(rating) as avg_rating
                        FROM recommendation_feedback
                    ''')
                
                row = cursor.fetchone()
                if row:
                    analytics = dict(row)
                
                # Distribución por tipo de feedback
                cursor.execute('''
                    SELECT feedback_type, COUNT(*) as count
                    FROM recommendation_feedback
                    GROUP BY feedback_type
                ''')
                
                analytics['distribution'] = dict(cursor.fetchall())
                
                # Top películas con mejor feedback
                cursor.execute('''
                    SELECT movie_id, movie_title, 
                           COUNT(*) as feedback_count,
                           SUM(CASE WHEN feedback_type = 'like' THEN 1 ELSE 0 END) as likes
                    FROM recommendation_feedback
                    WHERE movie_title IS NOT NULL
                    GROUP BY movie_id, movie_title
                    ORDER BY likes DESC
                    LIMIT 10
                ''')
                
                analytics['top_movies'] = [
                    dict(row) for row in cursor.fetchall()
                ]
                
                self._cache_set(cache_key, analytics)
                return analytics
                
        except Exception as e:
            self.logger.error(f"❌ Error obteniendo análisis de feedback: {e}")
            return {}
    
    # ========== GESTIÓN DE CORRECCIONES ==========
    
    def log_correction(self, correction_data: Dict[str, Any]) -> bool:
        """
        Registrar corrección de transcripción para aprendizaje.
        
        Args:
            correction_data: Datos de la corrección
            
        Returns:
            bool: True si se registró correctamente
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Calcular Word Error Rate (WER) aproximado
                original = correction_data.get('original_text', '')
                corrected = correction_data.get('corrected_text', '')
                wer = self._calculate_wer(original, corrected) if original and corrected else None
                
                cursor.execute('''
                    INSERT INTO transcription_corrections 
                    (user_id, session_id, original_text, corrected_text,
                     correction_type, word_error_rate, pattern_analysis_json,
                     learned_lesson)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    correction_data['user_id'],
                    correction_data.get('session_id'),
                    original,
                    corrected,
                    correction_data.get('correction_type', 'manual'),
                    wer,
                    json.dumps(correction_data.get('pattern_analysis'), ensure_ascii=False) if correction_data.get('pattern_analysis') else None,
                    correction_data.get('learned_lesson')
                ))
                
                # Registrar métrica
                self.log_system_metric(
                    'correction_logged',
                    1,
                    {
                        'correction_type': correction_data.get('correction_type', 'manual'),
                        'user_id': correction_data['user_id'],
                        'wer': wer
                    }
                )
                
                conn.commit()
                self.logger.debug(f"✏️ Corrección registrada para usuario {correction_data['user_id']}")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Error registrando corrección: {e}")
            return False
    
    def _calculate_wer(self, reference: str, hypothesis: str) -> float:
        """Calcular Word Error Rate aproximado."""
        if not reference or not hypothesis:
            return 0.0
        
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        
        if not ref_words:
            return 1.0 if hyp_words else 0.0
        
        # Distancia de Levenshtein simple a nivel de palabras
        distances = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
        
        for i in range(len(ref_words) + 1):
            distances[i][0] = i
        for j in range(len(hyp_words) + 1):
            distances[0][j] = j
        
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
                distances[i][j] = min(
                    distances[i-1][j] + 1,      # deletion
                    distances[i][j-1] + 1,      # insertion
                    distances[i-1][j-1] + cost  # substitution
                )
        
        wer = distances[len(ref_words)][len(hyp_words)] / len(ref_words)
        return round(wer, 3)
    
    def save_correction_analysis(self, user_id: str, correction_analysis: Dict[str, Any]) -> bool:
        """
        Guardar análisis de corrección para aprendizaje del sistema.
        
        Args:
            user_id: ID del usuario
            correction_analysis: Análisis de la corrección con patrones detectados
            
        Returns:
            bool: True si se guardó correctamente
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Asegurar que el usuario existe antes de insertar la corrección
                cursor.execute('SELECT user_id FROM users WHERE user_id = ?', (user_id,))
                if not cursor.fetchone():
                    # Crear usuario si no existe
                    cursor.execute('''
                        INSERT OR IGNORE INTO users 
                        (user_id, created_at, last_interaction, total_interactions, user_segment)
                        VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0, 'new')
                    ''', (user_id,))
                    self.logger.debug(f"👤 Usuario {user_id} creado automáticamente")
                
                # Preparar datos para inserción
                correction_type = correction_analysis.get('pattern', 'unknown')
                confidence = correction_analysis.get('confidence', 0.0)
                pattern_analysis = {
                    'pattern': correction_type,
                    'confidence': confidence,
                    'details': correction_analysis.get('details', {}),
                    'suggested_improvements': correction_analysis.get('suggested_improvements', [])
                }
                
                # Si hay texto original y corregido, calcular WER
                original_text = correction_analysis.get('original_text', '')
                corrected_text = correction_analysis.get('corrected_text', '')
                wer = self._calculate_wer(original_text, corrected_text) if original_text and corrected_text else None
                
                cursor.execute('''
                    INSERT INTO transcription_corrections 
                    (user_id, session_id, original_text, corrected_text,
                     correction_type, word_error_rate, pattern_analysis_json,
                     learned_lesson)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_id,
                    correction_analysis.get('session_id'),
                    original_text,
                    corrected_text,
                    correction_type,
                    wer,
                    json.dumps(pattern_analysis, ensure_ascii=False),
                    correction_analysis.get('learned_lesson')
                ))
                
                # Registrar métrica
                self.log_system_metric(
                    'correction_analysis_saved',
                    1,
                    {
                        'correction_type': correction_type,
                        'user_id': user_id,
                        'confidence': confidence,
                        'wer': wer
                    }
                )
                
                conn.commit()
                self.logger.debug(f"🎓 Análisis de corrección guardado para usuario {user_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Error guardando análisis de corrección: {e}")
            return False
    
    def get_correction_patterns(self, user_id: str = None, limit: int = 100) -> List[Dict]:
        """
        Obtener patrones de corrección para análisis.
        
        Args:
            user_id: ID específico del usuario (opcional)
            limit: Límite de correcciones a recuperar
            
        Returns:
            List[Dict]: Patrones de corrección
        """
        cache_key = f"correction_patterns_{user_id if user_id else 'global'}_{limit}"
        cached = self._cache_get(cache_key)
        if cached:
            return cached
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if user_id:
                    cursor.execute('''
                        SELECT correction_id, original_text, corrected_text,
                               correction_type, word_error_rate, pattern_analysis_json,
                               learned_lesson, timestamp
                        FROM transcription_corrections 
                        WHERE user_id = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    ''', (user_id, limit))
                else:
                    cursor.execute('''
                        SELECT correction_id, user_id, original_text, corrected_text,
                               correction_type, word_error_rate, pattern_analysis_json,
                               learned_lesson, timestamp
                        FROM transcription_corrections 
                        ORDER BY timestamp DESC
                        LIMIT ?
                    ''', (limit,))
                
                corrections = []
                for row in cursor.fetchall():
                    correction = dict(row)
                    if correction.get('pattern_analysis_json'):
                        correction['pattern_analysis'] = json.loads(correction['pattern_analysis_json'])
                        del correction['pattern_analysis_json']
                    
                    corrections.append(correction)
                
                self._cache_set(cache_key, corrections)
                return corrections
                
        except Exception as e:
            self.logger.error(f"❌ Error obteniendo patrones de corrección: {e}")
            return []
    
    # ========== MÉTRICAS Y ANÁLISIS ==========
    
    def log_system_metric(self, metric_type: str, metric_value: float, 
                         metadata: Dict = None) -> bool:
        """
        Registrar métrica del sistema.
        
        Args:
            metric_type: Tipo de métrica
            metric_value: Valor de la métrica
            metadata: Metadatos adicionales
            
        Returns:
            bool: True si se registró correctamente
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
                
                cursor.execute('''
                    INSERT INTO system_metrics (metric_type, metric_value, metadata_json)
                    VALUES (?, ?, ?)
                ''', (metric_type, metric_value, metadata_json))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Error registrando métrica: {e}")
            return False
    
    def get_system_metrics(self, metric_type: str = None, 
                          start_date: datetime = None,
                          end_date: datetime = None,
                          limit: int = 1000) -> List[Dict]:
        """
        Obtener métricas del sistema.
        
        Args:
            metric_type: Tipo específico de métrica (opcional)
            start_date: Fecha de inicio (opcional)
            end_date: Fecha de fin (opcional)
            limit: Límite de resultados
            
        Returns:
            List[Dict]: Métricas del sistema
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                query = '''
                    SELECT metric_id, metric_type, metric_subtype, metric_value,
                           metadata_json, timestamp
                    FROM system_metrics 
                    WHERE 1=1
                '''
                params = []
                
                if metric_type:
                    query += " AND metric_type = ?"
                    params.append(metric_type)
                
                if start_date:
                    query += " AND timestamp >= ?"
                    params.append(start_date.isoformat())
                
                if end_date:
                    query += " AND timestamp <= ?"
                    params.append(end_date.isoformat())
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                
                metrics = []
                for row in cursor.fetchall():
                    metric = dict(row)
                    if metric.get('metadata_json'):
                        metric['metadata'] = json.loads(metric['metadata_json'])
                        del metric['metadata_json']
                    
                    metrics.append(metric)
                
                return metrics
                
        except Exception as e:
            self.logger.error(f"❌ Error obteniendo métricas: {e}")
            return []
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas completas del sistema.
        
        Returns:
            Dict: Estadísticas del sistema
        """
        cache_key = "system_stats"
        cached = self._cache_get(cache_key)
        if cached:
            return cached
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Estadísticas básicas
                queries = {
                    'total_users': 'SELECT COUNT(*) FROM users',
                    'active_users_24h': '''
                        SELECT COUNT(DISTINCT user_id) 
                        FROM interactions 
                        WHERE timestamp > datetime("now", "-24 hours")
                    ''',
                    'total_interactions': 'SELECT COUNT(*) FROM interactions',
                    'interactions_24h': '''
                        SELECT COUNT(*) 
                        FROM interactions 
                        WHERE timestamp > datetime("now", "-24 hours")
                    ''',
                    'total_feedback': 'SELECT COUNT(*) FROM recommendation_feedback',
                    'total_corrections': 'SELECT COUNT(*) FROM transcription_corrections',
                    'avg_processing_time': '''
                        SELECT AVG(processing_time) 
                        FROM interactions 
                        WHERE processing_time IS NOT NULL
                    ''',
                    'avg_confidence': '''
                        SELECT AVG(confidence_score) 
                        FROM interactions 
                        WHERE confidence_score IS NOT NULL
                    '''
                }
                
                for key, query in queries.items():
                    cursor.execute(query)
                    stats[key] = cursor.fetchone()[0] or 0
                
                # Distribución de tipos de interacción
                cursor.execute('''
                    SELECT interaction_type, COUNT(*) 
                    FROM interactions 
                    GROUP BY interaction_type
                ''')
                stats['interaction_types'] = dict(cursor.fetchall())
                
                # Distribución de feedback
                cursor.execute('''
                    SELECT feedback_type, COUNT(*) 
                    FROM recommendation_feedback 
                    GROUP BY feedback_type
                ''')
                stats['feedback_distribution'] = dict(cursor.fetchall())
                
                # Usuarios por segmento
                cursor.execute('''
                    SELECT user_segment, COUNT(*) 
                    FROM users 
                    GROUP BY user_segment
                ''')
                stats['user_segments'] = dict(cursor.fetchall())
                
                # Tendencias temporales (últimos 7 días)
                cursor.execute('''
                    SELECT DATE(timestamp) as date, COUNT(*) as count
                    FROM interactions
                    WHERE timestamp > datetime("now", "-7 days")
                    GROUP BY DATE(timestamp)
                    ORDER BY date
                ''')
                
                stats['daily_trends'] = [
                    {'date': row[0], 'count': row[1]}
                    for row in cursor.fetchall()
                ]
                
                self._cache_set(cache_key, stats)
                return stats
                
        except Exception as e:
            self.logger.error(f"❌ Error obteniendo estadísticas: {e}")
            return {}
    
    # ========== FUNCIONES DE MANTENIMIENTO ==========
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """
        Limpiar datos antiguos para mantener la base de datos optimizada.
        
        Args:
            days_to_keep: Número de días de datos a mantener
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cutoff_date = f"datetime('now', '-{days_to_keep} days')"
                
                # Limpiar datos antiguos
                tables_to_clean = [
                    'interactions',
                    'recommendation_feedback', 
                    'transcription_corrections',
                    'system_metrics'
                ]
                
                stats = {}
                for table in tables_to_clean:
                    cursor.execute(f'''
                        DELETE FROM {table} 
                        WHERE timestamp < {cutoff_date}
                    ''')
                    stats[table] = cursor.rowcount
                
                # Vacuum para recuperar espacio
                cursor.execute('VACUM')
                
                conn.commit()
                
                # Limpiar cache
                self.cache.clear()
                
                self.logger.info(f"🧹 Limpieza completada: {stats}")
                
        except Exception as e:
            self.logger.error(f"❌ Error en limpieza de datos: {e}")
    
    def export_data(self, export_path: str = "exports/"):
        """
        Exportar datos para análisis o backup.
        
        Args:
            export_path: Ruta para guardar los exports
        """
        try:
            Path(export_path).mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            with self._get_connection() as conn:
                tables = [
                    'users', 'interactions', 'recommendation_feedback',
                    'transcription_corrections', 'user_sessions', 'system_metrics'
                ]
                
                for table in tables:
                    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                    if not df.empty:
                        export_file = f"{export_path}/{table}_{timestamp}.csv"
                        df.to_csv(export_file, index=False, encoding='utf-8')
                        self.logger.info(f"📤 Exportado {len(df)} registros de {table} a {export_file}")
                
            self.logger.info("✅ Exportación completada")
            
        except Exception as e:
            self.logger.error(f"❌ Error exportando datos: {e}")
    
    def optimize_database(self):
        """Optimizar base de datos para mejor rendimiento."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Analizar tablas para optimizar queries
                cursor.execute('ANALYZE')
                
                # Reconstruir índices
                cursor.execute('REINDEX')
                
                # Actualizar estadísticas
                cursor.execute('UPDATE sqlite_stat1')
                
                conn.commit()
                self.logger.info("⚡ Base de datos optimizada")
                
        except Exception as e:
            self.logger.error(f"❌ Error optimizando base de datos: {e}")
    
    # ========== FUNCIONES DE CONSULTA AVANZADA ==========
    
    async def get_user_count(self) -> int:
        """Obtener número total de usuarios únicos."""
        return self.get_system_stats().get('total_users', 0)
    
    async def get_interaction_count(self) -> int:
        """Obtener número total de interacciones."""
        return self.get_system_stats().get('total_interactions', 0)
    
    def get_recommendation_quality_metrics(self, user_id: str = None) -> Dict[str, Any]:
        """
        Obtener métricas de calidad de recomendaciones.
        
        Args:
            user_id: ID específico del usuario (opcional)
            
        Returns:
            Dict: Métricas de calidad
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if user_id:
                    cursor.execute('''
                        SELECT 
                            COUNT(*) as total_recommendations,
                            AVG(confidence_score) as avg_confidence,
                            SUM(CASE WHEN feedback_json LIKE '%"like"%' THEN 1 ELSE 0 END) as positive_feedback,
                            SUM(CASE WHEN feedback_json LIKE '%"dislike"%' THEN 1 ELSE 0 END) as negative_feedback
                        FROM interactions 
                        WHERE user_id = ? 
                        AND interaction_type = 'recommendation'
                    ''', (user_id,))
                else:
                    cursor.execute('''
                        SELECT 
                            COUNT(*) as total_recommendations,
                            AVG(confidence_score) as avg_confidence,
                            SUM(CASE WHEN feedback_json LIKE '%"like"%' THEN 1 ELSE 0 END) as positive_feedback,
                            SUM(CASE WHEN feedback_json LIKE '%"dislike"%' THEN 1 ELSE 0 END) as negative_feedback
                        FROM interactions 
                        WHERE interaction_type = 'recommendation'
                    ''')
                
                row = cursor.fetchone()
                if row:
                    metrics = dict(row)
                    
                    # Calcular tasa de aceptación
                    total_feedback = metrics.get('positive_feedback', 0) + metrics.get('negative_feedback', 0)
                    if total_feedback > 0:
                        metrics['acceptance_rate'] = metrics.get('positive_feedback', 0) / total_feedback
                    else:
                        metrics['acceptance_rate'] = 0.0
                    
                    return metrics
                
                return {}
                
        except Exception as e:
            self.logger.error(f"❌ Error obteniendo métricas de calidad: {e}")
            return {}

# Instancia global del gestor de base de datos
db_manager = DatabaseManager()