# Script de entrenamiento del modelo de recomendación LightFM

import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import (
    precision_at_k,
    recall_at_k
)
import pickle
import logging
from datetime import datetime
import os
from config.paths import DATA_DIR

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Entrenador para el modelo de recomendación LightFM.
    """
    
    def __init__(self):
        self.dataset = None
        self.model = None
        
    def load_data(self):
        """Cargar datos de entrenamiento."""
        try:
            logger.info("📂 Cargando datos de entrenamiento...")
            
            # Cargar datos de películas
            movies_df = pd.read_csv(
                DATA_DIR / "movies.csv",
                engine="python"
            )
            logger.info(f"🎬 {len(movies_df)} películas cargadas")
            
            # Cargar datos de ratings
            ratings_df = pd.read_csv(
                DATA_DIR / "ratings.csv",
                engine="python"
            )
            logger.info(f"⭐ {len(ratings_df)} ratings cargados")
            
            return movies_df, ratings_df
            
        except Exception as e:
            logger.error(f"❌ Error cargando datos: {e}")
            raise
    
    def prepare_dataset(self, movies_df, ratings_df):
        """Preparar dataset para LightFM."""
        try:
            logger.info("🛠️ Preparando dataset...")
            
            # Inicializar dataset
            self.dataset = Dataset()
            
            # Ajustar dataset con usuarios, items y características
            user_ids = ratings_df['user_id'].unique()
            movie_ids = movies_df['movie_id'].unique()
            
            # Obtener características de items
            item_features = self._extract_item_features(movies_df)
            
            self.dataset.fit(
                users=user_ids,
                items=movie_ids,
                item_features=item_features
            )
            
            logger.info("✅ Dataset preparado")
            return self.dataset
            
        except Exception as e:
            logger.error(f"❌ Error preparando dataset: {e}")
            raise
    
    def _extract_item_features(self, movies_df):
        """Extraer características de items desde los datos de películas."""
        # Extraer géneros únicos
        all_genres = set()
        for genres in movies_df['genres'].str.split('|'):
            if isinstance(genres, list):
                all_genres.update(genres)
        
        return list(all_genres)
    
    def build_interactions(self, ratings_df):
        """Construir matrices de interacción."""
        try:
            logger.info("🔨 Construyendo interacciones...")
            
            # Construir interacciones
            interactions = []
            for _, row in ratings_df.iterrows():
                interactions.append((row['user_id'], row['movie_id'], row['rating']))
            
            interactions_matrix, weights_matrix = self.dataset.build_interactions(interactions)
            
            logger.info(f"✅ Matriz de interacciones construida: {interactions_matrix.shape}")
            return interactions_matrix, weights_matrix
            
        except Exception as e:
            logger.error(f"❌ Error construyendo interacciones: {e}")
            raise
    
    def build_item_features(self, movies_df):
        """Construir matriz de características de items."""
        try:
            logger.info("🔨 Construyendo características de items...")
            
            item_features_list = []
            for _, row in movies_df.iterrows():
                movie_id = row['movie_id']
                genres = row['genres'].split('|') if pd.notna(row['genres']) else []
                item_features_list.append((movie_id, genres))
            
            item_features = self.dataset.build_item_features(item_features_list)
            
            logger.info(f"✅ Características de items construidas")
            return item_features
            
        except Exception as e:
            logger.error(f"❌ Error construyendo características: {e}")
            raise
    
    def train_model(self, interactions, item_features, epochs=30):
        """Entrenar modelo LightFM."""
        try:
            logger.info("🎯 Entrenando modelo LightFM...")
            
            # Configuración del modelo
            self.model = LightFM(
                loss='warp',
                learning_rate=0.05,
                no_components=30,
                user_alpha=0.0001,
                item_alpha=0.0001
            )
            
            # Entrenamiento
            self.model.fit(
                interactions,
                item_features=item_features,
                epochs=epochs,
                num_threads=4,
                verbose=True
            )
            
            logger.info("✅ Modelo entrenado exitosamente")
            return self.model
            
        except Exception as e:
            logger.error(f"❌ Error entrenando modelo: {e}")
            raise
    
    def save_model(self, filepath='trained_models/lightfm_model.pkl'):
        """Guardar modelo entrenado."""
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Preparar datos del modelo
            model_data = {
                'model': self.model,
                'dataset': self.dataset,
                'movie_features': self.dataset,
                'movie_mapping': self.dataset.mapping()[2],
                'user_mapping': self.dataset.mapping()[0],
                'training_date': datetime.now().isoformat(),
                'model_config': {
                    'loss': 'warp',
                    'no_components': 30,
                    'learning_rate': 0.05
                }
            }
            
            # Guardar
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"💾 Modelo guardado en: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Error guardando modelo: {e}")
            raise

    def ndcg_at_k(model, test_interactions, k=10):
        """
        Compute NDCG@k for LightFM model
        """
        num_users = test_interactions.shape[0]
        ndcg_scores = []

        for user_id in range(num_users):
            # items relevantes reales
            relevant_items = test_interactions.tocsr()[user_id].indices

            if len(relevant_items) == 0:
                continue

            # predicciones
            scores = model.predict(
                user_id,
                np.arange(test_interactions.shape[1])
            )

            # top-k items
            top_k_items = np.argsort(-scores)[:k]

            dcg = 0.0
            for i, item in enumerate(top_k_items):
                if item in relevant_items:
                    dcg += 1 / np.log2(i + 2)

            # ideal dcg
            idcg = sum(
                1 / np.log2(i + 2)
                for i in range(min(len(relevant_items), k))
            )

            ndcg_scores.append(dcg / idcg)

        return np.mean(ndcg_scores)
    
    def evaluate_model(self, test_interactions):

        real_precision = precision_at_k(self.model, self.dataset, k=10).mean()
        recall = recall_at_k(self.model, self.dataset, k=10).mean()
        ndcg = self.ndcg_at_k(self.model, test_interactions, k=10)

        return {
            'precision': real_precision,
            'recall': recall,
            'ndcg': ndcg 
        }

def main():
    """Función principal de entrenamiento."""
    logger.info("🚀 Iniciando entrenamiento del modelo de recomendación...")
    
    try:
        # Inicializar entrenador
        trainer = ModelTrainer()
        
        # 1. Cargar datos
        movies_df, ratings_df = trainer.load_data()
        
        # 2. Preparar dataset
        trainer.prepare_dataset(movies_df, ratings_df)
        
        # 3. Construir interacciones
        interactions, weights = trainer.build_interactions(ratings_df)
        
        # 4. Construir características
        item_features = trainer.build_item_features(movies_df)
        
        # 5. Entrenar modelo
        trainer.train_model(interactions, item_features, epochs=30)
        
        # 6. Guardar modelo
        trainer.save_model()
        
        logger.info("🎉 Entrenamiento completado exitosamente!")
        
    except Exception as e:
        logger.error(f"💥 Error en entrenamiento: {e}")
        raise

if __name__ == "__main__":
    main()