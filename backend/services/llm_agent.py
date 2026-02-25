import requests
import json
import re
from datetime import datetime

OLLAMA_URL = "http://ollama:11434/api/chat"
MODEL = "llama3.1"

SYSTEM_PROMPT = """
Eres un asistente experto en cine.

Tu objetivo es interpretar la intención del usuario y generar un JSON ESTRUCTURADO
para consultar una base de datos de películas (TMDB).

IMPORTANTE: Mapea términos coloquiales a géneros oficiales de TMDB (Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Family, Fantasy, History, Horror, Music, Mystery, Romance, Science Fiction, TV Movie, Thriller, War, Western).

FECHA ACTUAL: {date_context}
Debes DETECTAR si el usuario se está presentando y diciendo su nombre.
Debes DETECTAR si el usuario pide una cantidad específica de películas (ej: "dime una", "recomienda 3"). Si pide en singular ("una película"), limit=1.
Si detectas un nombre, el intent debe ser "greeting" y el response debe saludar al usuario por su nombre.

Devuelve SIEMPRE un JSON válido con este esquema exacto:

{
  "intent": "recommend_movies | greeting | other",
  "detected_name": null,
  "filters": {
    "genres": [],
    "keywords": [], 
    "actors": [],
    "director": null,
    "sort_by": "popularity.desc",
    "year_min": null,
    "year_max": null,
    "similar_to": null, 
    "max_duration": null,
    "limit": 5
  },
  "response": "Respuesta natural en español"
}

Reglas:

1. Si el usuario dice "Hola, soy [nombre]" o similar:
   - intent = "greeting"
   - response = "¡Hola, [nombre]! ¡Pulsa el icono del micrófono para ampliar tu horizonte cinéfilo!"
   - filters = {}

2. Si el usuario pide películas:
   - intent = "recommend_movies"
   - "quiero ver algo de miedo" -> genres: ["Horror"]
   - "recomiéndame una película de..." -> limit: 1
   - "una película de..." -> limit: 1
   - "la mejor película de..." -> sort_by: "vote_average.desc"
   - "películas de Brad Pitt" -> actors: ["Brad Pitt"]
   - "estrenos recientes" -> year_min: 2020, year_max: 2026
   - "películas recientes" -> year_min: 2020, year_max: 2026
   - "año pasado" (Calcula respecto a la fecha actual) -> year_min: [AÑO_ACTUAL-1], year_max: [AÑO_ACTUAL-1]
   - "década de los 80" -> year_min: 1980, year_max: 1989
   - "buena película reciente" -> year_min: 2020, year_max: 2026, sort_by: "vote_average.desc"
   - "mejores películas recientes de Brad Pitt" -> actors: ["Brad Pitt"], year_min: 2020, year_max: 2026, sort_by: "vote_average.desc"
   - "comedias románticas" -> genres: ["Comedy", "Romance"]
   - "comedias de terror" -> genres: ["Comedy", "Horror"]
   - "comedias de acción" -> genres: ["Comedy", "Action"]
   - "drama romántico" -> genres: ["Drama", "Romance"]
   - "ciencia ficción con terror" -> genres: ["Science Fiction", "Horror"]
   - "ciencia ficción de terror" -> genres: ["Science Fiction", "Horror"]
   - "thriller policiaco" -> genres: ["Thriller", "Crime"]
   - "aventura fantástica" -> genres: ["Adventure", "Fantasy"]
   - "acción y aventuras" -> genres: ["Action", "Adventure"]
   - "drama histórico" -> genres: ["Drama", "History"]
   - "musicales dramáticos" -> genres: ["Music", "Drama"]
   - "cine bélico dramático" -> genres: ["War", "Drama"]
   - "animación familiar" -> genres: ["Animation", "Family"]
   - "cine negro" -> genres: ["Crime", "Drama"]
   - "mejores películas de la historia" -> sort_by: "vote_average.desc", genres: []
   - "mejores películas de terror de la historia" -> sort_by: "vote_average.desc", genres: ["Horror"]
   - "thriller psicológico" -> genres: ["Thriller", "Mystery"]
   - "fantasía épica" -> genres: ["Fantasy", "Adventure"]
   - "terror sobrenatural" -> genres: ["Horror", "Fantasy"]
   - "superhéroes" -> genres: ["Action", "Science Fiction"]
   - "western espacial" -> genres: ["Western", "Science Fiction"]
   - "pelis de risa de los 90" -> genres: ["Comedy"], year_min: 1990, year_max: 1999
   
   IMPORTANTE SOBRE "response":
   - Debe ser una frase de transición genérica (ej: "Claro, aquí tienes algunas opciones.", "Buscando películas de terror...").
   - NO alucines ni inventes que el usuario ha mencionado películas.
   - NO hagas preguntas.

3. Si la petición es ambigua o insuficiente:
   - intent = "other"
   - response = "Por favor, dime qué tipo de películas te gustan o menciona un actor o género."
"""

def extract_name(text: str):
    """
    Extrae un nombre de usuario de frases tipo:
    'Hola, soy Alberto', 'Mi nombre es Alberto', etc.
    Soporta nombres compuestos (Juan Alberto).
    """
    patterns = [
        r"soy\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)?)",
        r"mi nombre es\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)?)",
        r"me llamo\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)?)",
        r"me dicen\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)?)",
        r"conocen como\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)?)"
    ]
    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        if match:
            return match.group(1).title()
    return None

def run_llm_agent(user_text: str):
    # Primero, intentamos detectar nombre localmente
    detected_name = extract_name(user_text)
    if detected_name:
        return {
            "intent": "greeting",
            "filters": {},
            "response": f"¡Hola, {detected_name}! ¡Pulsa el icono del micrófono para ampliar tu horizonte cinéfilo!",
            "detected_name": detected_name
        }

    # Si no es saludo, enviamos al LLM
    today = datetime.now().strftime("%Y-%m-%d")
    final_system_prompt = SYSTEM_PROMPT.replace("{date_context}", today)

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": final_system_prompt},
            {"role": "user", "content": user_text}
        ],
        "stream": False
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=60)
    r.raise_for_status()

    content = r.json()["message"]["content"]
    
    # Limpieza ROBUSTA: Buscar el primer '{' y el último '}' para extraer el JSON
    # Esto soluciona el problema cuando el LLM añade texto como "Lo siento..." antes del JSON
    start_idx = content.find('{')
    end_idx = content.rfind('}')
    if start_idx != -1 and end_idx != -1:
        content = content[start_idx:end_idx+1]
            
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Fallback si el LLM no devuelve JSON puro
        return {"intent": "other", "response": content, "filters": {}}

    # --- LÓGICA DE RESCATE DE NOMBRE ---
    # Si el LLM dice que es un saludo pero olvidó rellenar 'detected_name' en el JSON
    if data.get("intent") == "greeting" and not data.get("detected_name"):
        # 1. Intentamos extraerlo de nuevo del texto original con los regex mejorados
        name_local = extract_name(user_text)
        if name_local:
            data["detected_name"] = name_local
        # 2. Si falla, intentamos extraerlo de la respuesta generada por el LLM (ej: "Hola Juan,")
        elif "response" in data:
            match_resp = re.search(r"(?:Hola|Encantada|Bienvenido|Qué tal)[,\s]+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)", data["response"], re.IGNORECASE)
            if match_resp:
                data["detected_name"] = match_resp.group(1).title()

    return data

def generate_recommendation_response(user_text: str, movies: list, initial_response: str, user_name: str = None):
    """
    Genera una respuesta natural mencionando las películas encontradas,
    permitiendo al LLM añadir comentarios breves o variar la estructura.
    """
    
    # SANITIZACIÓN: Si la respuesta inicial fue una negativa/disculpa pero SÍ encontramos películas,
    # la sobrescribimos para evitar incoherencias ("Lo siento, no puedo... Aquí tienes El Padrino").
    if any(phrase in initial_response.lower() for phrase in ["lo siento", "no puedo", "no he encontrado", "disculpa", "unable", "sorry"]):
        initial_response = "¡Claro! He encontrado estas excelentes opciones para ti."
    
    # Preparar contexto de películas (Título + Año) para que el LLM sepa de qué habla
    movies_ctx = []
    for m in movies:
        title = m.get("title", "Película")
        year = m.get("release_year", "")
        director = ", ".join(m.get("directors", ["Desconocido"]))
        cast = ", ".join([c["name"] for c in m.get("main_cast", [])])
        genres = ", ".join(m.get("genres", []))
        overview = m.get("overview", "Sin descripción disponible.")
        movies_ctx.append(f"- Título: {title} ({year})\n  Director: {director}\n  Reparto: {cast}\n  Géneros: {genres}\n  Sinopsis Real: {overview}\n")
    movies_str = "\n".join(movies_ctx)

    today = datetime.now().strftime("%Y-%m-%d")
    
    personalization_instruction = ""
    if user_name:
        personalization_instruction = f"10. Dirígete al usuario por su nombre: {user_name}."

    system_prompt = f"""Eres un asistente de cine experto y carismático.
Tu objetivo es presentar las recomendaciones de películas al usuario de forma fluida y natural.
FECHA ACTUAL: {today}

Instrucciones:
1. Integra los títulos de las películas en tu respuesta.
2. NO hagas una lista esquemática ni uses "Te sugiero:" repetitivamente.
3. Puedes añadir breves comentarios, curiosidades o adjetivos sobre las películas para hacerlas atractivas.
4. Mantén la respuesta concisa (máximo 3-4 frases).
5. El tono debe ser entusiasta y personalizado.
6. Menciona explícitamente la cantidad de películas encontradas (ej: "He encontrado 3 películas...", "Aquí tienes esta recomendación...").
7. CRÍTICO: Usa EXCLUSIVAMENTE la información proporcionada (Sinopsis Real, Director, Reparto) para describir la película. NO inventes tramas ni mezcles datos de otras películas.
8. Si la búsqueda ha fallado y las películas no tienen nada que ver con lo pedido, DILO: "No he encontrado exactamente lo que buscabas, pero estas son populares...".
9. IMPORTANTE: Si hay 5 películas o menos en la lista, MENCIONA TODAS por su título. No omitas ninguna.
{personalization_instruction}
"""

    user_prompt = f"""
Usuario: "{user_text}"
Tu intención inicial: "{initial_response}"

Películas encontradas:
{movies_str}

Genera la respuesta final hablada (texto plano) que incluya las recomendaciones de forma natural.
"""

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=30)
        r.raise_for_status()
        content = r.json()["message"]["content"].strip()
        # Limpieza básica por si el LLM pone comillas
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1]
        return content
    except Exception as e:
        # Fallback básico en caso de error del LLM
        titles = [m.get("title", "") for m in movies]
        titles_str = ", ".join(filter(None, titles))
        return f"{initial_response} Aquí tienes: {titles_str}."