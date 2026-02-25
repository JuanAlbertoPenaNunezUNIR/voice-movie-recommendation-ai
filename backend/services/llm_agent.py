import requests
import json
import re
from datetime import datetime

OLLAMA_URL = "http://ollama:11434/api/chat"
MODEL = "llama3.1"

SYSTEM_PROMPT = """
Tu única función es convertir la petición de un usuario sobre películas en un objeto JSON.
Analiza el texto y rellena los campos del JSON según las siguientes reglas.
Debes devolver SIEMPRE un JSON válido, incluso si la petición es ambigua. Si no puedes extraer un filtro, deja su valor por defecto.

FECHA ACTUAL: {date_context}

REGLAS DE EXTRACCIÓN:
- "intent": "recommend_movies" para peticiones de cine, "greeting" para saludos. Si no es ninguna, "other".
- "filters":
  - "genres": Mapea géneros a su término en Inglés (ej: 'ciencia ficción' -> 'Science Fiction', 'terror' -> 'Horror', 'comedia' o 'de risa' -> 'Comedy').
  - "director": Extrae el nombre del director (ej: 'de Christopher Nolan' -> 'Christopher Nolan').
  - "actors": Extrae nombres de actores.
  - "year_min", "year_max": Extrae rangos de años (ej: 'años 80' -> 1980, 1989; 'recientes' -> {current_year}-5, {current_year}).
  - "sort_by": Usa 'vote_average.desc' si pide 'mejores' o 'top'. Usa 'primary_release_date.desc' si pide 'nuevas'. Por defecto, 'popularity.desc'.
  - "similar_to": Extrae el título de la película de referencia (ej: 'tipo Scary Movie' -> 'Scary Movie').
  - "keywords": Extrae subgéneros o conceptos (ej: 'humor absurdo', 'parodia', 'zombies'). Si usas 'similar_to', intenta inferir los keywords de esa película.
  - "limit": Extrae la cantidad (ej: 'una película' -> 1).

EJEMPLOS:
- Usuario: "Mejores películas de ciencia ficción de la historia"
  JSON: {{"intent": "recommend_movies", "filters": {{"genres": ["Science Fiction"], "sort_by": "vote_average.desc"}}}}
- Usuario: "¿Cuáles son las mejores películas de Christopher Nolan?"
  JSON: {{"intent": "recommend_movies", "filters": {{"director": "Christopher Nolan", "sort_by": "vote_average.desc"}}}}
- Usuario: "Recomiéndame película de risa tipo Scary Movie."
  JSON: {{"intent": "recommend_movies", "filters": {{"genres": ["Comedy"], "similar_to": "Scary Movie", "keywords": ["parody", "spoof"]}}}}
- Usuario: "Clásicos del humor absurdo"
  JSON: {{"intent": "recommend_movies", "filters": {{"genres": ["Comedy"], "keywords": ["absurd humor"], "sort_by": "vote_average.desc"}}}}

ESQUEMA JSON (obligatorio):
{
  "intent": "recommend_movies | greeting | other",
  "detected_name": null,
  "filters": {
    "genres": [],
    "actors": [],
    "director": null,
    "keywords": [],
    "sort_by": "popularity.desc",
    "year_min": null,
    "year_max": null,
    "similar_to": null,
    "limit": 5
  }
}

Ahora, procesa la siguiente petición del usuario y devuelve SÓLO el objeto JSON.
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
    current_year = datetime.now().year
    final_system_prompt = SYSTEM_PROMPT.replace("{date_context}", today).replace("{current_year}", str(current_year))

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
        return {"intent": "other", "response": "Lo siento, no he podido entender tu petición. ¿Puedes reformularla?", "filters": {}}

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
        director = ", ".join(m.get("directors", []))
        movies_ctx.append(f"- Título: {title} ({year}), dirigida por {director if director else 'un director desconocido'}.")
    movies_str = "\n".join(movies_ctx)

    today = datetime.now().strftime("%Y-%m-%d")
    
    personalization_instruction = ""
    if user_name:
        personalization_instruction = f"10. Dirígete al usuario por su nombre: {user_name}."

    system_prompt = f"""Eres un asistente de cine experto y carismático.
Tu objetivo es presentar las recomendaciones de películas al usuario de forma fluida y natural.

Instrucciones CLAVE:
1. **RESPUESTA MUY BREVE.** La respuesta final NO DEBE SUPERAR los 240 caracteres para que el motor de voz la procese rápido.
2. **CRÍTICO: Tu respuesta DEBE OBLIGATORIAMENTE mencionar los títulos de la sección 'Películas encontradas'. NO menciones ninguna otra película que no esté en esa lista.**
3. Menciona los títulos de las películas de forma fluida, no como una lista.
4. Si hay 3 películas o menos, menciónalas todas. Si hay más, puedes decir "He encontrado varias opciones como..." y mencionar las dos primeras.
5. Basa tus comentarios únicamente en el título, año y director. NO inventes detalles.
6. Tono entusiasta y directo.
{personalization_instruction}

Ejemplo de respuesta CORTA y válida:
"¡Claro, Alberto! Para los 80, te recomiendo 'Aliens: El regreso' de James Cameron y 'La Cosa' de John Carpenter. ¡Dos clásicos que no te puedes perder!"
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
