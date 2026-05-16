import os
import json
from turtle import st
from groq import Groq
import joblib
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn import pipeline
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import google.genai as genai
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors





# ============================================================
# 0. CONFIGURACION INICIAL
# ============================================================


# Cargar la API key desde el archivo .env
load_dotenv()
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
except KeyError:
        raise ValueError("Una o más variables de entorno no están definidas.")

# Modelo de Gemini que usaremos (rapido y economico)
MODELO = "gemini-2.0-flash"
client_gemini = genai.Client(api_key=GEMINI_API_KEY)
client_groq = Groq(api_key=GROQ_API_KEY)


# Caracteristicas de audio que usaremos para el clasificador
# Columnas que nos interesan para el analisis
audio_features = [
            "energy", "tempo", "danceability", "loudness",
            "valence", "speechiness", "instrumentalness",
            "acousticness", "liveness", "duration_ms"
        ]
descriptivas = [
            "track_name", "track_artist",
            "playlist_genre", "playlist_subgenre", "track_popularity"
        ]

# Valores por defecto para features que el LLM no infiere
DEFAULTS = {
    "energy": 0.5,
    "tempo": 120,
    "danceability": 0.5,
    "loudness": -10,
    "valence": 0.5,
    "speechiness": 0.5,
    "instrumentalness": 0.5,
    "acousticness": 0.5,
    "liveness": 0.5,
    "duration_ms": 210000,
}

# ============================================================
# 1. CARGA DE DATOS DE SPOTIFY
# ============================================================

def cargar_datos_spotify():
    """
    Descarga el dataset de Spotify si no existe localmente,
    lo limpia y lo guarda para usos futuros.
    Retorna un DataFrame con las canciones y sus caracteristicas.
    """
    
    # Si ya tenemos el archivo limpio, lo cargamos directamente
    if os.path.exists("spotify_clean.csv"):
        print("Cargando dataset desde disco...")
        df = pd.read_csv("spotify_clean.csv")
    else:
        # Descargar dataset desde Kaggle
        print("Descargando dataset de Spotify...")
        import kagglehub
        path = kagglehub.dataset_download("solomonameh/spotify-music-dataset")
        
        # Cargar los dos archivos (canciones populares y no populares)
        df_high = pd.read_csv(os.path.join(path, "high_popularity_spotify_data.csv"))
        df_low = pd.read_csv(os.path.join(path, "low_popularity_spotify_data.csv"))
        
        # Agregar columna que indica popularidad (1 = popular, 0 = no popular)
        df_high["popular"] = 1
        df_low["popular"] = 0
        
        # Unir ambos datasets y eliminar duplicados
        df = pd.concat([df_high, df_low], ignore_index=True)
        df = df.drop_duplicates()
        
        # Quedarnos solo con las columnas utiles y eliminar filas con datos faltantes
        df = df[audio_features + descriptivas + ["popular"]].dropna()
        df = df.reset_index(drop=True)
        df = agrupar_generos(df)

        print(df["genre_group"].value_counts())
        print("Total canciones:", len(df))
        
        # Guardar el dataset limpio para no tener que descargarlo otra vez
        df.to_csv("spotify_clean.csv", index=False)
    
    print(f"Dataset cargado: {len(df)} canciones")
    return df

def agrupar_generos(df):
        # ANTES: 35 géneros específicos, muchos con muy pocas canciones
    # DESPUÉS: 6 grupos grandes y balanceados

    GENRE_MAP = {
        # Electrónica — todo lo sintético, producido digitalmente
        "electronic": "electronic",
        "gaming":     "electronic",   # soundtracks de videojuegos, muy sintéticos
        "lofi":       "electronic",   # beats lo-fi, producción digital
        "disco":      "electronic",   # base electrónica/sintetizadores

        # Pop — comercial, pegajoso, amplio público
        "pop":        "pop",
        "latin":      "pop",          # reggaeton/pop latino, estructura pop
        "cantopop":   "pop",          # pop cantonés
        "k-pop":      "pop",          # pop coreano
        "j-pop":      "pop",          # pop japonés
        "mandopop":   "pop",          # pop mandarín
        "indie":      "pop",          # pop alternativo/independiente

        # Urban — ritmo, flow, vocal prominente
        "hip-hop":    "urban",
        "r&b":        "urban",
        "funk":       "urban",
        "reggae":     "urban",
        "soca":       "urban",        # caribeño, muy rítmico
        "afrobeats":  "urban",        # africano moderno, muy rítmico
        "korean":     "urban",        # urban coreano (distinto del k-pop)

        # Rock — guitarras, energía, banda
        "rock":       "rock",
        "metal":      "rock",
        "punk":       "rock",
        "blues":      "rock",         # raíz del rock, guitarra prominente
        "country":    "rock",         # guitarras acústicas, estructura similar

        # Calm — tranquilo, introspectivo, poco energético
        "ambient":    "calm",
        "classical":  "calm",
        "jazz":       "calm",
        "folk":       "calm",
        "wellness":   "calm",         # meditación, relajación
        "soul":       "calm",
        "gospel":     "calm",
        "lofi":       "calm",         # también encaja aquí, pero lo pusimos en electronic

        # World — música étnica/regional no occidental
        "world":      "world",
        "arabic":     "world",
        "brazilian":  "world",
        "turkish":    "world",
        "indian":     "world",
    }

    df["genre_group"] = df["playlist_genre"].map(GENRE_MAP)
    return df


# ============================================================
# 2. LLM: TRADUCIR DESCRIPCION A VALORES NUMERICOS
# ============================================================

def traducir_con_gemini(prompt):
    respuesta = client_gemini.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    texto = respuesta.text.replace("```json", "").replace("```", "").strip()
    return json.loads(texto)

def traducir_con_groq(prompt):
    respuesta = client_groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
    )
    texto = respuesta.choices[0].message.content
    texto = texto.replace("```json", "").replace("```", "").strip()
    return json.loads(texto)

def chat_bot(prompt):
    respuesta = client_groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
    )
    return respuesta.choices[0].message.content  # sin json.loads

def traducir_descripcion_a_features(descripcion_usuario):
    """
    Usa Gemini para convertir la descripcion del usuario
    en valores numericos para los features de Spotify.
    Retorna un diccionario con los valores: valence, energy, tempo, danceability, acousticness.
    """
    
    prompt = f"""
    Eres un experto en música y análisis de emociones musicales.

    Convierte la siguiente descripción musical a valores numéricos de Spotify.
    Responde ÚNICAMENTE con un JSON válido, sin texto adicional, sin markdown.

    Los valores deben estar en estos rangos exactos:
    - valence: 0 a 1 (0 = muy triste/oscuro, 1 = muy feliz/eufórico)
    - energy: 0 a 1 (0 = muy calmado, 1 = muy energético/explosivo)
    - tempo: 50 a 200 (BPM: 50 = muy lento,100 - 120 = moderado, 200 = muy rápido)
    - danceability: 0 a 1 (0 = no bailable, 1 = muy bailable)
    - acousticness: 0 a 1 (0 = electrónico/sintetizado, 1 = completamente acústico)
    - instrumentalness: 0 a 1 (0 = con voz/letra, 1 = completamente instrumental)
    - speechiness: 0 a 1 (0 = sin palabras habladas, 1 = todo hablado como podcast)
    - liveness: 0 a 1 (0 = estudio grabado, 1 = concierto en vivo)
    - loudness: -60 a 0 (decibelios: -60 = muy silencioso, 2 = muy fuerte)
    - duration_ms: 60000 a 600000 (duración en ms: 60000 = 1 min, 210000 = 3.5 min, 600000 = 10 min)

    Descripción del usuario: "{descripcion_usuario}"
    Ten en cuenta estas relaciones generales entre features:
    
    - Alta instrumentalness significa que la música tiene poco o ningún canto — guitarras, pianos, sintetizadores predominan
    - Alta speechiness indica rap, spoken word o podcasts — no simplemente canciones con letra
    - Loudness alto (cerca de 0) indica música fuerte y comprimida como rock o electrónica; bajo (cerca de -60) indica música suave o clásica
    - Alta acousticness indica instrumentos reales sin procesamiento electrónico: guitarra acústica, piano, violín
    - Alta liveness indica que suena a concierto: reverb de sala, público, imperfecciones en vivo
    Ejemplo de respuesta:
    {{"valence": 0.2, "energy": 0.3, "tempo": 80, "danceability": 0.2, 
    "acousticness": 0.8, "instrumentalness": 0.1, "speechiness": 0.05, 
    "liveness": 0.1, "loudness": -15, "duration_ms": 210000}}
    """
    
     # Primero intentamos con Groq
    try:
        features = traducir_con_groq(prompt)
        print("LLM usado: Groq")
        return features
    except Exception as e:
        print(f"Groq falló: {e} — intentando con Gemini...")

        # Fallback a Gemini
    try:
        features = traducir_con_gemini(prompt)
        print("LLM usado: Gemini")
        return features
    except Exception as e:
        print(f"Gemini falló: {e} — usando valores por defecto...")

   

    # Si los dos fallan, defaults
    return DEFAULTS.copy()


# ============================================================
# 3. CLASIFICADOR DE GENERO Y ESTADO DE ANIMO
# ============================================================
def entrenar_clasificadores(df):
    """
    Entrena dos clasificadores Random Forest:
    - Uno para predecir el género musical (genre_group)
    - Otro para predecir si es popular o no (popular)

    Retorna ambos modelos entrenados.
    """

    df = cargar_datos_spotify()

    # Definición de X y Y
    X = df[audio_features]
    y_genero = df['genre_group']
    y_popularidad = df['popular']


    # Primero separamos test (15%)
    X_temp, X_test, yp_temp, yp_test = train_test_split(
        X, y_popularidad, test_size=0.15, random_state=17, stratify=y_popularidad
    )
    _, _, yg_temp, yg_test = train_test_split(
        X, y_genero, test_size=0.15, random_state=17, stratify=y_genero
    )



    # Luego separamos train y val del restante 85%
    X_train, X_val, yp_train, yp_val = train_test_split(
        X_temp, yp_temp, test_size=0.176, random_state=17, stratify=yp_temp  # 15/85 ≈ 0.176 para obtener 15% del total
    )

    _, _, yg_train, yg_val = train_test_split(
        X_temp, yg_temp, test_size=0.176, random_state=17, stratify=yg_temp  
    )

    # Escalamos los features numericos


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) # Solo se ajusta el scaler con el set de entrenamiento para evitar data leakage
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")

    #RANDOM FOREST

    # Modelo de género
    model_genre = RandomForestClassifier(
        n_estimators=100, 
        random_state=42)
    model_genre.fit(X_train_scaled, yg_train)

    # Predicciones y reporte de clasificación para género

    #train
    pred_genre_train = model_genre.predict(X_train_scaled)
    print("Reporte de clasificación para Género (Train):")
    print(classification_report(
        yg_train,
        pred_genre_train
    ))
    # val
    pred_genre_val = model_genre.predict(X_val_scaled)
    print("Reporte de clasificación para Género (Val):")
    print(classification_report(
        yg_val,
        pred_genre_val
    ))
    # test 
    pred_genre_test = model_genre.predict(X_test_scaled)
    print("Reporte de clasificación para Género (Test):")
    print(classification_report(
        yg_test,
        pred_genre_test))


    # Modelo de popularidad
    model_popular = RandomForestClassifier(n_estimators=100, random_state=42)
    model_popular.fit(X_train_scaled, yp_train)


    # Predicciones y reporte de clasificación para popularidad

    #train
    pred_popular_train = model_popular.predict(X_train_scaled)
    print("Reporte de clasificación para Popularidad (Train):")
    print(classification_report(
        yp_train,
        pred_popular_train
    ))
    # val
    pred_popular_val = model_popular.predict(X_val_scaled)
    print("Reporte de clasificación para Popularidad (Val):")
    print(classification_report(
        yp_val,
        pred_popular_val
    ))
    # test 
    pred_popular_test = model_popular.predict(X_test_scaled)
    print("Reporte de clasificación para Popularidad (Test):")
    print(classification_report(
        yp_test,
        pred_popular_test))
    
    return scaler, model_genre, model_popular

# KNN

def recomendar_canciones(features_dict, df, n=5):
    """
    Dado un diccionario de features inferidos por el LLM,
    busca las n canciones más similares en el dataset usando KNN.
    """

    # Construir el vector de query con los features que el LLM infirió
    # Los que no infirió se rellenan con valores neutrales
    vector = np.array([[features_dict.get(f, DEFAULTS[f]) for f in audio_features]])

    # Escalar el vector con el mismo scaler del entrenamiento
    scaler = joblib.load("models/scaler.pkl")
    vector_scaled = scaler.transform(vector)

    # Escalar todo el dataset para la búsqueda
    X = df[audio_features]
    X_scaled = scaler.transform(X)

    # Entrenar KNN y buscar los más cercanos
    knn = NearestNeighbors(n_neighbors=n, metric="cosine")
    knn.fit(X_scaled)
    _, indices = knn.kneighbors(vector_scaled)

    # Retornar las canciones encontradas con sus columnas descriptivas
    resultados = df.iloc[indices[0]][
        ["track_name", "track_artist", "playlist_genre", 
         "genre_group", "track_popularity", "popular"] + audio_features
    ].reset_index(drop=True)

    return resultados


# pipeline

def pipeline_recomendacion(descripcion_usuario, df, scaler, model_genre, model_popular):
    features = traducir_descripcion_a_features(descripcion_usuario)
    vector = pd.DataFrame(
        [[features.get(f, DEFAULTS[f]) for f in audio_features]],
        columns=audio_features
    )
    vector_scaled = scaler.transform(vector)
    genero_predicho = model_genre.predict(vector_scaled)[0]
    es_popular = model_popular.predict(vector_scaled)[0]
    recomendaciones = recomendar_canciones(features, df)
    return {
        "features_inferidos": features,
        "genero_predicho": genero_predicho,
        "es_popular": "popular" if es_popular == 1 else "no popular",
        "recomendaciones": recomendaciones
    }


