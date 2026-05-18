# Recomendador de Música con IA
### Clasificación + NLP + Búsqueda por Similitud

**Asignatura:** Introducción a la Inteligencia Artificial  
**Institución:** EAFIT — 2026-1  
**Autores:** Miguel García · Juan Ignacio  

---

## 1. Planteamiento del Problema

Los sistemas de recomendación musical modernos, como los de Spotify o Apple Music, dependen en gran medida del historial de escucha del usuario. Esto representa una barrera significativa para usuarios nuevos o en contextos de sesión fría (*cold-start problem*): el sistema no tiene información suficiente para hacer sugerencias relevantes.

Además, incluso los usuarios con historial establecido no tienen una forma directa de expresar en lenguaje natural lo que quieren escuchar en un momento dado. Frases como *"algo tranquilo para estudiar de noche, sin letra"* o *"música bailable y feliz para una fiesta latina"* capturan de forma intuitiva una intención musical, pero los sistemas tradicionales no saben cómo procesarlas.

**Motivación:** Diseñar un sistema que permita al usuario describir libremente el tipo de música que desea, y que el sistema sea capaz de encontrar canciones que correspondan semánticamente a esa descripción, sin depender de historial previo.

---

## 2. Objetivo General

Desarrollar un sistema de recomendación musical basado en inteligencia artificial que permita al usuario describir en lenguaje natural el tipo de música que desea escuchar, y que, a partir de esa descripción, infiera características de audio, clasifique el perfil musical y recupere las canciones más similares disponibles en un dataset de Spotify.

---

## 3. Metodología

El sistema sigue un pipeline secuencial que integra procesamiento de lenguaje natural, clasificación supervisada y búsqueda por similitud:

```
Descripción del usuario
        │
        ▼
   LLM (Groq / Gemini)
   Traducción a features numéricos
        │
        ▼
   Extracción de Features
   (valence, energy, tempo, danceability, ...)
        │
        ▼
   Clasificador Random Forest
   → Género predicho
   → Popularidad predicha
        │
        ▼
   KNN con Similitud Coseno
   sobre el dataset escalado
        │
        ▼
   Top 5 Canciones Recomendadas
```

**NLP:** El modelo de lenguaje transforma texto libre del usuario en un vector numérico de 10 dimensiones, correspondiente a las audio features de Spotify.

**ML:** Un clasificador Random Forest predice el género y la popularidad. Un modelo KNN recupera las canciones más similares al vector inferido.

---

## 4. Desarrollo

### 4.1 Dataset y Procesamiento de Datos

**Origen de los datos:**  
El dataset proviene de Kaggle (`solomonameh/spotify-music-dataset`) y contiene dos archivos: `high_popularity_spotify_data.csv` y `low_popularity_spotify_data.csv`. Estos se descargan usando la librería `kagglehub`.

**Proceso de carga y limpieza (`cargar_datos_spotify`):**

1. Si el archivo `spotify_clean.csv` ya existe en disco, se carga directamente para evitar re-descargas.
2. Si no existe, se descargan ambos archivos de Kaggle y se etiquetan:
   - `popular = 1` para canciones de alta popularidad
   - `popular = 0` para canciones de baja popularidad
3. Se concatenan ambos DataFrames y se eliminan duplicados.
4. Se seleccionan únicamente las columnas relevantes: los 10 audio features, las columnas descriptivas (`track_name`, `track_artist`, `playlist_genre`, `playlist_subgenre`, `track_popularity`) y la etiqueta `popular`.
5. Se eliminan filas con valores nulos (`dropna()`).
6. Se aplica la función `agrupar_generos()` para reducir la dimensionalidad del campo de género.
7. El resultado se guarda como `spotify_clean.csv`.

**Dataset final:** 4.830 canciones limpias.

**Agrupación de géneros (`agrupar_generos`):**  
El dataset original contiene 35 géneros con distribución muy desigual. Se agruparon en 6 categorías balanceadas para mejorar la clasificación:

| Grupo | Géneros incluidos |
|-------|-------------------|
| `electronic` | electronic, gaming, lofi, disco |
| `pop` | pop, latin, k-pop, j-pop, indie, cantopop, mandopop |
| `urban` | hip-hop, r&b, funk, reggae, afrobeats, soca |
| `rock` | rock, metal, punk, blues, country |
| `calm` | ambient, classical, jazz, folk, soul, gospel, wellness |
| `world` | world, arabic, brazilian, turkish, indian |

**Audio features utilizadas:**

| Feature | Descripción | Rango |
|---------|-------------|-------|
| `valence` | Estado emocional (triste → feliz) | 0 – 1 |
| `energy` | Intensidad y actividad | 0 – 1 |
| `tempo` | Velocidad en BPM | 50 – 200 |
| `danceability` | Qué tan bailable | 0 – 1 |
| `acousticness` | Nivel acústico | 0 – 1 |
| `instrumentalness` | Presencia de voz vs. instrumental | 0 – 1 |
| `speechiness` | Presencia de habla o rap | 0 – 1 |
| `liveness` | Probabilidad de grabación en vivo | 0 – 1 |
| `loudness` | Volumen promedio (dB) | -60 – 0 |
| `duration_ms` | Duración en milisegundos | 60.000 – 600.000 |

---

### 4.2 Traducción de Lenguaje Natural a Features (LLM)

La función `traducir_descripcion_a_features()` construye un prompt detallado que incluye:

- Los rangos exactos de cada feature.
- Relaciones semánticas clave (ej. alta `instrumentalness` → sin canto; alta `speechiness` → rap/podcast).
- Un ejemplo de respuesta JSON.

El sistema intenta primero con **Groq** (modelo `llama-3.3-70b-versatile`, más rápido y económico). Si falla, hace fallback a **Gemini** (`gemini-2.0-flash`). Si ambos fallan, retorna valores por defecto definidos en el diccionario `DEFAULTS`.

La respuesta del LLM es un JSON con los 10 features numéricos, que se usa como entrada para los modelos de clasificación y para la búsqueda KNN.

---

### 4.3 Clasificadores (Random Forest)

La función `entrenar_clasificadores(df)` entrena dos modelos:

**División de datos:**
- 70% entrenamiento / 15% validación / 15% test
- Se usa estratificación en ambas variables objetivo para mantener proporciones de clase.

**Escalado:**  
Se aplica `StandardScaler` ajustado únicamente sobre el set de entrenamiento para evitar *data leakage*. El scaler se guarda en `models/scaler.pkl` para ser reutilizado durante la inferencia.

**Modelos:**
- `RandomForestClassifier(n_estimators=100, random_state=42)` para género (`genre_group`)
- `RandomForestClassifier(n_estimators=100, random_state=42)` para popularidad (`popular`)

---

### 4.4 Recomendador KNN

La función `recomendar_canciones(features_dict, df, n=5)`:

1. Construye un vector de query a partir del diccionario de features inferidos por el LLM (los no inferidos se rellenan con `DEFAULTS`).
2. Escala el vector con el mismo `scaler.pkl` del entrenamiento.
3. Escala todo el dataset.
4. Entrena un `NearestNeighbors(n_neighbors=5, metric="cosine")` sobre el dataset escalado.
5. Retorna las 5 canciones con mayor similitud coseno al vector del usuario.

---

### 4.5 Bot Asistente Musical

El sistema incluye un chatbot conversacional accesible desde la interfaz de Streamlit, bajo el encabezado **"🤖 Asistente musical"**.

**Acceso:** el usuario escribe en el campo `st.chat_input` ubicado en la parte inferior de la página. El bot responde directamente en el chat con contexto musical.

**Manejo del contexto con `st.session_state`:**  
Streamlit no mantiene estado entre reruns de forma automática. El sistema usa dos variables de sesión:

- `st.session_state.mensajes`: lista de diccionarios `{"role": ..., "content": ...}` con el historial completo de la conversación (usuario y asistente). Se inicializa con un mensaje de bienvenida del asistente.
- `st.session_state.recomendaciones`: lista acumulativa de resultados del pipeline (features inferidos, género predicho, popularidad y canciones recomendadas). Se actualiza cada vez que el usuario hace una búsqueda.

**Construcción del prompt del bot:**  
Cada vez que el usuario envía un mensaje, se construye un prompt que incluye:
- Todo el historial de la conversación (`contexto`)
- Todas las recomendaciones generadas en la sesión (`contexto_recomendaciones`)

Esto permite que el bot responda preguntas como *"¿cuál de las 5 canciones me recomiendas más?"* o *"¿por qué me salió música electrónica?"* con información real del sistema.

**Flujo de mensajes:**

```python
# 1. Usuario escribe → se agrega al historial
st.session_state.mensajes.append({"role": "user", "content": pregunta})

# 2. Se construye el prompt con todo el contexto
prompt_bot = f"""
Eres un asistente musical...
Contexto actual de conversacion: {contexto},
contexto de recomendaciones: {contexto_recomendaciones}.
"""

# 3. Se llama a chat_bot() → Groq LLM
respuesta_bot = chat_bot(prompt_bot)

# 4. Respuesta se agrega al historial
st.session_state.mensajes.append({"role": "assistant", "content": respuesta_bot})
```

El historial completo se renderiza en cada rerun usando un loop sobre `st.session_state.mensajes`, garantizando que la conversación sea coherente y acumulativa durante toda la sesión.

---

### 4.6 Interfaz (Streamlit)

La aplicación `app.py` presenta:
- Un campo de texto para la descripción del usuario.
- Botón de recomendación que dispara el pipeline completo.
- Métricas de género y popularidad predichos.
- Expansor con barras de progreso para cada feature inferido.
- Cards por canción con nombre, artista, género, popularidad y features propios.
- Sección de chat con el asistente musical al final.

---

### 4.7 Archivos entregados

| Archivo | Descripción |
|---------|-------------|
| `proyecto_final.py` | Lógica principal: carga de datos, LLM, clasificadores, KNN, pipeline |
| `app.py` | Interfaz Streamlit |
| `requirements.txt` | Dependencias del proyecto |
| `spotify_clean.csv` | Dataset limpio y procesado |

---

## 5. Resultados

### 5.1 Clasificador de Género (6 clases — baseline aleatorio ≈ 16%)

| Split | Accuracy | Precision (macro) | Recall (macro) |
|-------|----------|-------------------|----------------|
| Train | **96%** | 0.96 | 0.96 |
| Val | 22% | 0.18 | 0.18 |
| Test | **19%** | 0.14 | 0.15 |

### 5.2 Clasificador de Popularidad (binario — baseline ≈ 50%)

| Split | Accuracy | Precision (macro) | Recall (macro) |
|-------|----------|-------------------|----------------|
| Train | **99%** | 0.99 | 0.99 |
| Val | 73% | 0.71 | 0.69 |
| Test | **71%** | 0.70 | 0.71 |

---

### 5.3 Pruebas del Sistema

#### Ejemplo 1: "algo alocado"

<img width="781" alt="recomendacion1-descripcion" src="https://github.com/user-attachments/assets/c6a9f275-fc58-4e73-9dc4-54f16ff661c8" />

<img width="826" alt="recomendacion1-respuesta" src="https://github.com/user-attachments/assets/1f733dd3-0820-4b2f-9c98-040250ceaa7e" />

#### Ejemplo 2: "algo movido pero triste"

<img width="751" alt="recomendacion2-descripcion" src="https://github.com/user-attachments/assets/5a13f883-8649-4069-90a0-caa810179400" />

<img width="793" alt="recomendacion2-resultado" src="https://github.com/user-attachments/assets/e47f544b-888c-44cb-863d-44b81c37c57e" />

---

### 5.4 Pruebas del Asistente Musical

#### 1. Consulta del contexto completo

<img width="786" alt="muestra-contexto1" src="https://github.com/user-attachments/assets/deaba97c-4c3e-4095-878b-d5cea2a76fe5" />

#### 2. Evaluación de las recomendaciones

<img width="775" alt="muestra-contexto2" src="https://github.com/user-attachments/assets/986db896-46b9-40b6-addb-dff48ab6feb1" />

#### 3. Selección de una canción específica

<img width="738" alt="muestra-contexto3" src="https://github.com/user-attachments/assets/60e2c6ba-759b-4a9e-b055-f6ea022bf137" />

---

## 6. Discusión

### 6.1 Overfitting en el clasificador de género

El clasificador de género presentó overfitting severo: 96% en entrenamiento vs. 19% en test. Aunque el 19% supera marginalmente el baseline aleatorio (16%), la diferencia entre splits evidencia que el modelo memoriza patrones del conjunto de entrenamiento sin generalizar bien.

Esto se explica porque los audio features de alto nivel de Spotify —diseñados para ser interpretados por humanos— tienen límites para separar géneros musicalmente distintos. Por ejemplo, el `energy` de una canción de rock y una de electrónica puede ser idéntico, aunque pertenezcan a géneros completamente diferentes. Tzanetakis & Cook (2002) demostraron que la clasificación robusta de géneros requiere features espectrales de bajo nivel: coeficientes MFCC, centroide espectral y contenido rítmico extraídos directamente del audio crudo. Nuestro sistema, al depender exclusivamente de features precalculados por Spotify, no tiene acceso a esa información.

### 6.2 Clasificador de popularidad

El clasificador de popularidad mostró un comportamiento más robusto: 71% de accuracy en test. Al ser un problema binario, la generalización es más alcanzable, y los features de Spotify sí capturan diferencias relevantes entre canciones populares e independientes (tempo, energy, danceability).

### 6.3 Calidad del recomendador KNN

A pesar de la limitación del clasificador de género, el pipeline completo genera recomendaciones coherentes cualitativamente. Esto ocurre porque el KNN opera directamente en el espacio de features, sin depender de la etiqueta de género. La similitud coseno en 10 dimensiones es suficiente para recuperar canciones con características sonoras similares a la descripción del usuario.

### 6.4 Comparación con sistemas profesionales

| Fuente de información | Spotify | Nuestro sistema |
|-----------------------|---------|-----------------|
| Audio features | ✓ | ✓ |
| Comportamiento de usuarios | ✓ | ✗ |
| Metadata y playlists | ✓ | ✗ |
| Historial de escucha | ✓ | ✗ |
| Descripción en lenguaje natural | ✗ | ✓ |

Nuestro sistema aporta una ventaja diferencial concreta: permite la entrada en lenguaje natural, resolviendo el problema de *cold-start* para usuarios nuevos. La integración de un LLM como puente semántico es una contribución que los sistemas comerciales no ofrecen de forma directa.

---

## Referencias

Tzanetakis, G., & Cook, P. (2002). *Musical genre classification of audio signals*. IEEE Transactions on Speech and Audio Processing, 10(5), 293–302. https://doi.org/10.1109/TSA.2002.800560
