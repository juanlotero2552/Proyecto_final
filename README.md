# Music Mood Recommender

## Planteamiento del problema

Las plataformas de música recomiendan canciones basándose en el historial de escucha, pero no permiten describir en lenguaje natural cómo se siente el usuario.

**Pregunta:** ¿Cómo recomendar canciones basándose en una descripción textual del estado de ánimo?

**Ejemplo:** Si el usuario escribe "algo triste y lento para una noche de lluvia", el sistema debe devolver canciones con valence baja (triste), tempo lento y energía baja.

## Objetivo general

Construir un sistema que:
1. Reciba una descripción en lenguaje natural del estado de ánimo
2. Traduzca esa descripción a características musicales usando Gemini
3. Clasifique el género y mood aproximado
4. Filtre el catálogo de Spotify a ese género
5. Use KNN para encontrar las canciones más cercanas
6. Devuelva las 5 mejores recomendaciones en una interfaz web

## Diagrama de flujo

```mermaid
graph TD
    A[Usuario describe lo que quiere] --> B[LLM traduce a features numéricos]
    B --> C[Clasificador predice género y mood]
    C --> D[KNN busca en subconjunto filtrado]
    D --> E[Top 5 canciones recomendadas]

### **5. Metodología** (explicar las 4 etapas)

```markdown
## Metodología

### Etapa 1: LLM (Gemini)
Convierte frases como "algo triste y lento" en valores numéricos:
- valence = 0.2 (triste)
- energy = 0.3 (baja energía)
- tempo = 80 (lento)

### Etapa 2: Clasificador
Predice el género musical y el estado de ánimo (sad, neutral, happy)

### Etapa 3: Filtrado
Reduce el dataset de ~20,000 canciones a solo las de ese género (~2,000 canciones)

### Etapa 4: KNN con similitud de coseno
Encuentra las 5 canciones más parecidas a la descripción del usuario

## Dataset

**Fuente:** Spotify Music Dataset (Kaggle) - solomonameh/spotify-music-dataset

**Registros:** ~20,000 canciones

**Columnas seleccionadas:**
| Columna | Qué significa |
|---------|---------------|
| valence | Qué tan triste (0) o feliz (1) es la canción |
| energy | Qué tan calmada (0) o enérgica (1) es |
| tempo | Velocidad en BPM (60=lento, 120=normal, 200=rápido) |
| track_name | Nombre de la canción |
| track_artist | Artista |
| playlist_genre | Género musical (indie, pop, rock, etc.) |

## Resultados

| Componente | Métrica | Valor |
|------------|---------|-------|
| Clasificador de género | Accuracy | (por definir - lo pone tu compañero cuando termine) |
| KNN | Precisión en recomendaciones | (por definir después de pruebas manuales) |

### Ejemplo de salida

**Usuario:** "algo triste y lento para una noche de lluvia"

| # | Canción | Artista | Género |
|---|---------|---------|--------|
| 1 | Skinny Love | Bon Iver | indie folk |
| 2 | Holocene | Bon Iver | indie folk |
| 3 | Roslyn | Bon Iver | indie folk |
| 4 | The Night We Met | Lord Huron | indie folk |
| 5 | To Build A Home | The Cinematic Orchestra | chamber folk |

## Discusión

### Comparación con Spotify

| Sistema | Fortaleza | Debilidad |
|---------|-----------|-----------|
| Spotify | Basado en historial del usuario | No acepta descripciones en lenguaje natural |
| Nuestro sistema | Acepta lenguaje natural | Más lento, depende del LLM |

### Limitaciones

1. El dataset tiene solo ~20,000 canciones
2. El clasificador puede tener errores
3. Gemini puede malinterpretar descripciones ambiguas

### Trabajo futuro

1. Agregar más canciones al dataset
2. Permitir feedback del usuario para mejorar recomendaciones

## Cómo ejecutar

### Instalar dependencias
```bash
pip install kagglehub pandas scikit-learn google-generativeai gradio python-dotenv


---

### **10. Integrantes**

```markdown
## Integrantes

- Miguel Garcia 
- Juan Ignacio Lotero Franco

---

