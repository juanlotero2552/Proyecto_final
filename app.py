import streamlit as st
import pandas as pd
from proyecto_final import pipeline_recomendacion, chat_bot, cargar_datos_spotify, entrenar_clasificadores, audio_features, DEFAULTS
# ============================================================
# CONFIGURACION DE LA PAGINA
# ============================================================

st.set_page_config(
    page_title="Recomendador Musical",
    page_icon="🎵",
    layout="centered"
)

# ============================================================
# CARGAR MODELOS 
# ============================================================
def cargar_todo():
    df = cargar_datos_spotify()
    scaler, model_genre, model_popular = entrenar_clasificadores(df)
    return df, scaler, model_genre, model_popular

with st.spinner("Cargando modelos..."):
    df, scaler, model_genre, model_popular = cargar_todo()





# ============================================================
# INTERFAZ PRINCIPAL
# ============================================================

st.title("🎵 Recomendador de Música con IA")
st.markdown("Describe qué quieres escuchar y el sistema encontrará las canciones perfectas para ti.")

st.divider()

# Input del usuario
descripcion = st.text_area(
    label="¿Qué quieres escuchar hoy?",
    placeholder="Ej: algo tranquilo para estudiar de noche, sin letra...",
    height=100
)    

col1, col2 = st.columns([1, 4])
with col1:
    buscar = st.button("🔍 Recomendar", use_container_width=True)

# ============================================================
# RESULTADOS
# Logica para la muestra de resultados
# ============================================================


if buscar:
    if not descripcion.strip():
        st.warning("Por favor escribe una descripción primero.")
    else:
        with st.spinner("Analizando tu descripción..."):
            resultado = pipeline_recomendacion(
                descripcion, df, scaler, model_genre, model_popular
            )
            resultado_completo = [resultado, descripcion]  # Guardar resultado completo para el bot
                    # GUARDAR CONTEXTO
        if "recomendaciones" not in st.session_state:
            st.session_state.recomendaciones = []
        st.session_state.recomendaciones.append(resultado_completo)  # Guardar resultado completo para el bot

        # Género y popularidad predichos
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🎸 Género detectado", resultado["genero_predicho"].upper())
        with col2:
            st.metric("📈 Perfil", resultado["es_popular"].upper())

        with st.expander("🔬 Features musicales inferidos"):
            features = resultado["features_inferidos"]
            
            # Normalizar cada feature al rango 0-1 para la barra de progreso
            rangos = {
                "energy": (0, 1), "tempo": (50, 200), "danceability": (0, 1),
                "loudness": (-60, 0), "valence": (0, 1), "speechiness": (0, 1),
                "instrumentalness": (0, 1), "acousticness": (0, 1),
                "liveness": (0, 1), "duration_ms": (60000, 600000)
            }

            col1, col2 = st.columns(2)
            items = list(features.items())
            mitad = len(items) // 2

            with col1:
                for k, v in items[:mitad]:
                    mn, mx = rangos.get(k, (0, 1))
                    normalizado = (float(v) - mn) / (mx - mn)
                    normalizado = max(0.0, min(1.0, normalizado))  # clamp por seguridad
                    st.progress(normalizado, text=f"{k}: {v}")

            with col2:
                for k, v in items[mitad:]:
                    mn, mx = rangos.get(k, (0, 1))
                    normalizado = (float(v) - mn) / (mx - mn)
                    normalizado = max(0.0, min(1.0, normalizado))
                    st.progress(normalizado, text=f"{k}: {v}")
        # Canciones recomendadas
        st.divider()
        st.subheader("🎶 Canciones recomendadas")
        recomendaciones = resultado["recomendaciones"]

        for i, row in recomendaciones.iterrows():
            with st.container(border=True):
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.markdown(f"**{row['track_name']}**")
                    st.caption(row["track_artist"])
                with col2:
                    st.caption(f"🎸 {row['playlist_genre']}  ·  {row['genre_group']}")
                with col3:
                    popular_label = "⭐ Popular" if row["popular"] == 1 else "💎 Indie"
                    st.caption(popular_label)
                    st.caption(f"🔥 {int(row['track_popularity'])}/100")

                # Features de la canción
                with st.expander("Ver características"):
                    rangos = {
                        "energy": (0, 1), "tempo": (50, 200), "danceability": (0, 1),
                        "loudness": (-60, 0), "valence": (0, 1), "speechiness": (0, 1),
                        "instrumentalness": (0, 1), "acousticness": (0, 1),
                        "liveness": (0, 1), "duration_ms": (60000, 600000)
                    }
                    col1, col2 = st.columns(2)
                    features_cancion = [(f, row[f]) for f in audio_features]
                    mitad = len(features_cancion) // 2
                    with col1:
                        for k, v in features_cancion[:mitad]:
                            mn, mx = rangos[k]
                            norm = max(0.0, min(1.0, (float(v) - mn) / (mx - mn)))
                            st.progress(norm, text=f"{k}: {round(v, 3)}")
                    with col2:
                        for k, v in features_cancion[mitad:]:
                            mn, mx = rangos[k]
                            norm = max(0.0, min(1.0, (float(v) - mn) / (mx - mn)))
                            st.progress(norm, text=f"{k}: {round(v, 3)}")



# ============================================================
# BOT DE APOYO 
# ============================================================

st.divider()
st.subheader("🤖 Asistente musical")

if "mensajes" not in st.session_state:
    st.session_state.mensajes = [
        {"role": "assistant", "content": "¡Hola! Soy tu asistente musical. Puedo explicarte los resultados, ayudarte a refinar tu búsqueda o responder preguntas sobre las canciones recomendadas."}
    ]

# Mostrar historial
for msg in st.session_state.mensajes:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input del chat
if pregunta := st.chat_input("Pregúntame algo sobre las recomendaciones..."):
    st.session_state.mensajes.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.write(pregunta)

    # Respuesta del bot
    contexto = st.session_state.mensajes
    contexto_recomendaciones = st.session_state.recomendaciones 
    

        
    prompt_bot = f"""
    Eres un asistente musical de un sistema de recomendación de canciones.
    Contexto actual de conversacion: {contexto} , contexto de recomendaciones: {contexto_recomendaciones}.  Solo habla relacionado a temas sobre musica.
    Responde de forma corta y útil en español.

    """
    try:
        respuesta_bot = chat_bot(prompt_bot)
    except Exception as e:
        respuesta_bot = f"En este momento no puedo responder, pero puedes refinar tu búsqueda escribiendo una descripción más detallada. Error: {e}"

    st.session_state.mensajes.append({"role": "assistant", "content": str(respuesta_bot)})
    with st.chat_message("assistant"):
        st.write(str(respuesta_bot))