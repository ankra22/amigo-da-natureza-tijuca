# app.py
import os, json, re
import streamlit as st
from dotenv import load_dotenv

from agents.router import route_intent
from agents.general_agent import GeneralAgent
from agents.trails_agent import TrailsAgent
from agents.weather_agent import forecast_days
from nlp.groq_client import chat as groq_chat

load_dotenv()

st.set_page_config(page_title="Amigo da Natureza - PNT", page_icon="🌳", layout="centered")

# ===== Helpers =====
def _strip_citations(text: str) -> str:
    text = re.sub(r"\[(?:Fonte|Trecho)\s*\d+\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\s*\d+\s*\]", "", text)
    return re.sub(r"\s{2,}", " ", text).strip()

def show_image(img_path):
    """Compatível com versões antigas e novas do Streamlit"""
    try:
        st.image(img_path, use_container_width=True)
    except TypeError:
        st.image(img_path, use_column_width=True)

# ===== Inicialização =====
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🌳 Amigo da Natureza — Parque Nacional da Tijuca")

# Carregar agentes
general = GeneralAgent()
trails  = TrailsAgent()

# Render histórico
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "images" in m:
            cols = st.columns(3)
            for i, im in enumerate(m["images"]):
                with cols[i % 3]:
                    show_image(im)

# Entrada de texto
prompt = st.chat_input("Pergunte sobre o Parque, trilhas ou clima…")
if not prompt:
    st.stop()

# Mostrar mensagem do usuário
st.session_state.messages.append({"role": "user", "content": prompt})
with st.chat_message("user"):
    st.markdown(prompt)

# Determinar intenção
intent = route_intent(prompt)

# ====== AGENTE CLIMA ======
if intent == "clima":
    data = forecast_days()
    if "error" in data:
        reply = "Configure a chave da WeatherAPI no `.env`."
    else:
        loc = data["location"]["name"]
        curr = data["current"]
        lines = [
            f"**Previsão para {loc}**",
            f"- Agora: {curr['temp_c']}°C, {curr['condition']['text']}",
            f"- Vento: {curr['wind_kph']} km/h | Umidade: {curr['humidity']}%",
            "---"
        ]
        for d in data["forecast"]["forecastday"]:
            day = d["date"]
            cond = d["day"]["condition"]["text"]
            lines.append(f"{day}: {d['day']['mintemp_c']}–{d['day']['maxtemp_c']}°C • {cond}")
        msg = [
            {"role": "system", "content": "Resuma a previsão em 3 bullets amigáveis, em PT-BR."},
            {"role": "user", "content": "\n".join(lines)}
        ]
        groq = groq_chat(msg, model="llama-3.3-70b-versatile")
        reply = groq.get("content", "") if isinstance(groq, dict) else groq

    with st.chat_message("assistant"):
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})

# ====== AGENTE TRILHAS ======
elif intent == "trilhas":
    txt, imgs = trails.answer(prompt)
    msg = [
        {"role": "system", "content": "Você é um guia de trilhas do PNT. Responda em PT-BR, com foco em segurança e orientação prática."},
        {"role": "user", "content": txt}
    ]
    groq = groq_chat(msg, model="llama-3.3-70b-versatile")
    reply = groq.get("content", "") if isinstance(groq, dict) else groq
    reply = _strip_citations(reply)

    with st.chat_message("assistant"):
        st.markdown(reply)
        if imgs:
            cols = st.columns(3)
            for i, im in enumerate(imgs):
                with cols[i % 3]:
                    show_image(im)
        else:
            # aviso curto + caminho que deve conter as imagens
            dbg = trails.last_debug
            st.warning(
                "Não encontrei imagens cadastradas para essa trilha.\n\n"
                f"Pasta esperada: `{dbg.get('folder_abs','(desconhecida)')}`"
            )

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "images": imgs
    })


# ====== AGENTE GERAL ======
else:
    ctx = general.retrieve(prompt, k=5)
    prompt_msgs = general.build_prompt(prompt, ctx)
    ans = groq_chat(prompt_msgs, model="llama-3.3-70b-versatile")
    reply = ans.get("content", "") if isinstance(ans, dict) else ans
    reply = _strip_citations(reply)

    with st.chat_message("assistant"):
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
