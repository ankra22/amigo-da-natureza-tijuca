import io
from contextlib import redirect_stdout

import streamlit as st

from agente_orquestrador import OrquestradorAgentes  # usa toda a estrutura que você já tem


# ============= ESTADO GLOBAL (SESSION STATE) =============

if "orquestrador" not in st.session_state:
    buf = io.StringIO()
    with redirect_stdout(buf):
        st.session_state["orquestrador"] = OrquestradorAgentes()
    st.session_state["init_log"] = buf.getvalue()

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "🌿 Olá! Eu sou o assistente do Parque Nacional da Tijuca. "
                       "Pode perguntar sobre clima, trilhas ou informações gerais!",
        }
    ]


# ============= CONFIG E LAYOUT =============

st.set_page_config(
    page_title="Amigo da Natureza - Tijuca",
    page_icon="🌿",
)

st.sidebar.title("Amigo da Natureza 🌿")
st.sidebar.markdown(
    """
Multiagente usando **a estrutura existente**:

- 🌦️ Clima e previsão  
- 🥾 Trilhas e mapas  
- 🌿 Informações gerais  

O `agente_orquestrador` decide automaticamente qual especialista responder.
"""
)

with st.sidebar.expander("🔧 Log de inicialização", expanded=False):
    st.text(st.session_state.get("init_log", "").strip() or "Sem logs.")


st.title("Parque Nacional da Tijuca – Assistente Inteligente 🤖🌿")

col1, col2 = st.columns(2)
with col1:
    if st.button("🧹 Limpar histórico da conversa"):
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "🧹 Histórico limpo! Pode mandar sua próxima pergunta. 🌿",
            }
        ]
        st.session_state["orquestrador"].limpar_historico()

with col2:
    st.markdown(
        "<div style='text-align:right; opacity:0.8;'>"
        "Agentes: clima • trilhas • geral"
        "</div>",
        unsafe_allow_html=True,
    )


# ============= MOSTRAR HISTÓRICO NO ESTILO CHATGPT =============

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ============= INPUT DO USUÁRIO =============

user_input = st.chat_input("Faça sua pergunta sobre o parque...")

if user_input:
    # 1) mostra mensagem do usuário
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) captura TUDO que o orquestrador imprime (classificação + resposta)
    buf = io.StringIO()
    with redirect_stdout(buf):
        st.session_state["orquestrador"].processar_pergunta(user_input)
    resposta_texto = buf.getvalue().strip()

    if not resposta_texto:
        resposta_texto = "❌ O orquestrador não retornou nenhuma saída visível. Veja se os agentes foram inicializados sem erro."

    # 3) mostra resposta da IA no chat
    with st.chat_message("assistant"):
        st.markdown(f"```text\n{resposta_texto}\n```")

    st.session_state["messages"].append(
        {"role": "assistant", "content": f"```text\n{resposta_texto}\n```"}
    )


