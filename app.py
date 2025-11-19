import io
from contextlib import redirect_stdout

import streamlit as st
from agente_orquestrador import OrquestradorAgentes


# ================== EXTRATOR DE RESPOSTA ================== #

def extrair_resposta(saida: str) -> str:
    """
    Recebe toda a saída que o orquestrador printou e devolve só a parte útil.

    - Se encontrar uma linha com 'resposta', devolve tudo que vem depois dela.
    - Se NÃO encontrar 'resposta', devolve a saída inteira.
    - Se dentro da resposta aparecer 'Buscando clima atual' ou
      'Buscando previsão do tempo', corta dali pra frente
      (para não anexar o clima em respostas gerais).
    - Para o agente de trilhas, também corta antes do menu de mapas e
      de linhas só com "=====" (separadores).
    """
    if not saida:
        return ""

    linhas = saida.splitlines()

    # Procura a primeira linha com 'resposta'
    idx_resp = None
    for i, linha in enumerate(linhas):
        if "resposta" in linha.lower():
            idx_resp = i
            break

    if idx_resp is None:
        # fallback: nada marcado, devolve tudo
        return saida.strip()

    # Pega tudo depois de 'Resposta:'
    resto = linhas[idx_resp + 1:]

    # Remove linhas vazias no começo
    while resto and not resto[0].strip():
        resto = resto[1:]

    # 🔪 Corta fora qualquer trecho de clima acoplado por engano
    # e também o menu de mapas / separadores do agente de trilhas
    corte_idx = None
    for j, linha in enumerate(resto):
        l = linha.lower().strip()

        # cortes para clima
        if "buscando clima atual" in l or "buscando previsão do tempo" in l:
            corte_idx = j
            break

        # corte para menu de mapas do agente de trilhas
        if "deseja visualizar algum mapa" in l:
            corte_idx = j
            break

        # corte para linha de separador "====="
        if l and set(l) == {"="} and len(l) >= 3:
            corte_idx = j
            break

    if corte_idx is not None:
        resto = resto[:corte_idx]

    return "\n".join(resto).strip()


# ================== CONFIG DA PÁGINA ================== #

st.set_page_config(
    page_title="Amigo da Natureza",
    page_icon="🌿",
    layout="centered",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 800px;
        padding-top: 2rem;
    }
    /* Deixa o texto do chat mais "normalzinho" */
    .stChatMessage p {
        font-size: 16px;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌿 Amigo da Natureza")


# ================== FUNÇÃO DE RENDERIZAÇÃO ================== #

def render_mensagem(conteudo: str):
    """Renderiza texto no chat com fonte normal e respeitando quebras de linha."""
    if not conteudo:
        return
    st.markdown(
        f"<div style='font-size:16px; line-height:1.6; white-space:pre-wrap;'>{conteudo}</div>",
        unsafe_allow_html=True,
    )


# ================== ESTADO GLOBAL ================== #

if "orquestrador" not in st.session_state:
    buf = io.StringIO()
    with redirect_stdout(buf):
        st.session_state["orquestrador"] = OrquestradorAgentes()
    # se quiser ver log de inicialização, está em init_log
    st.session_state["init_log"] = buf.getvalue()

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "Olá! Eu sou o guia virtual do Parque Nacional da Tijuca.\n\n"
                "Pode perguntar sobre clima, trilhas, regras do parque ou curiosidades."
            ),
        }
    ]


# ================== HISTÓRICO DE CHAT ================== #

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        render_mensagem(msg["content"])


# ================== INPUT DO USUÁRIO ================== #

pergunta = st.chat_input("Faça sua pergunta sobre o parque...")

if pergunta:
    # mostra mensagem do usuário
    st.session_state["messages"].append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        render_mensagem(pergunta)

    # balãozinho de "digitando"
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown(
            "<div style='font-size:16px; line-height:1.6;'>⌛ Conversando com os guardiões da floresta...</div>",
            unsafe_allow_html=True,
        )

        # captura TUDO o que o orquestrador imprimir
        buf = io.StringIO()
        with redirect_stdout(buf):
            st.session_state["orquestrador"].processar_pergunta(pergunta)

        bruto = buf.getvalue().strip()
        resposta_texto = extrair_resposta(bruto)

        if not resposta_texto:
            resposta_texto = (
                "❌ O orquestrador não retornou uma resposta clara.\n\n"
                "Verifique se os agentes foram inicializados corretamente."
            )

        # mostra resposta com estilo controlado
        placeholder.empty()
        render_mensagem(resposta_texto)

    # salva no histórico
    st.session_state["messages"].append(
        {"role": "assistant", "content": resposta_texto}
    )
