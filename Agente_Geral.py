import chromadb
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

#Carregar variável de ambiente do arquivo .env
load_dotenv()

DB_FOLDER = os.path.join(os.path.dirname(__file__), "Banco de dados")
COLLECTION_NAME = "PlanoManejo_Tijuca"
TOP_K = 5  #Número de chunks mais relevantes a recuperar

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("\n⚠️  AVISO: Verifique se a GROQ_API_KEY esá configurada corretamente")
    exit(1)

#Inicializar LLM do Groq
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=2000
)

#Inicializar embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)


def inicializar_vectorstore():
    #Conecta ao banco de dados ChromaDB existente usando LangChain
    if not os.path.exists(DB_FOLDER):
        raise FileNotFoundError(
            f"Banco de dados não encontrado em: {DB_FOLDER}\n"
            "Execute primeiro o script de processamento dos PDFs."
        )

    try:
        #Conectar ao ChromaDB persistente via LangChain
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=DB_FOLDER
        )

        #Verificar quantidade de documentos
        collection = vectorstore._collection
        total_docs = collection.count()

        return vectorstore

    except Exception as e:
        raise Exception(f"Erro ao acessar coleção: {e}")


def criar_prompt_template():
    #Cria o template de prompt para o agente com suporte a histórico

    template = """Você é um guia experiente e apaixonado do Parque Nacional da Tijuca no Rio de Janeiro! 

Você ADORA compartilhar curiosidades sobre o parque e tem um jeito descontraído e entusiasmado de falar. Você usa uma linguagem amigável, às vezes faz piadas leves sobre a natureza, e sempre tenta deixar as pessoas empolgadas com o que estão aprendendo.

SOBRE O SEU ESTILO:
- Fale como se estivesse guiando as pessoas pelas trilhas do parque
- Use expressões como "olha só", "sabe o que é incrível?", "vem comigo que eu te conto"
- Conte histórias e curiosidades de forma envolvente
- Seja entusiasta: "cara, isso é demais!", "você não vai acreditar!"
- Use emojis ocasionalmente para dar mais vida: 🌿🦜🐒🏞️
- Mantenha um tom caloroso e acolhedor, como um carioca receptivo

SUAS ÁREAS DE EXPERTISE:
- Fauna (animais, comportamentos curiosos, onde avistar)
- Flora (plantas medicinais, árvores centenárias, flores raras)
- Trilhas e pontos turísticos (mirantes, cachoeiras, vistas)
- História do parque (reflorestamento, curiosidades)
- Dicas práticas (melhor horário, o que levar, cuidados)

REGRAS IMPORTANTES:
1. Baseie TUDO no contexto dos documentos fornecidos - você é um guia responsável!
2. Se não souber algo, diga: "Poxa, não possuo nenhuma informção a respeito disso, mas posso te falar sobre..."
3. Considere o histórico da conversa - você lembra do que já conversaram!
4. Cite as fontes de forma natural: "segundo o plano de manejo do parque..."
5. Explique termos técnicos de forma simples e divertida
6. NUNCA invente informações - sua credibilidade como guia é importante!

CONTEXTO DOS DOCUMENTOS:
{context}

PERGUNTA DO VISITANTE: {question}"""

    return ChatPromptTemplate.from_messages([
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])


def criar_chain_rag(vectorstore):
    #Cria a chain RAG (Retrieval-Augmented Generation) com LangChain e memória

    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    from langchain_core.output_parsers import StrOutputParser

    #Criar retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    #Criar prompt
    prompt = criar_prompt_template()

    #Função para formatar os documentos
    def format_docs(docs):
        return "\n\n".join([
            f"[Fonte: {doc.metadata.get('arquivo', 'Desconhecido')} - Parte {doc.metadata.get('parte', '?')}]\n{doc.page_content}"
            for doc in docs
        ])

    #Criar chain usando LCEL (LangChain Expression Language) com histórico
    retrieval_chain = (
            {
                "context": lambda x: format_docs(retriever.invoke(x["question"])),
                "question": lambda x: x["question"],
                "chat_history": lambda x: x.get("chat_history", [])
            }
            | prompt
            | llm
            | StrOutputParser()
    )

    return retrieval_chain, retriever


def processar_pergunta_langchain(chain_tuple, pergunta, chat_history=None):
    #processa uma pergunta usando a chain do LangChain com histórico de conversa

    chain, retriever = chain_tuple

    if chat_history is None:
        chat_history = []

    print(f"Pergunta: {pergunta}\n")

    #print("💭 Pensando...\n")

    try:
        #Buscar documentos relevantes primeiro (para mostrar fontes)
        documentos = retriever.invoke(pergunta)

        #Mostrar fontes
        if documentos:
            fontes = set()
            for doc in documentos:
                if hasattr(doc, 'metadata') and 'arquivo' in doc.metadata:
                    fontes.add(doc.metadata['arquivo'])

            #if fontes:
                #print(f"Fontes consultadas: {', '.join(fontes)}")
                #print(f"Total de trechos analisados: {len(documentos)}\n")

        #Executar a chain para gerar resposta com histórico
        resposta = chain.invoke({
            "question": pergunta,
            "chat_history": chat_history
        })

        #Exibir resposta
        print("Resposta:\n")
        print(resposta)
        #qubra de linha
        print()

        #Atualizar histórico
        chat_history.append(HumanMessage(content=pergunta))
        chat_history.append(AIMessage(content=resposta))

        return resposta, documentos, chat_history

    except Exception as e:
        print(f"❌ Erro ao processar pergunta: {e}")
        import traceback
        traceback.print_exc()
        return None, [], chat_history


def modo_interativo():
   #Modo interativo para fazer múltiplas perguntas com memória de conversa
    print("=" * 70)
    print("🌿 Olá! Eu sou o guia virtual do Parque Nacional da Tijuca, pronto para te ajudar a explorar a maior floresta urbana replantada do mundo.")
    print("=" * 70)
    print("Posso te ajudar com:\n")
    print("  • Informações detalhadas sobre fauna e flora")
    print("  • Informações detalhadas sobre regras do parque")

    #Inicializar vectorstore e chain
    try:
        vectorstore = inicializar_vectorstore()
        chain_tuple = criar_chain_rag(vectorstore)
    except Exception as e:
        print(f"❌ {e}")
        import traceback
        traceback.print_exc()
        return

    #Inicializar histórico de conversa
    chat_history = []

    print("\n" + "=" * 70)
    print(" COMANDOS DISPONÍVEIS:")
    print("  • Digite sua pergunta normalmente")
    print("  • 'sair' - encerrar o programa")
    print("  • 'limpar' - resetar histórico da conversa")
    print("=" * 70 + "\n")

    while True:
        try:
            pergunta = input("🌿Sua pergunta: ").strip()

            if not pergunta:
                continue

            if pergunta.lower() in ['sair', 'Sair']:
                print("\n" + "=" * 70)
                print("👋 Obrigado por usar o Guia do Parque Nacional da Tijuca!")
                print("   Aproveite sua aventura na natureza! 🌿🏞️")
                print("=" * 70 + "\n")
                break

            if pergunta.lower() in ['limpar', 'Limpar']:
                chat_history = []
                print("\n🗑️ Histórico de conversa limpo!\n")
                continue

            #Processar pergunta com histórico
            resposta, docs, chat_history = processar_pergunta_langchain(
                chain_tuple,
                pergunta,
                chat_history
            )

        except KeyboardInterrupt:
            print("\n\n👋 Até logo!\n")
            break
        except Exception as e:
            print(f"\n❌ Erro: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    #Executar modo interativo
    modo_interativo()