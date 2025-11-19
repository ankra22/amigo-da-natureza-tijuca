import chromadb
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from PIL import Image
import numpy as np
from io import BytesIO
import base64
from datetime import datetime

#Carregar variável de ambiente do arquivo .env
load_dotenv()

DB_FOLDER_TEXTO = r"C:\chroma\banco"
DB_FOLDER_IMAGENS = os.path.join(os.path.dirname(__file__), "Banco de dados imagens PDF")
COLLECTION_NAME_TEXTO = "PlanoManejo_Tijuca"
COLLECTION_NAME_IMAGENS = "Imagens_PDF_Collection"
TOP_K_TEXTO = 5
TOP_K_IMAGENS = 3  #buscar top 3 imagens mais relevantes

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("\n⚠️  AVISO: Verifique se a GROQ_API_KEY está configurada corretamente")
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


def inicializar_vectorstores():
    #Conecta aos bancos de dados ChromaDB (texto e imagens)

    #Vectorstore de texto
    if not os.path.exists(DB_FOLDER_TEXTO):
        raise FileNotFoundError(
            f"Banco de dados de texto não encontrado em: {DB_FOLDER_TEXTO}\n"
            "Execute primeiro o script de processamento dos PDFs."
        )

    vectorstore_texto = Chroma(
        collection_name=COLLECTION_NAME_TEXTO,
        embedding_function=embeddings,
        persist_directory=DB_FOLDER_TEXTO
    )

    #Vectorstore de imagens
    vectorstore_imagens = None
    if os.path.exists(DB_FOLDER_IMAGENS):
        try:
            client = chromadb.PersistentClient(path=DB_FOLDER_IMAGENS)
            collection_imagens = client.get_collection(name=COLLECTION_NAME_IMAGENS)
            vectorstore_imagens = collection_imagens
            print(f"✓ Banco de imagens carregado: {collection_imagens.count()} imagens disponíveis\n")
        except Exception as e:
            print(f"⚠️  Aviso: Não foi possível carregar banco de imagens: {e}\n")
    else:
        print(f"⚠️  Banco de imagens não encontrado em: {DB_FOLDER_IMAGENS}\n")

    return vectorstore_texto, vectorstore_imagens


def buscar_mapas_relevantes(vectorstore_imagens, query_text, top_k=TOP_K_IMAGENS):
    #Busca mapas/imagens relevantes usando filtros nos metadados
    if vectorstore_imagens is None:
        return []

    try:
        #Extrair palavras-chave da query
        keywords = query_text.lower().split()

        #Palavras-chave relacionadas a trilhas/locais comuns
        trilha_keywords = [
            'cascatinha', 'taunay', 'pico', 'tijuca', 'mirante',
            'mayrink', 'excelsior', 'imperador', 'conde', 'estrada',
            'caminho', 'trilha', 'vale', 'floresta', 'cachoeira'
        ]

        #Buscar TODOS os documentos primeiro
        results = vectorstore_imagens.get(
            include=['metadatas', 'documents']
        )

        if not results['ids']:
            return []

        #Filtrar e pontuar manualmente baseado em palavras-chave
        mapas_pontuados = []

        for i, (doc_id, metadata, document) in enumerate(zip(
                results['ids'],
                results['metadatas'],
                results['documents']
        )):
            arquivo = metadata.get('arquivo_pdf', '').lower()
            caminho = metadata.get('caminho_relativo', '').lower()
            doc_text = document.lower() if document else ''

            #Calcular score de relevância
            score = 0

            #Verificar palavras-chave da query
            for keyword in keywords:
                if len(keyword) > 2:  #Ignorar palavras muito curtas
                    if keyword in arquivo:
                        score += 3
                    if keyword in caminho:
                        score += 2
                    if keyword in doc_text:
                        score += 1

            #Bonus para palavras-chave de trilhas conhecidas
            for trilha_kw in trilha_keywords:
                if trilha_kw in arquivo or trilha_kw in caminho:
                    score += 1

            if score > 0:
                mapas_pontuados.append({
                    'id': doc_id,
                    'arquivo': metadata.get('arquivo_pdf', 'Desconhecido'),
                    'pagina': metadata.get('pagina', '?'),
                    'caminho': metadata.get('caminho_relativo', ''),
                    'dimensoes': metadata.get('dimensoes', ''),
                    'relevancia': score,
                    'metodo': metadata.get('metodo_extracao', 'desconhecido')
                })

        #Ordenar por relevância (score decrescente)
        mapas_pontuados.sort(key=lambda x: x['relevancia'], reverse=True)

        #retornar top K
        return mapas_pontuados[:top_k]

    except Exception as e:
        print(f"⚠️  Erro ao buscar mapas: {e}")
        import traceback
        traceback.print_exc()
        return []


def recuperar_imagem_do_banco(vectorstore_imagens, doc_id):
    #recupera a imagem armazenada no banco de dados ChromaDB
    try:
        #buscar o documento específico pelo ID
        result = vectorstore_imagens.get(
            ids=[doc_id],
            include=['embeddings', 'metadatas']
        )

        #validar resultado de forma segura
        if result is None:
            print("⚠️  Resultado vazio do banco de dados")
            return None, None

        if 'embeddings' not in result:
            print("⚠️  Sem embeddings no resultado")
            return None, None

        embeddings_list = result['embeddings']

        if embeddings_list is None or len(embeddings_list) == 0:
            print("⚠️  Lista de embeddings vazia")
            return None, None

        embedding = embeddings_list[0]

        if embedding is None:
            print("⚠️  Embedding é None")
            return None, None

        #converter para numpy array se necessário
        embedding_array = np.array(embedding)

        if embedding_array.size == 0:
            print("⚠️  Embedding vazio")
            return None, None

        metadata = result['metadatas'][0] if result.get('metadatas') and len(result['metadatas']) > 0 else {}

        #detectar tamanho do embedding e reconstruir adequadamente
        print(f"   📊 Tamanho do embedding: {embedding_array.size}")

        #suportar múltiplas resoluções
        if embedding_array.size == 602112:  # 448x448x3 (ALTA QUALIDADE)
            img_array = embedding_array.reshape(448, 448, 3)
            print(f"   ✨ Resolução detectada: 448x448 (Alta Qualidade)")
        elif embedding_array.size == 150528:  # 224x224x3 (QUALIDADE PADRÃO)
            img_array = embedding_array.reshape(224, 224, 3)
            print(f"   📐 Resolução detectada: 224x224 (Qualidade Padrão)")
        else:
            print(f"   ⚠️  Tamanho não reconhecido: {embedding_array.size}")
            #tentar descobrir dimensões quadradas
            total_pixels = embedding_array.size // 3
            lado = int(np.sqrt(total_pixels))
            if lado * lado * 3 == embedding_array.size:
                img_array = embedding_array.reshape(lado, lado, 3)
                print(f"   🔧 Tentando reconstruir como {lado}x{lado}")
            else:
                print("   ❌ Não foi possível determinar as dimensões da imagem")
                return None, None

        #desnormalizar (multiplicar por 255)
        img_array = (img_array * 255).astype(np.uint8)

        # Converter para PIL Image
        img = Image.fromarray(img_array, mode='RGB')

        return img, metadata

    except Exception as e:
        print(f"❌ Erro ao recuperar imagem: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def exibir_mapa_do_banco(vectorstore_imagens, mapa_info):
    """Extrai e exibe a imagem diretamente do banco de dados"""
    try:
        print(f"\n📍 Recuperando mapa do banco de dados...")

        # Recuperar imagem do banco
        img, metadata = recuperar_imagem_do_banco(vectorstore_imagens, mapa_info['id'])

        if img is None:
            print("❌ Não foi possível recuperar a imagem")
            return None

        # Informações do mapa
        print(f"\n🗺️  MAPA: {mapa_info['arquivo']}")
        print(f"   📄 Página: {mapa_info['pagina']}")
        print(f"   📐 Dimensões originais: {mapa_info['dimensoes']}")
        print(f"   📊 Dimensões recuperadas: {img.size[0]}x{img.size[1]} pixels")
        print(f"   🔧 Método de extração: {mapa_info.get('metodo', 'desconhecido')}")

        # Salvar em ALTA QUALIDADE
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nome_base = mapa_info['arquivo'].replace('.pdf', '').replace(' ', '_')
        filename = f"mapa_{nome_base}_p{mapa_info['pagina']}_{timestamp}.png"
        output_path = os.path.join(os.path.dirname(__file__), filename)

        # Se a imagem for 448x448, fazer upscale 3x para 1344x1344
        # Se for 224x224, fazer upscale 6x para 1344x1344
        if img.size[0] == 448:
            scale_factor = 3
        elif img.size[0] == 224:
            scale_factor = 6
        else:
            scale_factor = 3

        # Upscale usando LANCZOS para máxima qualidade
        new_size = (img.size[0] * scale_factor, img.size[1] * scale_factor)
        img_display = img.resize(new_size, Image.Resampling.LANCZOS)

        # Salvar com qualidade máxima
        img_display.save(output_path, quality=100, optimize=False)

        print(
            f"   🎨 Upscale aplicado: {scale_factor}x ({img.size[0]}x{img.size[1]} → {img_display.size[0]}x{img_display.size[1]})")
        print(f"   💾 Mapa salvo em: {output_path}\n")

        # Tentar abrir a imagem automaticamente
        try:
            img_display.show()
            print("   ✓ Mapa aberto no visualizador de imagens\n")
        except Exception as e:
            print(f"   ⚠️  Não foi possível abrir automaticamente. Abra manualmente: {output_path}\n")

        return output_path

    except Exception as e:
        print(f"❌ Erro ao exibir mapa: {e}")
        import traceback
        traceback.print_exc()
        return None


def criar_prompt_template():
    template = """Você é um guia especializado em trilhas do Parque Nacional da Tijuca.

Use uma linguagem clara, direta e acolhedora, mas sem vícios de linguagem.
Evite expressões como "olá, olha só", "sabe o que é incrível?", "vem comigo que eu te conto"
ou qualquer outra muleta de linguagem repetitiva.

ESTILO DE RESPOSTA:
- Comece com um parágrafo resumindo a trilha: tipo de caminhada, nível de esforço e para quem ela é indicada.
- Em seguida, SE houver informação nos documentos, você pode apresentar um pequeno bloco com dados técnicos,
  como distância aproximada, tempo médio de caminhada, dificuldade e principais cuidados.
- Se algum desses dados não aparecer claramente nos documentos, NÃO invente valores e NÃO crie linhas do tipo
  "Distância aproximada: não informada nos documentos.".
  Em vez disso, explique em texto normal que essa informação não está detalhada nos documentos.
- Use no máximo um bloco simples de lista, por exemplo:

  Distância aproximada: ...
  Tempo médio de caminhada: ...
  Dificuldade: ...
  Principais cuidados: ...

  Somente quando fizer sentido e os dados existirem.
- Evite repetir as mesmas ideias em várias frases diferentes.
- Não use emojis nas respostas.

VOCÊ PODE FALAR SOBRE:
- Trilhas específicas (como a Trilha do Pico da Tijuca) e suas características principais.
- Grau de dificuldade (leve, moderada, pesada) e para qual tipo de visitante é indicada.
- Tempo médio de ida e volta.
- Riscos mais relevantes (trechos íngremes, escorregadios, necessidade de atenção redobrada etc.).
- Recomendações práticas: água, calçado adequado, horário recomendável e itens básicos de segurança.

REGRAS IMPORTANTES:
1. Baseie TUDO no contexto dos documentos fornecidos – você é um guia responsável.
2. Se não tiver informação suficiente sobre uma trilha específica, seja honesto e ofereça alternativas
   ou informações gerais de segurança.
3. Considere o histórico da conversa – você lembra do que já conversaram.
4. Quando citar documentos, faça isso de forma natural, como:
   "De acordo com o plano de manejo do parque..." ou "Nos documentos oficiais do parque é indicado que...".
5. Explique termos técnicos de forma simples e objetiva.
6. NUNCA invente informações sobre trilhas, distâncias ou tempo de caminhada.

CONTEXTO DOS DOCUMENTOS:
{context}

PERGUNTA DO VISITANTE: {question}"""

    return ChatPromptTemplate.from_messages([
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])


def criar_chain_rag(vectorstore_texto, vectorstore_imagens):
    """Cria a chain RAG com suporte a texto e busca de mapas"""

    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    from langchain_core.output_parsers import StrOutputParser

    # Criar retriever de texto
    retriever = vectorstore_texto.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_TEXTO}
    )

    # Criar prompt
    prompt = criar_prompt_template()

    # Função para formatar os documentos
    def format_docs(docs):
        return "\n\n".join([
            f"[Fonte: {doc.metadata.get('arquivo', 'Desconhecido')} - Parte {doc.metadata.get('parte', '?')}]\n{doc.page_content}"
            for doc in docs
        ])

    # Função para buscar e formatar info de mapas
    def buscar_info_mapas(query):
        if vectorstore_imagens is None:
            return "Nenhum mapa disponível no momento."

        mapas = buscar_mapas_relevantes(vectorstore_imagens, query)

        if not mapas:
            return "Nenhum mapa específico encontrado para esta consulta."

        info = "📍 MAPAS DISPONÍVEIS:\n"
        for i, mapa in enumerate(mapas, 1):
            info += f"{i}. {mapa['arquivo']} (Página {mapa['pagina']}) - Score: {mapa['relevancia']}\n"

        return info

    # Criar chain usando LCEL com busca de mapas
    retrieval_chain = (
            {
                "context": lambda x: format_docs(retriever.invoke(x["question"])),
                "question": lambda x: x["question"],
                "chat_history": lambda x: x.get("chat_history", []),
                "mapas_info": lambda x: buscar_info_mapas(x["question"])
            }
            | prompt
            | llm
            | StrOutputParser()
    )

    return retrieval_chain, retriever, vectorstore_imagens


def processar_pergunta_com_mapas(chain_tuple, pergunta, chat_history=None):
    """Processa uma pergunta e busca mapas relacionados"""

    chain, retriever, vectorstore_imagens = chain_tuple

    if chat_history is None:
        chat_history = []

    print(f"\n🔍 Pergunta: {pergunta}\n")
    print("💭 Pensando e buscando informações...\n")

    try:
        # Buscar documentos relevantes
        documentos = retriever.invoke(pergunta)

        # Buscar mapas relevantes
        mapas = []
        if vectorstore_imagens:
            mapas = buscar_mapas_relevantes(vectorstore_imagens, pergunta)

        # Mostrar fontes de texto
        if documentos:
            fontes = set()
            for doc in documentos:
                if hasattr(doc, 'metadata') and 'arquivo' in doc.metadata:
                    fontes.add(doc.metadata['arquivo'])

            if fontes:
                print(f"📄 Fontes consultadas: {', '.join(fontes)}")
                print(f"   Total de trechos analisados: {len(documentos)}")

        # Mostrar mapas encontrados
        if mapas:
            print(f"\n🗺️  Mapas encontrados: {len(mapas)}")
            for i, mapa in enumerate(mapas, 1):
                print(f"   {i}. {mapa['arquivo']} (Página {mapa['pagina']}) - Score: {mapa['relevancia']}")

        # Executar a chain
        resposta = chain.invoke({
            "question": pergunta,
            "chat_history": chat_history
        })

        # Exibir resposta
        print(f"\n{'=' * 70}")
        print("💬 RESPOSTA:\n")
        print(resposta)
        print(f"{'=' * 70}\n")

        # Perguntar se deseja ver o mapa (modo interativo em terminal)
        if mapas:
            print("🗺️  Deseja visualizar algum mapa? (digite o número ou 'não')")
            for i, mapa in enumerate(mapas, 1):
                print(f"   [{i}] {mapa['arquivo']} - Página {mapa['pagina']}")

            escolha = input("\n   Escolha: ").strip()

            if escolha.isdigit() and 1 <= int(escolha) <= len(mapas):
                mapa_escolhido = mapas[int(escolha) - 1]
                exibir_mapa_do_banco(vectorstore_imagens, mapa_escolhido)

        # Atualizar histórico
        chat_history.append(HumanMessage(content=pergunta))
        chat_history.append(AIMessage(content=resposta))

        return resposta, documentos, mapas, chat_history

    except Exception as e:
        print(f"❌ Erro ao processar pergunta: {e}")
        import traceback
        traceback.print_exc()
        return None, [], [], chat_history


def modo_interativo():
    """Modo interativo para fazer múltiplas perguntas com memória e visualização de mapas"""

    print("\n" + "=" * 70)
    print("🌿 GUIA DE TRILHAS DO PARQUE NACIONAL DA TIJUCA 🗺️")
    print("=" * 70)
    print("\nOlá! Sou seu guia especialista em trilhas do Parque Nacional da Tijuca!")
    print("Posso te ajudar com:")
    print("  • Informações detalhadas sobre trilhas")
    print("  • Visualização de mapas das trilhas")
    print("  • Pontos de interesse e atrações")
    print("  • Dicas de segurança e melhores horários")
    print("  • Flora e fauna ao longo dos caminhos")

    # Inicializar vectorstores e chain
    try:
        vectorstore_texto, vectorstore_imagens = inicializar_vectorstores()
        chain_tuple = criar_chain_rag(vectorstore_texto, vectorstore_imagens)
    except Exception as e:
        print(f"\n❌ {e}")
        import traceback
        traceback.print_exc()
        return

    # Inicializar histórico de conversa
    chat_history = []

    print("\n" + "=" * 70)
    print("💡 COMANDOS DISPONÍVEIS:")
    print("  • Digite sua pergunta normalmente")
    print("  • 'sair' - encerrar o programa")
    print("  • 'limpar' - resetar histórico da conversa")
    print("=" * 70 + "\n")

    while True:
        try:
            pergunta = input("🌿 Sua pergunta: ").strip()

            if not pergunta:
                continue

            if pergunta.lower() in ['sair', 'exit', 'quit']:
                print("\n" + "=" * 70)
                print("👋 Obrigado por usar o Guia de Trilhas do Parque Nacional da Tijuca!")
                print("   Aproveite sua aventura na natureza! 🌿🏞️")
                print("=" * 70 + "\n")
                break

            if pergunta.lower() in ['limpar', 'clear', 'reset']:
                chat_history = []
                print("\n🗑️  Histórico de conversa limpo!\n")
                continue

            # Processar pergunta com mapas
            resposta, docs, mapas, chat_history = processar_pergunta_com_mapas(
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
    modo_interativo()
