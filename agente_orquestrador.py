import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# Importar os agentes especializados
# Nota: Ajuste os nomes dos arquivos conforme necessário
import sys
import importlib.util


def importar_modulo(caminho_arquivo, nome_modulo):
    """Importa um módulo Python a partir de um caminho de arquivo"""
    try:
        spec = importlib.util.spec_from_file_location(nome_modulo, caminho_arquivo)
        if spec and spec.loader:
            modulo = importlib.util.module_from_spec(spec)
            sys.modules[nome_modulo] = modulo
            spec.loader.exec_module(modulo)
            return modulo
        return None
    except Exception as e:
        print(f"⚠️  Erro ao importar {nome_modulo}: {e}")
        return None


# Tentar importar agente de clima
agente_clima = None
CLIMA_DISPONIVEL = False
for nome_possivel in ['agente_clima.py']:
    caminho = os.path.join(os.path.dirname(__file__), nome_possivel)
    if os.path.exists(caminho):
        agente_clima = importar_modulo(caminho, 'agente_clima')
        if agente_clima and hasattr(agente_clima, 'buscar_clima_atual'):
            CLIMA_DISPONIVEL = True
            print(f"✓ Módulo de clima carregado: {nome_possivel}")
            break

# Tentar importar agente RAG
agente_rag = None
RAG_DISPONIVEL = False
for nome_possivel in ['agente_geral.py']:
    caminho = os.path.join(os.path.dirname(__file__), nome_possivel)
    if os.path.exists(caminho):
        agente_rag = importar_modulo(caminho, 'agente_geral')
        if agente_rag and hasattr(agente_rag, 'inicializar_vectorstore'):
            RAG_DISPONIVEL = True
            print(f"✓ Módulo RAG carregado: {nome_possivel}")
            break

# Tentar importar agente de trilhas
agente_trilhas = None
TRILHAS_DISPONIVEL = False
for nome_possivel in ['agente_trilhas.py']:
    caminho = os.path.join(os.path.dirname(__file__), nome_possivel)
    if os.path.exists(caminho):
        agente_trilhas = importar_modulo(caminho, 'agente_trilhas')
        if agente_trilhas and hasattr(agente_trilhas, 'inicializar_vectorstores'):
            TRILHAS_DISPONIVEL = True
            print(f"✓ Módulo de trilhas carregado: {nome_possivel}")
            break

print(f"\n📊 Status dos agentes:")
print(f"   🌦️  Clima: {'Disponível ✓' if CLIMA_DISPONIVEL else 'Indisponível ✗'}")
print(f"   🌿 Informações Gerais: {'Disponível ✓' if RAG_DISPONIVEL else 'Indisponível ✗'}")
print(f"   🗺️  Trilhas e Mapas: {'Disponível ✓' if TRILHAS_DISPONIVEL else 'Indisponível ✗'}")
print()

load_dotenv()

# Configurar LLM para classificação
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("\n⚠️  AVISO: Verifique se a GROQ_API_KEY está configurada corretamente")
    exit(1)

llm_classificador = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=100
)


class OrquestradorAgentes:
    """
    Orquestrador inteligente que analisa perguntas e direciona para o agente apropriado
    """

    def __init__(self):
        self.chat_history = []
        self.agentes_inicializados = {}

        # Inicializar agentes disponíveis
        print("\n🔧 Inicializando agentes especializados...\n")

        if RAG_DISPONIVEL:
            try:
                vectorstore = agente_rag.inicializar_vectorstore()
                self.agentes_inicializados['rag'] = agente_rag.criar_chain_rag(vectorstore)
                print("✓ Agente de Informações Gerais (RAG) inicializado")
            except Exception as e:
                print(f"✗ Erro ao inicializar agente RAG: {e}")

        if TRILHAS_DISPONIVEL:
            try:
                vectorstore_texto, vectorstore_imagens = agente_trilhas.inicializar_vectorstores()
                self.agentes_inicializados['trilhas'] = agente_trilhas.criar_chain_rag(
                    vectorstore_texto,
                    vectorstore_imagens
                )
                print("✓ Agente de Trilhas e Mapas inicializado")
            except Exception as e:
                print(f"✗ Erro ao inicializar agente de trilhas: {e}")

        if CLIMA_DISPONIVEL:
            self.agentes_inicializados['clima'] = True
            print("✓ Agente de Clima inicializado")

        print()

    def classificar_pergunta(self, pergunta: str) -> str:
        """
        Classifica a pergunta do usuário em uma das categorias:
        - clima: perguntas sobre tempo, temperatura, previsão
        - trilhas: perguntas sobre trilhas, mapas, rotas, caminhos
        - geral: perguntas sobre fauna, flora, história, regras
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um classificador de perguntas sobre o Parque Nacional da Tijuca.

Analise a pergunta do usuário e classifique em UMA das seguintes categorias:

1. **clima** - Use para perguntas sobre:
   - Tempo, temperatura, condições climáticas
   - Previsão do tempo
   - Chuva, sol, vento
   - "Como está o tempo?", "Vai chover?", "Qual a temperatura?"

2. **trilhas** - Use para perguntas sobre:
   - Trilhas específicas (Cascatinha, Pico da Tijuca, etc)
   - Mapas, rotas, caminhos
   - Como chegar em algum lugar
   - Distância, dificuldade de trilhas
   - Pontos turísticos e mirantes
   - "Como faço para ir até...", "Qual trilha leva para...", "Mostre o mapa"

3. **geral** - Use para perguntas sobre:
   - Fauna (animais, aves, macacos)
   - Flora (plantas, árvores, vegetação)
   - História do parque
   - Regras e normas
   - Informações gerais sobre o parque
   - "Quais animais posso ver?", "O que é permitido?"

Responda APENAS com uma palavra: clima, trilhas ou geral"""),
            ("human", "{pergunta}")
        ])

        try:
            chain = prompt | llm_classificador
            resposta = chain.invoke({"pergunta": pergunta})
            categoria = resposta.content.strip().lower()

            # Validar categoria
            if categoria not in ['clima', 'trilhas', 'geral']:
                print(f"⚠️  Categoria inválida '{categoria}', usando 'geral' como padrão")
                categoria = 'geral'

            return categoria

        except Exception as e:
            print(f"⚠️  Erro na classificação: {e}. Usando 'geral' como padrão")
            return 'geral'

    def processar_pergunta(self, pergunta: str):
        """
        Processa a pergunta direcionando para o agente apropriado
        """

        print(f"\n{'=' * 70}")
        print(f"🔍 Pergunta: {pergunta}")
        print(f"{'=' * 70}\n")

        # Classificar a pergunta
        print("🤔 Analisando sua pergunta...\n")
        categoria = self.classificar_pergunta(pergunta)

        # Emojis para cada categoria
        emoji_categoria = {
            'clima': '🌦️',
            'trilhas': '🗺️',
            'geral': '🌿'
        }

        nome_categoria = {
            'clima': 'Clima e Previsão',
            'trilhas': 'Trilhas e Mapas',
            'geral': 'Informações Gerais'
        }

        print(
            f"{emoji_categoria.get(categoria, '📋')} Direcionando para: Agente de {nome_categoria.get(categoria, 'Informações Gerais')}\n")
        print(f"{'=' * 70}\n")

        # Direcionar para o agente apropriado
        try:
            if categoria == 'clima' and 'clima' in self.agentes_inicializados:
                self._processar_clima(pergunta)

            elif categoria == 'trilhas' and 'trilhas' in self.agentes_inicializados:
                self._processar_trilhas(pergunta)

            elif categoria == 'geral' and 'rag' in self.agentes_inicializados:
                self._processar_geral(pergunta)

            else:
                print(f"❌ Agente para '{categoria}' não está disponível no momento.")
                print("💡 Tente outra pergunta ou verifique se todos os agentes estão configurados.\n")

        except Exception as e:
            print(f"❌ Erro ao processar pergunta: {e}")
            import traceback
            traceback.print_exc()

    def _processar_clima(self, pergunta: str):
        """Processa perguntas sobre clima"""

        # Detectar se é clima atual ou previsão
        palavras_previsao = ['previsão', 'previsao', 'próximos', 'proximos',
                             'amanhã', 'amanha', 'semana', 'dias']

        eh_previsao = any(palavra in pergunta.lower() for palavra in palavras_previsao)

        try:
            if eh_previsao:
                print("📅 Buscando previsão do tempo...\n")
                resultado = agente_clima.buscar_previsao(dias=3)
            else:
                print("🌤️  Buscando clima atual...\n")
                resultado = agente_clima.buscar_clima_atual()

            print(resultado)
            print()

        except Exception as e:
            print(f"❌ Erro ao buscar clima: {e}\n")

    def _processar_trilhas(self, pergunta: str):
        """Processa perguntas sobre trilhas e mapas"""

        chain_tuple = self.agentes_inicializados['trilhas']

        try:
            resposta, docs, mapas, self.chat_history = agente_trilhas.processar_pergunta_com_mapas(
                chain_tuple,
                pergunta,
                self.chat_history
            )
        except Exception as e:
            print(f"❌ Erro no agente de trilhas: {e}\n")

    def _processar_geral(self, pergunta: str):
        """Processa perguntas gerais sobre o parque"""

        chain_tuple = self.agentes_inicializados['rag']

        try:
            resposta, docs, self.chat_history = agente_rag.processar_pergunta_langchain(
                chain_tuple,
                pergunta,
                self.chat_history
            )
        except Exception as e:
            print(f"❌ Erro no agente geral: {e}\n")

    def limpar_historico(self):
        """Limpa o histórico de conversa"""
        self.chat_history = []
        print("\n🗑️  Histórico de conversa limpo!\n")


def modo_interativo():
    """
    Interface principal do orquestrador
    """

    print("\n" + "=" * 70)
    print("🌿 ASSISTENTE INTELIGENTE DO PARQUE NACIONAL DA TIJUCA 🏞️")
    print("=" * 70)
    print("\n👋 Olá! Sou seu assistente virtual do Parque Nacional da Tijuca!")
    print("\nPosso te ajudar com:\n")
    print("  🌦️  Clima e previsão do tempo")
    print("  🗺️  Trilhas, mapas e rotas")
    print("  🌿  Fauna, flora e informações gerais")
    print("\nFaça sua pergunta naturalmente e eu vou direcioná-la para o")
    print("especialista mais adequado!")

    # Inicializar orquestrador
    orquestrador = OrquestradorAgentes()

    print("\n" + "=" * 70)
    print("💡 COMANDOS DISPONÍVEIS:")
    print("  • Digite sua pergunta naturalmente")
    print("  • 'sair' - encerrar o programa")
    print("  • 'limpar' - resetar histórico da conversa")
    print("  • 'ajuda' - ver exemplos de perguntas")
    print("=" * 70 + "\n")

    while True:
        try:
            entrada = input("💬 Sua pergunta: ").strip()

            if not entrada:
                continue

            if entrada.lower() in ['sair', 'exit', 'quit']:
                print("\n" + "=" * 70)
                print("👋 Obrigado por usar o Assistente do Parque Nacional da Tijuca!")
                print("   Aproveite sua visita! 🌿🏞️")
                print("=" * 70 + "\n")
                break

            if entrada.lower() in ['limpar', 'clear', 'reset']:
                orquestrador.limpar_historico()
                continue

            if entrada.lower() in ['ajuda', 'help', 'exemplos']:
                print("\n" + "=" * 70)
                print("📖 EXEMPLOS DE PERGUNTAS:")
                print("=" * 70)
                print("\n🌦️  CLIMA:")
                print("  • Como está o tempo agora?")
                print("  • Vai chover hoje?")
                print("  • Qual a previsão para os próximos dias?")
                print("\n🗺️  TRILHAS:")
                print("  • Como faço para chegar no Pico da Tijuca?")
                print("  • Qual a dificuldade da trilha Cascatinha?")
                print("  • Mostre o mapa da trilha do Horto")
                print("\n🌿 INFORMAÇÕES GERAIS:")
                print("  • Quais animais posso ver no parque?")
                print("  • O que é permitido fazer?")
                print("  • Conte sobre a história do parque")
                print("=" * 70 + "\n")
                continue

            # Processar a pergunta
            orquestrador.processar_pergunta(entrada)

        except KeyboardInterrupt:
            print("\n\n👋 Até logo!\n")
            break
        except Exception as e:
            print(f"\n❌ Erro: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    modo_interativo()