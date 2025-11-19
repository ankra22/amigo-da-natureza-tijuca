import os
import io
import sys
import discord
from discord.ext import commands
from dotenv import load_dotenv

from agente_orquestrador import OrquestradorAgentes

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

if not DISCORD_TOKEN:
    raise RuntimeError("❌ DISCORD_TOKEN não está no .env")

# Intents necessários
intents = discord.Intents.default()
intents.message_content = True   # importante para ler o texto das mensagens

bot = commands.Bot(command_prefix="!", intents=intents)

# Instância do orquestrador
orc = OrquestradorAgentes()


def extrair_resposta_discord(saida: str) -> str:
    """
    Extrai só a parte útil da resposta que o orquestrador/agents imprimem.

    - Procura a primeira linha que contenha 'resposta' (case-insensitive)
    - Pega tudo que vem DEPOIS dessa linha
    - Remove separadores '====', menus de mapas e logs desnecessários
    """
    if not saida:
        return "Não consegui gerar uma resposta no momento."

    linhas = saida.splitlines()

    # acha a primeira linha que contenha 'resposta'
    idx_resp = None
    for i, linha in enumerate(linhas):
        if "resposta" in linha.lower():
            idx_resp = i
            break

    if idx_resp is None:
        # fallback: devolve tudo, mas limpando um pouco
        limpa = []
        for linha in linhas:
            l = linha.strip()
            if not l:
                limpa.append(linha)
                continue
            if set(l) == {"="}:
                continue
            if "Agente de" in l or "Direcionando para" in l:
                continue
            if "mapas encontrados" in l.lower():
                continue
            if "deseja visualizar algum mapa" in l.lower():
                continue
            limpa.append(linha)
        resposta = "\n".join(limpa).strip()
        return resposta or "Não consegui gerar uma resposta no momento."

    # pega tudo depois da linha 'Resposta:'
    resto = linhas[idx_resp + 1 :]

    # tira linhas vazias no começo
    while resto and not resto[0].strip():
        resto = resto[1:]

    # corta qualquer coisa que venha depois de menus / logs
    corte_idx = None
    for j, linha in enumerate(resto):
        l = linha.strip().lower()

        # separadores "====="
        if l and set(l) == {"="} and len(l) >= 3:
            corte_idx = j
            break

        # menu de mapas do agente de trilhas
        if "deseja visualizar algum mapa" in l:
            corte_idx = j
            break

        # logs diversos
        if "mapas encontrados" in l:
            corte_idx = j
            break

    if corte_idx is not None:
        resto = resto[:corte_idx]

    resposta = "\n".join(resto).strip()
    return resposta or "Não consegui gerar uma resposta no momento."


@bot.event
async def on_ready():
    print(f"🤖 Bot conectado como {bot.user}")


@bot.command(name="limpar")
async def limpar(ctx):
    """Reseta o histórico de conversa do orquestrador"""
    orc.limpar_historico()
    await ctx.send("🧹 Histórico de conversa limpo com sucesso!")


@bot.event
async def on_message(message: discord.Message):
    """
    Comportamento do bot no Discord:

    - Se a mensagem começar com "!", processa como comando (!limpar)
    - Qualquer OUTRA mensagem de usuário é tratada como pergunta pro orquestrador
    """
    # não responder a si mesmo
    if message.author == bot.user:
        return

    # deixa os comandos normais funcionarem (!limpar, etc.)
    if message.content.startswith("!"):
        await bot.process_commands(message)
        return

    # tudo que não for comando vira pergunta para o sistema
    pergunta = message.content.strip()
    if not pergunta:
        return

    # feedback rápido
    aguardando = await message.channel.send("🔎 Processando sua pergunta...")

    # captura tudo que o orquestrador imprimir
    buffer = io.StringIO()
    stdout_original = sys.stdout
    sys.stdout = buffer

    try:
        orc.processar_pergunta(pergunta)
    finally:
        sys.stdout = stdout_original

    bruto = buffer.getvalue()
    resposta = extrair_resposta_discord(bruto)

    # apaga a mensagem "processando" e envia a resposta final
    try:
        await aguardando.delete()
    except Exception:
        pass

    # Discord tem limite de 2000 caracteres
    if len(resposta) <= 2000:
        await message.channel.send(resposta)
    else:
        partes = [resposta[i:i + 1900] for i in range(0, len(resposta), 1900)]
        for parte in partes:
            await message.channel.send(parte)


bot.run(DISCORD_TOKEN)
