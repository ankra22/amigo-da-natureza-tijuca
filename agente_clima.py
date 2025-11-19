import os
import requests
from dotenv import load_dotenv

load_dotenv()

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

if not WEATHER_API_KEY:
    raise RuntimeError(
        "⚠️ WEATHER_API_KEY não encontrada no .env.\n"
        "Adicione no arquivo .env a linha:\n"
        "WEATHER_API_KEY=SUA_CHAVE_AQUI"
    )

# Coordenadas aproximadas do Parque Nacional da Tijuca
PARQUE_LAT = -22.9517
PARQUE_LON = -43.2644

BASE_URL = "http://api.weatherapi.com/v1"


# ------------------------------------------------------------------
# Funções auxiliares
# ------------------------------------------------------------------

def _buscar_forecast(dias: int = 3):
    """
    Consulta a API de previsão e retorna a lista de dias de forecast.
    """
    dias = max(1, min(dias, 3))
    url = f"{BASE_URL}/forecast.json"
    params = {
        "key": WEATHER_API_KEY,
        "q": f"{PARQUE_LAT},{PARQUE_LON}",
        "days": dias,
        "lang": "pt"
    }

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    return data["forecast"]["forecastday"]


def _formatar_dia(dia: dict, nome: str) -> str:
    """
    Formata o texto de previsão para um único dia.
    """
    info = dia["day"]
    cond = info["condition"]["text"]
    tmax = info["maxtemp_c"]
    tmin = info["mintemp_c"]
    chuva = float(info.get("daily_chance_of_rain", 0) or 0)

    texto = (
        f"{nome}, a previsão indica {cond.lower()}, "
        f"com temperaturas variando de cerca de {tmin:.1f} °C nas horas mais frescas "
        f"até aproximadamente {tmax:.1f} °C nos períodos mais quentes. "
    )

    if chuva >= 70:
        texto += (
            f"A chance de chuva é alta, em torno de {chuva:.0f}%, "
            "então é bem provável pegar trechos molhados ou pancadas ao longo do dia. "
        )
    elif chuva >= 40:
        texto += (
            f"A chance de chuva é moderada, perto de {chuva:.0f}%, "
            "com possibilidade de instabilidades em alguns momentos. "
        )
    else:
        texto += (
            f"A chance de chuva é baixa, por volta de {chuva:.0f}%, "
            "e o dia tende a ser mais estável. "
        )

    texto += (
        "Para trilhas, vale sempre levar água, protetor solar e um casaco leve, "
        "e redobrar a atenção em trechos íngremes caso o solo esteja úmido."
    )

    return texto


# ------------------------------------------------------------------
# Consultas à API
# ------------------------------------------------------------------

def buscar_clima_atual() -> str:
    """
    Consulta o clima atual no Parque Nacional da Tijuca e devolve
    um texto natural (sem listas / menus).
    """
    url = f"{BASE_URL}/current.json"
    params = {
        "key": WEATHER_API_KEY,
        "q": f"{PARQUE_LAT},{PARQUE_LON}",
        "lang": "pt"
    }

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    cur = data["current"]
    cond = cur["condition"]["text"]

    temp = cur["temp_c"]
    sens = cur["feelslike_c"]
    umid = cur["humidity"]
    vento = cur["wind_kph"]
    chuva_mm = cur.get("precip_mm", 0.0)

    texto = (
        f"Neste momento, o Parque Nacional da Tijuca está com {cond.lower()}. "
        f"A temperatura gira em torno de {temp:.1f} °C, com sensação térmica próxima de {sens:.1f} °C. "
        f"A umidade do ar está em cerca de {umid}% e o vento sopra a aproximadamente {vento:.0f} km/h. "
    )

    if chuva_mm and chuva_mm > 0:
        texto += (
            f"Houve chuva recente, com cerca de {chuva_mm:.1f} mm acumulados, "
            "então alguns trechos de trilha podem estar mais úmidos ou escorregadios. "
        )
    else:
        texto += "Não há registro de chuva significativa nas últimas horas, então as trilhas tendem a estar mais secas. "

    texto += (
        "Mesmo assim, é sempre importante usar calçados adequados, levar água e ficar atento a mudanças rápidas no tempo."
    )

    return texto


def buscar_previsao(dias: int = 3) -> str:
    """
    Consulta a previsão para os próximos 'dias' (máx. 3 no plano gratuito)
    e devolve um texto natural, separado por dias.
    """
    forecast_days = _buscar_forecast(dias)
    partes = []
    nomes = ["Hoje", "Amanhã", "Depois de amanhã"]

    for i, dia in enumerate(forecast_days):
        nome = nomes[i] if i < len(nomes) else f"No dia {dia['date']}"
        partes.append(_formatar_dia(dia, nome))

    return "\n\n".join(partes)


# ------------------------------------------------------------------
# Função principal chamada pelo orquestrador
# ------------------------------------------------------------------

def responder_clima(pergunta: str) -> str:
    """
    Decide automaticamente se o usuário quer clima atual ou previsão,
    e se for previsão, tenta respeitar o dia pedido (hoje, amanhã, depois de amanhã).
    """
    p = pergunta.lower()

    fala_hoje = "hoje" in p
    fala_amanha = "amanhã" in p or "amanha" in p
    fala_depois = "depois de amanhã" in p or "depois de amanha" in p
    fala_proximos = "próximos dias" in p or "proximos dias" in p or \
        "próximos 3 dias" in p or "proximos 3 dias" in p

    quer_previsao = any(palavra in p for palavra in [
        "previsão", "previsao", "vai chover", "chover",
        "tempo para", "previsão do tempo", "previsao do tempo"
    ]) or fala_hoje or fala_amanha or fala_depois or fala_proximos

    quer_agora = any(palavra in p for palavra in [
        "agora", "no momento", "nesse momento",
        "clima atual", "como está o tempo", "como ta o tempo", "como tá o tempo"
    ])

    try:
        # Só clima atual
        if quer_agora and not quer_previsao:
            return buscar_clima_atual()

        # Dia específico: depois de amanhã
        if fala_depois:
            forecast_days = _buscar_forecast(3)
            if len(forecast_days) >= 3:
                return _formatar_dia(forecast_days[2], "Depois de amanhã")
            else:
                return buscar_previsao(dias=len(forecast_days))

        # Dia específico: amanhã (sem pedir "próximos dias")
        if fala_amanha and not fala_proximos:
            forecast_days = _buscar_forecast(2)
            if len(forecast_days) >= 2:
                return _formatar_dia(forecast_days[1], "Amanhã")
            else:
                return buscar_previsao(dias=len(forecast_days))

        # Dia específico: hoje (quando a pessoa fala em previsão de hoje)
        if fala_hoje and quer_previsao and not (fala_amanha or fala_depois or fala_proximos):
            forecast_days = _buscar_forecast(1)
            return _formatar_dia(forecast_days[0], "Hoje")

        # Previsão geral: próximos dias
        if quer_previsao and not quer_agora:
            return buscar_previsao(dias=3)

        # Pergunta genérica tipo "me fale sobre o clima"
        atual = buscar_clima_atual()
        prev = buscar_previsao(dias=3)

        return (
            atual
            + "\n\n"
            + "Para os próximos dias, a previsão é a seguinte:\n\n"
            + prev
        )

    except Exception as e:
        return f"Não foi possível consultar o clima no momento: {e}"
