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


def buscar_clima_atual():
    """
    Consulta o clima atual no Parque Nacional da Tijuca usando WeatherAPI.
    """
    url = f"{BASE_URL}/current.json"
    params = {
        "key": WEATHER_API_KEY,
        "q": f"{PARQUE_LAT},{PARQUE_LON}",
        "lang": "pt"  # respostas em português
    }

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    loc = data["location"]
    cur = data["current"]

    cond = cur["condition"]["text"]
    temp = cur["temp_c"]
    sens = cur["feelslike_c"]
    umid = cur["humidity"]
    vento = cur["wind_kph"]
    chuva_mm = cur.get("precip_mm", 0)

    texto = (
        f"🌤️ Clima agora no Parque Nacional da Tijuca:\n\n"
        f"📍 Local: {loc['name']} ({loc['region']}, {loc['country']})\n"
        f"🕒 Horário local: {loc['localtime']}\n\n"
        f"🌡️ Temperatura: {temp:.1f} °C (sensação de {sens:.1f} °C)\n"
        f"🌧️ Condição: {cond}\n"
        f"💧 Umidade: {umid}%\n"
        f"💨 Vento: {vento:.1f} km/h\n"
        f"🌧️ Chuva nas últimas horas: {chuva_mm} mm\n"
    )

    # Regrinha marota pra trilha
    dicas = []
    if chuva_mm > 0 or "chuva" in cond.lower():
        dicas.append("⚠️ Choveu recentemente ou está chovendo: cuidado com trilhas escorregadias.")
    if temp > 30:
        dicas.append("🥵 Muito calor: leve bastante água, chapéu e protetor solar.")
    if temp < 18:
        dicas.append("🧥 Tempo mais frio: leve um agasalho leve para as partes mais altas do parque.")
    if vento > 20:
        dicas.append("💨 Vento moderado a forte: atenção em mirantes e áreas expostas.")

    if dicas:
        texto += "\nDicas para trilha hoje:\n" + "\n".join(dicas)

    return texto


def buscar_previsao(dias: int = 3):
    """
    Consulta a previsão do tempo para os próximos 'dias' (máx. 3 no plano free).
    """
    dias = max(1, min(dias, 3))  # limita entre 1 e 3
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

    loc = data["location"]
    forecast_days = data["forecast"]["forecastday"]

    texto = (
        f"📅 Previsão do tempo para o Parque Nacional da Tijuca\n"
        f"📍 {loc['name']} ({loc['region']}, {loc['country']})\n\n"
    )

    for dia in forecast_days:
        data_str = dia["date"]
        dia_info = dia["day"]
        cond = dia_info["condition"]["text"]
        tmax = dia_info["maxtemp_c"]
        tmin = dia_info["mintemp_c"]
        chuva_mm = dia_info["totalprecip_mm"]
        chance_chuva = dia_info.get("daily_chance_of_rain", 0)

        texto += (
            f"📆 {data_str}:\n"
            f"   🌡️ Máx: {tmax:.1f} °C / Mín: {tmin:.1f} °C\n"
            f"   🌧️ Condição: {cond}\n"
            f"   💧 Chuva prevista: {chuva_mm} mm (chance: {chance_chuva}%)\n\n"
        )

    texto += "💡 Use essas informações para planejar melhor sua visita e escolher as trilhas com segurança.\n"

    return texto


def modo_interativo():
    print("=" * 70)
    print("🌦️ Agente de Clima – Parque Nacional da Tijuca")
    print("=" * 70)
    print("Comandos:")
    print("  • 'agora'     → clima atual no parque")
    print("  • 'previsao'  → previsão para os próximos dias")
    print("  • 'sair'      → encerrar")
    print("=" * 70)

    while True:
        cmd = input("\nDigite um comando (agora/previsao/sair): ").strip().lower()

        if cmd == "sair":
            print("👋 Saindo do agente de clima.")
            break

        try:
            if cmd == "agora":
                resp = buscar_clima_atual()
                print("\n" + resp)
            elif cmd == "previsao":
                dias_str = input("Quantos dias de previsão? (1 a 3): ").strip()
                try:
                    dias = int(dias_str)
                except ValueError:
                    dias = 3
                resp = buscar_previsao(dias=dias)
                print("\n" + resp)
            else:
                print("❓ Comando não reconhecido. Use 'agora', 'previsao' ou 'sair'.")
        except Exception as e:
            print(f"❌ Erro ao consultar clima: {e}")


if __name__ == "__main__":
    modo_interativo()


