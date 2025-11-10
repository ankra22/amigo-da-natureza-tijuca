import os, httpx

# Endpoint OpenAI-compatível da Groq
GROQ_BASE = "https://api.groq.com/openai/v1/chat/completions"

def chat(messages, model="llama-3.3-70b-versatile", temperature=0.6, stream=False):
    """
    Envia mensagens no formato OpenAI para o endpoint da Groq e retorna a resposta.
    Compatível com modelos como llama-3.3-70b-versatile.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {
            "role": "assistant",
            "content": "[Groq não configurado: defina GROQ_API_KEY no arquivo .env]"
        }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "stream": bool(stream)
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        with httpx.Client(timeout=60) as client:
            r = client.post(GROQ_BASE, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]
    except Exception as e:
        return {"role": "assistant", "content": f"[Erro na requisição Groq: {e}]"}
