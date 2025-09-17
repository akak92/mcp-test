import os
import httpx
from starlette.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from mcp.server.fastmcp import FastMCP

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://llm:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
TIMEOUT = float(os.getenv("HTTP_TIMEOUT_SEC", "60"))

mcp = FastMCP("Llama3 MCP")

@mcp.tool()
async def chat_llama(
    prompt: str,
    system: str | None = None,
    temperature: float = 0.2,
) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }

    async with httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=TIMEOUT) as client:
        r = await client.post("/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "")
    
# --------------- BACKEND SWITCH ---------------
BACKEND = os.getenv("MCP_BACKEND", "ollama").lower()

def set_backend_value(val: str):
    global BACKEND
    BACKEND = val.lower()

@mcp.tool()
async def current_backend() -> str:
    """Indica qué backend está activo: 'ollama' o 'azure'."""
    return BACKEND

@mcp.tool()
async def set_backend(backend: str) -> str:
    """
    Cambia backend por defecto en caliente (proceso en memoria): 'ollama' o 'azure'.
    """
    val = backend.strip().lower()
    if val not in ("ollama", "azure"):
        return "Backend inválido. Usá 'ollama' o 'azure'."
    set_backend_value(val)
    return f"backend set to {val}"


# --------------- AZURE OPENAI CONFIG ---------------
AZ_OAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AZ_OAI_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZ_OAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
AZ_OAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "")
AZ_OAI_TIMEOUT = float(os.getenv("AZURE_OPENAI_TIMEOUT_SEC", "60"))

def _azure_chat_url() -> str:
    if not AZ_OAI_ENDPOINT:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT no configurado")
    return f"{AZ_OAI_ENDPOINT}/openai/deployments/{AZ_OAI_DEPLOYMENT}/chat/completions?api-version={AZ_OAI_API_VERSION}"

_headers_azure = {
    "Content-Type": "application/json",
    "api-key": AZ_OAI_KEY or "",   # header esperado por Azure OpenAI
}

@mcp.tool()
async def chat_azure(
    prompt: str,
    system: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    """
    Envía un chat a Azure OpenAI (deployment gpt-4.1, etc.) usando Chat Completions.
    """
    if not AZ_OAI_KEY or not AZ_OAI_ENDPOINT:
        return "Azure OpenAI no está configurado (faltan AZURE_OPENAI_* en .env)."

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        # Podés agregar top_p, frequency_penalty, presence_penalty, etc.
    }

    url = _azure_chat_url()
    async with httpx.AsyncClient(timeout=AZ_OAI_TIMEOUT) as client:
        r = await client.post(url, headers=_headers_azure, json=payload)
        r.raise_for_status()
        data = r.json()
        # Estructura típica: choices[0].message.content
        choices = data.get("choices") or []
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "") or ""


# --------------- TOOL UNIFICADA (opcional) ---------------
@mcp.tool()
async def chat(
    prompt: str,
    system: str | None = None,
    temperature: float = 0.2,
) -> str:
    """
    Chat unificado: envía a Ollama o Azure según MCP_BACKEND/set_backend.
    """
    if BACKEND == "azure":
        return await chat_azure(prompt=prompt, system=system, temperature=temperature)
    # default → ollama
    return await chat_llama(prompt=prompt, system=system, temperature=temperature)


@mcp.tool()
async def list_models() -> list[dict]:
    """
    Lista los modelos disponibles en Ollama (usa /api/tags).
    Devuelve una lista de dicts con: name, size (bytes), digest, modified.
    """
    async with httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=TIMEOUT) as client:
        r = await client.get("/api/tags")
        r.raise_for_status()
        data = r.json() or {}
        models = data.get("models", [])
        out = []
        for m in models:
            out.append({
                "name": m.get("name"),
                "size": m.get("size"),                 # bytes (si viene)
                "digest": m.get("digest"),
                "modified": m.get("modified_at") or m.get("modified"),
            })
        return out

@mcp.tool()
async def current_model() -> str:
    """
    Muestra el modelo por defecto configurado vía OLLAMA_MODEL.
    """
    return os.getenv("OLLAMA_MODEL", "llama3")

# *** Usar la app de FastMCP como app principal (lifespan ON adentro) ***
app = mcp.streamable_http_app()  # expone /mcp

# Healthcheck simple
@app.route("/health")
async def health(request):
    return JSONResponse({"status": "ok"})

# CORS amplio (para Inspector/clients web)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],    # incluye OPTIONS (preflight)
    allow_headers=["*"],    # o lista específica si querés
    expose_headers=["Mcp-Session-Id"],
)
