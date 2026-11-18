
import os

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Multi-Model AI Gateway", version="0.1.0")

LLAMA_URL = os.getenv("LLAMA_URL", "http://127.0.0.1:8080")

MODEL_MAP = {
    "fast": "qwen3-0.6b-q4_k_m",
    "general": "qwen3-1.7b-q4_k_m",
    "coding": "qwen2.5-coder-1.5b-q4_k_m",
    "reasoning": "deepseek-r1-distill-qwen-1.5b-q4_k_m",
    "advanced": "qwen3-8b-q4_k_m",
}


class ChatRequest(BaseModel):
    task_type: str
    message: str = Field(min_length=1, max_length=10000)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "ai-gateway"}


@app.get("/models")
async def models():
    return {"available_tasks": MODEL_MAP}


@app.post("/chat")
async def chat(request: ChatRequest):
    model = MODEL_MAP.get(request.task_type)

    if model is None:
        raise HTTPException(
            status_code=400,
            detail="Unsupported task_type",
        )

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": request.message}
        ],
        "max_tokens": 256,
        "temperature": 0.2,
        "chat_template_kwargs": {
            "enable_thinking": False
        },
    }

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{LLAMA_URL}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()
    except httpx.HTTPError:
        raise HTTPException(
            status_code=502,
            detail="Inference service unavailable",
        )

    return {
        "task_type": request.task_type,
        "model": model,
        "answer": result["choices"][0]["message"].get("content", ""),
        "usage": result.get("usage", {}),
    }
