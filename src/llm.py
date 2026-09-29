"""
LLM module supporting Groq (Llama 3.1 / Open-Source LLMs) and optional local Ollama.
Manages provider selection, dynamic model discovery, API key validation, and temperature controls.
"""

import os
import json
from typing import Optional, List
import urllib.request
import urllib.error
from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.llms.ollama import Ollama
from src.config import (
    DEFAULT_GROQ_MODEL,
    AVAILABLE_GROQ_MODELS,
    FALLBACK_GROQ_MODELS,
    DEFAULT_OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    GROQ_API_KEY
)
from src.prompts import SYSTEM_PROMPT
from utils.logging_utils import get_logger

logger = get_logger("llm")

def check_ollama_available(base_url: str = OLLAMA_BASE_URL) -> bool:
    """
    Checks if local Ollama daemon is currently running and reachable.
    """
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False

def fetch_available_groq_models(api_key: Optional[str] = None) -> List[str]:
    """
    Queries Groq's /openai/v1/models endpoint to find chat models accessible on the current API key.
    Filters out non-chat models (whisper, guard, speech, tts, orpheus, playai).
    Falls back to FALLBACK_GROQ_MODELS on any failure.
    """
    key = api_key or os.getenv("GROQ_API_KEY", GROQ_API_KEY)
    if not key or not key.strip():
        return FALLBACK_GROQ_MODELS.copy()

    try:
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/models",
            headers={
                "Authorization": f"Bearer {key.strip()}",
                "User-Agent": "RAG-Knowledge-Assistant/1.0"
            }
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            model_ids = [m["id"] for m in data.get("data", [])]

            # Exclude non-chat models
            excluded_keywords = ["whisper", "guard", "orpheus", "safeguard", "tts", "playai", "speech"]
            chat_models = [
                m for m in model_ids
                if not any(ex in m.lower() for ex in excluded_keywords)
            ]
            if chat_models:
                def model_sort_key(m_id):
                    m_lower = m_id.lower()
                    if "llama-3.3" in m_lower:
                        return 0
                    elif "llama-3.1" in m_lower:
                        return 1
                    elif "llama" in m_lower:
                        return 2
                    elif "qwen" in m_lower:
                        return 3
                    elif "gpt-oss" in m_lower:
                        return 4
                    return 5

                return sorted(chat_models, key=model_sort_key)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            logger.warning(
                f"Groq API key may be invalid or expired (HTTP {e.code}). "
                "Using fallback model list."
            )
        else:
            logger.warning(f"Could not query Groq models (HTTP {e.code}): {e}")
    except Exception as e:
        logger.warning(f"Could not dynamically query Groq models: {e}")

    return FALLBACK_GROQ_MODELS.copy()

def get_llm(
    provider: str = "Groq",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.1
):
    """
    Factory function returning the configured LLM instance.
    Includes smart auto-fallback if a model is unavailable on the account.

    Args:
        provider: "Groq" or "Ollama"
        api_key: Groq API key (defaults to GROQ_API_KEY environment variable)
        model_name: Optional custom model name
        temperature: Sampling temperature (low temperature = 0.1 for grounded answers)

    Returns:
        LLM instance compatible with LlamaIndex.
    """
    provider_clean = provider.strip().capitalize()

    if provider_clean == "Groq":
        active_key = api_key.strip() if api_key else os.getenv("GROQ_API_KEY", GROQ_API_KEY).strip()

        if not active_key:
            raise ValueError(
                "GROQ_API_KEY is not configured. Please add your key to the .env file "
                "or enter it in the sidebar settings."
            )

        target_model = model_name or DEFAULT_GROQ_MODEL

        # Dynamic validation against available models
        try:
            available = fetch_available_groq_models(active_key)
            if target_model not in available and available:
                fallback = available[0]
                logger.warning(
                    f"Model '{target_model}' not found on this Groq account. "
                    f"Auto-falling back to available model: '{fallback}'"
                )
                target_model = fallback
        except Exception:
            pass

        logger.info(f"Initializing Groq LLM with model: {target_model}, temperature: {temperature}")

        llm = Groq(
            model=target_model,
            api_key=active_key,
            temperature=temperature,
            system_prompt=SYSTEM_PROMPT
        )
        Settings.llm = llm
        return llm

    elif provider_clean == "Ollama":
        active_model = model_name or DEFAULT_OLLAMA_MODEL
        base_url = OLLAMA_BASE_URL

        if not check_ollama_available(base_url):
            raise ConnectionError(
                f"Ollama server is not reachable at {base_url}. "
                "Please verify Ollama is installed and running (`ollama serve`)."
            )

        logger.info(f"Initializing Ollama LLM with model: {active_model} at {base_url}")
        llm = Ollama(
            model=active_model,
            base_url=base_url,
            temperature=temperature,
            request_timeout=60.0
        )
        Settings.llm = llm
        return llm

    else:
        raise ValueError(f"Unsupported LLM provider: '{provider}'. Choose 'Groq' or 'Ollama'.")
