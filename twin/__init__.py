"""Digital Twin core: configuration, prompt construction, LLM client, tools."""

from .config import GROQ_MODEL, MODEL, api_key_configured, get_api_key
from .contact import extract_email, load_contacts, save_contact_info
from .llm import TwinLLMError, generate_reply, generate_reply_stream
from .prompt import build_system_prompt

__all__ = [
    "MODEL",
    "GROQ_MODEL",
    "api_key_configured",
    "get_api_key",
    "TwinLLMError",
    "generate_reply",
    "generate_reply_stream",
    "build_system_prompt",
    "extract_email",
    "save_contact_info",
    "load_contacts",
]
