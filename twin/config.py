"""Environment-driven configuration. The API key lives only here (from .env)."""

import os
from pathlib import Path

from dotenv import load_dotenv

# .env sits at the project root, one level above the twin/ package.
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

API_KEY: str = os.getenv("OPENROUTER_API_KEY", "").strip()
MODEL: str = os.getenv("TWIN_MODEL", "nvidia/nemotron-3-super:free").strip()

TWIN_NAME = "Padmanav Mohanty"
APP_NAME = f"{TWIN_NAME} — Digital Twin"

# Runtime endpoint defaults (overridable via env for flexibility).
OPENROUTER_URL: str = os.getenv(
    "OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions"
)
REFERER: str = os.getenv("TWIN_SITE_URL", "").strip()

# Fallback provider (Groq). Inactive unless GROQ_API_KEY is set.
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()
GROQ_URL: str = os.getenv(
    "GROQ_URL", "https://api.groq.com/openai/v1/chat/completions"
)

# Hosted-demo mode (Hugging Face Space): contact saving is disabled and
# visitors are directed to the public contact email instead. Auto-ON when
# running on a Space (SPACE_ID is set by the platform); override locally with
# TWIN_DEPLOY_MODE=1/0 to simulate the hosted behavior.
_env_deploy = os.getenv("TWIN_DEPLOY_MODE", "").strip().lower()
DEPLOY_MODE: bool = (
    _env_deploy in ("1", "true", "yes", "on")
    if _env_deploy
    else bool(os.getenv("SPACE_ID", "").strip())
)

# Public contact email shown in hosted mode (Padmanav's public email, also in
# profile.json). Override with PUBLIC_CONTACT_EMAIL if it ever changes.
PUBLIC_CONTACT_EMAIL: str = os.getenv(
    "PUBLIC_CONTACT_EMAIL", "padmanav.mohanty26@gmail.com"
).strip()

# Conversation is bounded so requests never grow indefinitely.
MAX_HISTORY_MESSAGES = 16
MAX_MESSAGE_CHARS = 2000


def get_api_key() -> str:
    """Return the configured OpenRouter API key (empty string if unset)."""
    return API_KEY


def api_key_configured() -> bool:
    return bool(API_KEY)
