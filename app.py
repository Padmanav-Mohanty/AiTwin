"""Padmanav Mohanty — Digital Twin (Gradio application).

Launch with:  python app.py
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

import gradio as gr

try:
    import spaces  # type: ignore[import-not-found]  # preinstalled on HF ZeroGPU
except ImportError:  # local dev: the package is optional and a no-op here
    spaces = None

from twin.config import (
    APP_NAME,
    DEPLOY_MODE,
    MAX_HISTORY_MESSAGES,
    MAX_MESSAGE_CHARS,
    PUBLIC_CONTACT_EMAIL,
    TWIN_NAME,
    api_key_configured,
)
from twin.contact import extract_email, save_contact_info
from twin.llm import TwinLLMError, generate_reply_stream
from twin.prompt import build_system_prompt

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
AVATAR_PATH = str(ASSETS_DIR / "avatar.svg")
AVATAR_SVG = (ASSETS_DIR / "avatar.svg").read_text(encoding="utf-8")

HEADER = f"""
<div class="twin-header">
  <div class="twin-header-row">
    <div class="twin-avatar">{AVATAR_SVG}</div>
    <div>
      <div class="twin-kicker">DIGITAL TWIN</div>
      <h1 class="twin-name">Padmanav Mohanty</h1>
      <div class="twin-sub">AI / ML &amp; AI Engineering · LLMs · RAG · Fine-Tuning · Agentic AI</div>
    </div>
  </div>
  <div class="twin-rule"></div>
  <div class="twin-intro">
    <h2>Meet my Digital Twin</h2>
    <p>
      I built this AI representation of myself so you can ask it anything about my
      background, projects, skills, and what I'm currently learning. It answers only
      from my profile — if it doesn't know, it will say so.
    </p>
  </div>
</div>
"""

FOOTER = """
<div class="twin-footer">
  AI-generated answers · Padmanav's Digital Twin may not have every detail ·
  <a href="https://www.linkedin.com/in/padmanav-mohanty-a4989a30a" target="_blank">LinkedIn</a>
</div>
"""

CUSTOM_CSS = """
/* ---------- Digital Twin portfolio theme ---------- */
.gradio-container {
  background:
    radial-gradient(1100px 500px at 85% -10%, rgba(139, 91, 255, 0.16), transparent 60%),
    radial-gradient(900px 420px at 5% 110%, rgba(109, 141, 255, 0.13), transparent 60%),
    #0b0f19 !important;
  color: #e8ecf5 !important;
  font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif !important;
}
#twin-column { max-width: 880px !important; margin: 0 auto !important; width: 100% !important; }

/* Header */
.twin-header { padding: 26px 8px 6px; }
.twin-header-row { display: flex; align-items: center; gap: 16px; }
.twin-avatar { flex-shrink: 0; }
.twin-kicker { font-size: 11px; letter-spacing: 2.5px; font-weight: 700; color: #8fa3ff; margin-bottom: 4px; }
.twin-name { font-size: 30px; font-weight: 700; margin: 0; color: #f4f6fb; line-height: 1.15; }
.twin-sub { font-size: 13.5px; color: #9aa8c4; margin-top: 5px; }
.twin-rule {
  height: 1px; margin: 18px 0 14px;
  background: linear-gradient(90deg, rgba(139,91,255,0.55), rgba(109,141,255,0.25), transparent);
}
.twin-intro h2 { font-size: 19px; font-weight: 650; color: #eef1f8; margin: 0 0 6px; }
.twin-intro p { font-size: 14.5px; color: #a9b4cc; line-height: 1.6; margin: 0; max-width: 660px; }

/* Chat area */
#twin-chat {
  background: rgba(21, 27, 43, 0.75) !important;
  border: 1px solid #26304d !important;
  border-radius: 16px !important;
  box-shadow: 0 14px 40px rgba(0, 0, 0, 0.35);
}

/* Chips */
.twin-chips { display: flex; flex-wrap: wrap; gap: 8px; }
.twin-chips button {
  background: rgba(109, 141, 255, 0.08) !important;
  border: 1px solid #2c3a63 !important;
  color: #aebbdd !important;
  border-radius: 999px !important;
  padding: 7px 14px !important;
  font-size: 13px !important;
  min-width: 0 !important;
  box-shadow: none !important;
}
.twin-chips button:hover { border-color: #6d8dff !important; color: #e8ecf5 !important; }

/* Input row */
#twin-input textarea, #twin-input input {
  background: #121828 !important;
  border: 1px solid #2a3550 !important;
  border-radius: 12px !important;
  color: #e8ecf5 !important;
  font-size: 14.5px !important;
}
#twin-input textarea:focus, #twin-input input:focus { border-color: #6d8dff !important; }
#twin-send {
  background: linear-gradient(135deg, #6d8dff, #8b5bff) !important;
  border: none !important;
  border-radius: 12px !important;
  color: white !important;
  font-weight: 600 !important;
  box-shadow: 0 6px 18px rgba(109, 91, 255, 0.35) !important;
}
#twin-send:hover { filter: brightness(1.08); }
#twin-clear {
  background: transparent !important;
  border: 1px solid #2c3a63 !important;
  color: #8d9cbd !important;
  border-radius: 10px !important;
  font-size: 13px !important;
  box-shadow: none !important;
}
#twin-clear:hover { border-color: #41507d !important; color: #c3cde4 !important; }

/* Footer */
.twin-footer { text-align: center; font-size: 12.5px; color: #6d7894; padding: 14px 0 6px; }
.twin-footer a { color: #8fa3ff; text-decoration: none; }
.twin-footer a:hover { text-decoration: underline; }
footer { visibility: hidden; }

/* Mobile */
@media (max-width: 640px) {
  .twin-name { font-size: 23px; }
  .twin-intro p { font-size: 13.5px; }
}
"""

EXAMPLE_QUESTIONS = [
    "Who are you?",
    "Tell me about Padmanav.",
    "What projects has he built?",
    "What is ForgeMind?",
    "What is he currently learning?",
    "🤝 I'd like to collaborate with you! Please contact me.",
]


if spaces is not None:
    @spaces.GPU(duration=10)
    def _zerogpu_marker() -> str:
        # Hugging Face ZeroGPU (mandatory on free accounts since July 2026)
        # refuses to start a Space unless at least one @spaces.GPU function
        # exists. This app is a pure network client (OpenRouter/Groq APIs) and
        # never uses the GPU, so this marker is deliberately NOT wired to any
        # UI event: it satisfies the startup detection without spending any
        # visitor GPU quota. All chat handlers below run on CPU as normal.
        return "ok"
else:
    def _zerogpu_marker() -> str:
        return "ok"


COLLAB_STARTER = EXAMPLE_QUESTIONS[-1]

# Streaming "typewriter" pacing.
_TICK_DELAY = 0.015

# A contact request = visitor wants Padmanav to reach them. Deterministic
# keyword gate so casual mentions of "internship" etc. don't trigger it.
_CONTACT_INTENT_RE = re.compile(
    r"\b(contact|reach(ed)?\s*out|get\s+in\s+touch|touch\s+base|connect(\s+with)?|"
    r"collaborat\w*|hire|hiring|recruit\w*|work\s+with|talk\s+to|speak\s+with|"
    r"my\s+email|email\s+me|contact\s+me|reach\s+me)",
    re.IGNORECASE,
)

# Detects email addresses in visitor messages (same pattern as twin/contact).
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# Hosted mode: appended when a visitor shares contact details. Nothing is
# stored or logged; visitors are directed to the public contact channels.
_DEPLOY_CONTACT_NOTE = (
    "\n\n---\n📬 Contact saving is disabled in this hosted demo, but you can "
    f"reach Padmanav directly at [{PUBLIC_CONTACT_EMAIL}](mailto:{PUBLIC_CONTACT_EMAIL}) "
    "or via [LinkedIn](https://www.linkedin.com/in/padmanav-mohanty-a4989a30a)."
)


def _welcome_history() -> list[dict[str, str]]:
    """Empty-state opening message shown when the conversation starts."""
    return [
        {
            "role": "assistant",
            "content": (
                "Hi! 👋 I'm **Padmanav's Digital Twin** — an AI chatbot he built to "
                "represent him. Ask me about his projects (ForgeMind, Document RAG), "
                "his skills, education, or what he's exploring in AI/ML.\n\n"
                "If I don't know something, I'll tell you honestly."
            ),
        }
    ]


# ------------------------------------------------------------------- events

def _content_to_text(content) -> str:
    """Normalize Gradio 6 message content (string or list of parts) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        return str(content.get("text", ""))
    return str(content)


def user_submit(message: str, history: list[dict[str, str]]):
    """Add the user's message to the chat and clear the textbox."""
    return "", (history or []) + [{"role": "user", "content": message or ""}]


def _maybe_save_contact(history: list[dict[str, str]]) -> dict | None:
    """Save contact details when the latest user message contains an email
    and the conversation shows contact intent. Returns the save result or None."""
    user_msgs = [
        _content_to_text(m.get("content", ""))
        for m in (history or [])
        if m.get("role") == "user"
    ]
    if not user_msgs:
        return None
    latest = user_msgs[-1]
    email = extract_email(latest)
    if not email:
        return None
    recent_text = " ".join(user_msgs[-4:])
    if not _CONTACT_INTENT_RE.search(recent_text):
        return None

    name = ""
    name_match = re.search(r"my name is\s+(.+)", latest, re.IGNORECASE)
    if name_match:
        raw = name_match.group(1)
        # Stop the name at natural delimiters ("and", "my email", punctuation).
        raw = re.split(
            r"\s+and\b|,|;|\u2026|\s+my\s+email\b|\s+email\b|\n", raw, flags=re.IGNORECASE
        )[0]
        raw = raw.strip(" .,;:!?-")
        if 1 <= len(raw) <= 40:
            name = raw
    return save_contact_info(email=email, name=name, message=latest[:300])


def _contact_note(result: dict | None) -> str:
    if result is None:
        return ""
    if result.get("saved"):
        return (
            "\n\n---\n💾 **Your contact details were saved** — Padmanav will get "
            "back to you soon! (stored locally in contacts.json / contacts.csv)"
        )
    return "\n\n---\n⚠️ Sorry — there was a technical issue saving your contact " \
           "details. Please try again in a moment."


async def bot_stream(history: list[dict[str, str]]):
    """Validate input, stream the twin's reply token-by-token, and handle
    contact requests. Yields the updated chat history for the typewriter
    effect (single output: the chatbot component)."""
    if not api_key_configured():
        raise gr.Error(
            "The Digital Twin is not configured yet: the owner needs to set "
            "OPENROUTER_API_KEY in the Space secrets."
            if DEPLOY_MODE
            else "The Digital Twin is not configured yet: add OPENROUTER_API_KEY "
            "to .env and restart the app.",
            duration=8,
        )

    history = history or []
    user_msgs = [m for m in history if m.get("role") == "user"]
    message = _content_to_text(user_msgs[-1]["content"] if user_msgs else "").strip()

    if not message:
        raise gr.Error("Please type a message first.", duration=3)
    if len(message) > MAX_MESSAGE_CHARS:
        raise gr.Error(
            f"That message is too long ({len(message)} characters). "
            f"Please keep it under {MAX_MESSAGE_CHARS}.",
            duration=5,
        )

    # Contact handling: in hosted mode nothing is ever written to disk and
    # visitors are pointed at the public email; locally the lead is saved.
    contact_result = None if DEPLOY_MODE else _maybe_save_contact(history)

    # Bound the conversation so requests never grow indefinitely, and make
    # sure every message has plain-string content for the LLM.
    recent = history[-MAX_HISTORY_MESSAGES:]
    while recent and recent[0].get("role") != "user":
        recent = recent[1:]
    recent = [
        {"role": m.get("role", "user"), "content": _content_to_text(m.get("content", ""))}
        for m in recent
    ]

    messages = (
        [{"role": "system", "content": build_system_prompt()}]
        + recent
    )

    reply_chunks: list[str] = []
    try:
        async for delta in generate_reply_stream(messages):
            reply_chunks.append(delta)
            current = "".join(reply_chunks)
            yield history + [{"role": "assistant", "content": current}]
            await asyncio.sleep(_TICK_DELAY)
    except TwinLLMError as exc:
        # Internal errors stay out of the chat history; shown as a toast.
        raise gr.Error(str(exc), duration=8) from exc

    final_reply = "".join(reply_chunks).strip()
    if not final_reply:
        raise gr.Error("The model returned an empty response. Please try again.", duration=8)

    tail = _contact_note(contact_result)
    if not tail and DEPLOY_MODE and _EMAIL_RE.search(message):
        tail = _DEPLOY_CONTACT_NOTE
    yield history + [
        {"role": "assistant", "content": final_reply + tail}
    ]


def clear_chat():
    return _welcome_history(), ""


# --------------------------------------------------------------------- UI

def make_theme() -> gr.themes.Soft:
    """Professional dark-neutral Gradio theme for the Digital Twin."""
    return gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="violet",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "Segoe UI", "system-ui", "sans-serif"],
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title=APP_NAME, analytics_enabled=False) as demo:
        gr.HTML(HEADER)

        with gr.Column(elem_id="twin-column"):
            chatbot = gr.Chatbot(
                value=_welcome_history(),
                elem_id="twin-chat",
                height=420,
                label="Ask my Digital Twin",
                buttons=["copy", "copy_all"],
                avatar_images=(None, AVATAR_PATH),
                render_markdown=True,
            )

            with gr.Row():
                msg = gr.Textbox(
                    placeholder=f"Ask something about {TWIN_NAME.split()[0]}…",
                    show_label=False,
                    container=False,
                    autofocus=True,
                    elem_id="twin-input",
                    max_lines=4,
                )
                send_btn = gr.Button("Send", variant="primary", elem_id="twin-send", scale=0)

            with gr.Group(elem_classes=["twin-chips"]):
                with gr.Row():
                    chip_buttons = [gr.Button(q, size="sm") for q in EXAMPLE_QUESTIONS]

            with gr.Row():
                clear_btn = gr.Button("🗑 Clear conversation", size="sm", elem_id="twin-clear")

            gr.HTML(FOOTER)

        # Wiring ----------------------------------------------------------
        # Generator handlers stream into a single output (the chatbot);
        # never wrap them in lambdas — Gradio validates output counts.
        send_event = msg.submit(user_submit, [msg, chatbot], [msg, chatbot]).then(
            bot_stream, chatbot, chatbot
        )
        send_btn.click(user_submit, [msg, chatbot], [msg, chatbot]).then(
            bot_stream, chatbot, chatbot
        )

        for btn in chip_buttons:
            (
                btn.click(lambda q: q, [btn], [msg])
                .then(user_submit, [msg, chatbot], [msg, chatbot])
                .then(bot_stream, chatbot, chatbot)
            )

        clear_btn.click(clear_chat, None, [chatbot, msg], cancels=[send_event])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    print(f"Launching {APP_NAME} on port {port}...")
    demo.launch(
        # Spaces require binding to 0.0.0.0; local dev stays loopback-only.
        server_name="0.0.0.0" if DEPLOY_MODE else "127.0.0.1",
        server_port=port,
        show_error=False,
        theme=make_theme(),
        css=CUSTOM_CSS,
        # File-serving hardening: no extra allow-list, and sensitive/dev files
        # can never be reached through /file= routes even if present on disk.
        allowed_paths=[],
        blocked_paths=[".env", "contacts.json", "contacts.csv", ".git", ".venv"],
        # SSR (Node proxy) crashes on HF Spaces with Gradio 6 — serve plain.
        ssr_mode=False,
    )
