# Padmanav Mohanty — Digital Twin

A Gradio chatbot that represents **Padmanav Mohanty** and answers questions from
recruiters, developers, and visitors about his background, projects, and skills.

**Stack:** Python + Gradio → OpenRouter → free LLM. No separate frontend, no
database, no agent framework.

**Features:**
- Token-by-token streaming replies (typewriter effect) via OpenRouter SSE
- Contact capture: visitors who ask to connect/collaborate are asked for their
  email; the twin saves the lead locally to `contacts.json` + `contacts.csv`
  (gitignored). Suggested prompt: "🤝 I'd like to collaborate with you! Please contact me."
- Set `GRADIO_SERVER_PORT` to change the port (default 7860)

> **Model note:** the originally requested `openai/gpt-oss-20b:free` was
> **delisted from OpenRouter** (verified September 2026: absent from
> `/api/v1/models`, direct endpoint returns 404). The paid sibling
> `openai/gpt-oss-20b` (~$0.02/M tokens) remains available — set
> `TWIN_MODEL=openai/gpt-oss-20b` in `.env` to use it.

## Project structure

```
Digital twin/
├── app.py               # Gradio UI (Blocks) + event wiring
├── profile/
│   └── profile.json     # profile facts (identity, education, skills, projects…)
├── twin/
│   ├── __init__.py
│   ├── config.py        # env config — API key lives only here
│   ├── prompt.py        # system prompt: identity, behavior, security
│   ├── profile.py       # loads profile.json, renders LLM context
│   └── llm.py           # OpenRouter client + friendly error mapping
├── tests/
│   └── test_twin.py     # pytest suite
├── requirements.txt
├── .env.example         # copy to .env, add your key
└── README.md
```

The **system prompt** (behavior/identity) and the **profile** (facts) are
separate by design: update `profile/profile.json` anytime without touching code.

## Run locally

```bash
cd "Digital twin"

python -m venv .venv
.venv\Scripts\activate            # Windows (.venv/bin/activate on macOS/Linux)

pip install -r requirements.txt

copy .env.example .env            # then paste your OpenRouter key into .env

python app.py                     # opens http://127.0.0.1:7860
```

## Tests

```bash
.venv\Scripts\python.exe -m pytest tests -q
```

## Deployment (Hugging Face Space)

Deployed at **https://huggingface.co/spaces/Padmanav/AiTwin** (Gradio SDK,
ZeroGPU hardware — free accounts are ZeroGPU-only as of July 2026).

- Secrets (`OPENROUTER_API_KEY`, `GROQ_API_KEY`) are configured as **Space
  secrets**, never committed; `.env` is never uploaded.
- **Contact saving is disabled on the Space** (auto-detected via `SPACE_ID`):
  nothing visitors type is stored or logged; the collaboration flow points to
  the public contact email instead. Locally, saving works as before.
- The `@spaces.GPU` marker in `app.py` satisfies ZeroGPU startup detection
  without wiring any GPU usage into the chat (the app is a pure API client).
- Deploy payload: `app.py`, `twin/`, `profile/profile.json`, `assets/avatar.svg`,
  plus `deploy/` (Space README, requirements, defense-in-depth `.gitignore`).
  Simulate hosted behavior locally with `TWIN_DEPLOY_MODE=1`.

## Security notes

- The API key stays in `.env` (gitignored) and is only read server-side.
- The system prompt and raw profile context are never revealed in chat.
- Identity-injection attempts ("You are Claude") are refused; the twin keeps
  its canonical identity as Padmanav's Digital Twin.
- History is bounded (16 messages) so requests never grow indefinitely.
- Internal errors never enter the chat history; users see friendly messages.

## Roadmap (not built, by design)

RAG over documents, agents/LangGraph, persistent memory.
Ask before adding anything beyond this milestone.
