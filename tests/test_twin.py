"""Tests for the Digital Twin core (twin/ package and app handlers)."""

import json
from collections.abc import AsyncIterator  # noqa: F401

import pytest

import app as app_mod
from app import bot_stream, user_submit
from twin.config import MAX_HISTORY_MESSAGES
from twin.contact import extract_email, save_contact_info
from twin.llm import TwinLLMError, generate_reply
from twin.profile import load_profile, render_profile_context
from twin.prompt import build_system_prompt


async def collect(agen) -> list:
    """Consume an async generator into a list."""
    out = []
    async for item in agen:
        out.append(item)
    return out


async def _noop_sleep(_):
    return None


# ---------------------------------------------------------------- profile


def test_profile_loads():
    p = load_profile()
    assert p["identity"]["name"] == "Padmanav Mohanty"
    assert p["education"], "education should not be empty"
    assert p["projects"], "projects should not be empty"


def test_profile_context_contains_key_facts():
    ctx = render_profile_context()
    for fact in ("Padmanav Mohanty", "KIIT", "ForgeMind", "Document RAG", "QLoRA"):
        assert fact in ctx


def test_document_rag_is_simple_rag():
    ctx = render_profile_context()
    assert "simple" in ctx.lower()
    assert "BM25" not in ctx and "RRF" not in ctx and "cross-encoder" not in ctx.lower()


def test_profile_has_no_experience_or_achievements():
    p = load_profile()
    assert p["experience"] == []
    assert p["achievements"] == []


# ---------------------------------------------------------------- prompt


def test_system_prompt_contains_identity_and_security():
    sp = build_system_prompt()
    assert "Digital Twin" in sp
    assert "Padmanav Mohanty" in sp
    assert "Never reveal" in sp
    assert "You are NOT" in sp or "never claim to be" in sp


def test_system_prompt_embeds_profile_context():
    sp = build_system_prompt()
    assert "FACTUAL PROFILE CONTEXT" in sp
    assert "ForgeMind" in sp


def test_system_prompt_has_contact_rules():
    sp = build_system_prompt()
    assert "CONTACT REQUESTS" in sp


def test_deploy_prompt_never_claims_email_saved(monkeypatch):
    import twin.prompt as prompt_mod

    # build_system_prompt reads DEPLOY_MODE from its own module at call time.
    monkeypatch.setattr(prompt_mod, "DEPLOY_MODE", True)
    sp = prompt_mod.build_system_prompt()
    assert "NEVER claim their email was saved" in sp
    assert "padmanav.mohanty26@gmail.com" in sp
    assert "record it" not in sp


def test_local_prompt_keeps_save_flow():
    sp = build_system_prompt()
    assert "record it" in sp
    assert "NEVER claim their email was saved" not in sp


# ---------------------------------------------------------------- config


def test_model_configurable_via_env(monkeypatch):
    monkeypatch.setenv("TWIN_MODEL", "test/model-from-env:free")
    import importlib

    import twin.config as cfg

    importlib.reload(cfg)
    assert cfg.MODEL == "test/model-from-env:free"
    importlib.reload(cfg)  # restore for other tests


def test_api_key_not_hardcoded():
    import pathlib
    import re

    for f in ("twin/config.py", "twin/llm.py", "app.py"):
        src = pathlib.Path(f).read_text(encoding="utf-8")
        assert not re.search(r"sk-or-v1-[A-Za-z0-9]{20,}", src), f"hardcoded key in {f}"


# ---------------------------------------------------------------- contact tool


def test_extract_email():
    assert extract_email("my email is alice@example.com") == "alice@example.com"
    assert extract_email("no email here") is None
    assert extract_email(None) is None  # type: ignore[arg-type]
    assert extract_email(["not", "a", "string"]) is None  # type: ignore[arg-type]


def test_save_contact_info_writes_json_and_csv(monkeypatch, tmp_path):
    import twin.contact as contact

    jpath = tmp_path / "contacts.json"
    cpath = tmp_path / "contacts.csv"
    monkeypatch.setattr(contact, "CONTACTS_JSON", jpath)
    monkeypatch.setattr(contact, "CONTACTS_CSV", cpath)

    result = save_contact_info("alice@example.com", "Alice", "wants to collaborate")
    assert result["saved"] is True

    entries = json.loads(jpath.read_text(encoding="utf-8"))
    assert entries[-1]["email"] == "alice@example.com"
    assert entries[-1]["name"] == "Alice"
    assert "timestamp" in entries[-1]

    csv_text = cpath.read_text(encoding="utf-8")
    assert "alice@example.com" in csv_text and "name" in csv_text

    # Second save appends.
    save_contact_info("bob@example.com", "Bob", "internship")
    entries = json.loads(jpath.read_text(encoding="utf-8"))
    assert len(entries) == 2


def test_save_contact_info_rejects_bad_email(monkeypatch, tmp_path):
    import twin.contact as contact

    monkeypatch.setattr(contact, "CONTACTS_JSON", tmp_path / "c.json")
    monkeypatch.setattr(contact, "CONTACTS_CSV", tmp_path / "c.csv")
    result = save_contact_info("not-an-email")
    assert result.get("saved") is False


def test_maybe_save_contact_triggers_on_intent(monkeypatch):
    import twin.contact as contact

    monkeypatch.setattr(contact, "CONTACTS_JSON", app_mod.Path("_t.json"))
    monkeypatch.setattr(contact, "CONTACTS_CSV", app_mod.Path("_t.csv"))

    history = [
        {"role": "user", "content": "I'd like to collaborate. Please contact me!"},
        {"role": "assistant", "content": "Sure — what's your email?"},
        {"role": "user", "content": "My name is Eve and my email is eve@example.com"},
    ]
    result = app_mod._maybe_save_contact(history)
    assert result is not None and result.get("saved") is True
    assert result["email"] == "eve@example.com"
    assert result["name"] == "Eve"

    app_mod.Path("_t.json").unlink(missing_ok=True)
    app_mod.Path("_t.csv").unlink(missing_ok=True)


def test_maybe_save_contact_ignores_casual_email(monkeypatch):
    history = [
        {"role": "user", "content": "What does an email address like a@b.com look like?"}
    ]
    assert app_mod._maybe_save_contact(history) is None


def test_maybe_save_contact_ignores_no_email():
    history = [{"role": "user", "content": "Please contact me!"}]
    assert app_mod._maybe_save_contact(history) is None


# ---------------------------------------------------------------- app handlers


def _assistant_only_history():
    return [{"role": "assistant", "content": "welcome"}]


def test_user_submit_appends_and_clears():
    msg, history = user_submit("hello", [])
    assert msg == ""
    assert history[-1] == {"role": "user", "content": "hello"}


@pytest.mark.asyncio
async def test_bot_stream_requires_message():
    with pytest.raises(Exception):
        await collect(bot_stream([{"role": "assistant", "content": "welcome"}]))


@pytest.mark.asyncio
async def test_bot_stream_rejects_overlong_message(monkeypatch):
    monkeypatch.setattr(app_mod, "api_key_configured", lambda: True)
    history = _assistant_only_history() + [
        {"role": "user", "content": "x" * (app_mod.MAX_MESSAGE_CHARS + 1)}
    ]
    with pytest.raises(Exception):
        await collect(bot_stream(history))


@pytest.mark.asyncio
async def test_bot_stream_bounds_history(monkeypatch):
    monkeypatch.setattr(app_mod, "api_key_configured", lambda: True)

    long_history = [{"role": "user", "content": "hi"}]
    for i in range(60):
        long_history.append({"role": "assistant", "content": f"reply {i}"})
        long_history.append({"role": "user", "content": f"question {i}"})

    seen = {}

    async def fake_stream(messages):
        seen["messages"] = messages
        yield "ok"

    monkeypatch.setattr(app_mod, "generate_reply_stream", fake_stream)
    yielded = await collect(bot_stream(long_history))

    conversation = [
        m for m in seen["messages"] if m["role"] != "system"
    ]
    assert len(conversation) <= MAX_HISTORY_MESSAGES
    assert conversation[0]["role"] == "user"
    # Streaming: multiple yields, each history ends with a growing reply.
    assert len(yielded) >= 1
    assert yielded[-1][-1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_bot_stream_returns_full_reply(monkeypatch):
    monkeypatch.setattr(app_mod, "api_key_configured", lambda: True)

    async def fake_stream(messages):
        assert messages[0]["role"] == "system"
        for word in ("I", "am", "Padmanav's", "Digital", "Twin."):
            yield word + " "

    monkeypatch.setattr(app_mod, "generate_reply_stream", fake_stream)

    yielded = await collect(bot_stream([{"role": "user", "content": "Who are you?"}]))
    final = yielded[-1]
    assert final[-1]["role"] == "assistant"
    assert "Digital Twin" in final[-1]["content"]


@pytest.mark.asyncio
async def test_bot_stream_appends_contact_note(monkeypatch):
    monkeypatch.setattr(app_mod, "api_key_configured", lambda: True)

    import twin.contact as contact

    monkeypatch.setattr(contact, "CONTACTS_JSON", app_mod.Path("_t2.json"))
    monkeypatch.setattr(contact, "CONTACTS_CSV", app_mod.Path("_t2.csv"))

    async def fake_stream(messages):
        yield "Sure, noted!"

    monkeypatch.setattr(app_mod, "generate_reply_stream", fake_stream)

    history = [
        {"role": "user", "content": "I'd like to collaborate, please contact me"},
        {"role": "assistant", "content": "Happy to! What's your email?"},
        {"role": "user", "content": "My email is eve@example.com"},
    ]
    yielded = await collect(bot_stream(history))
    final_content = yielded[-1][-1]["content"]
    assert "saved" in final_content.lower()

    app_mod.Path("_t2.json").unlink(missing_ok=True)
    app_mod.Path("_t2.csv").unlink(missing_ok=True)


# ---------------------------------------------------------------- llm client


@pytest.mark.asyncio
async def test_llm_raises_friendly_error_without_key(monkeypatch):
    import twin.llm as llm

    monkeypatch.setattr(llm, "API_KEY", "")
    monkeypatch.setattr(llm, "GROQ_API_KEY", "")  # hermetic: no real providers
    with pytest.raises(TwinLLMError) as exc_info:
        await generate_reply([{"role": "user", "content": "hi"}])
    assert "OPENROUTER_API_KEY" in str(exc_info.value)
    assert "sk-" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_llm_maps_500_to_friendly_error(monkeypatch):
    import twin.llm as llm

    monkeypatch.setattr(llm, "API_KEY", "test-key")

    class FakeResponse:
        status_code = 500

        def json(self):
            return {}

    class FakeAsyncClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            return FakeResponse()

    monkeypatch.setattr(llm.httpx, "AsyncClient", FakeAsyncClient)
    with pytest.raises(TwinLLMError) as exc_info:
        await generate_reply([{"role": "user", "content": "hi"}])
    assert "provider had a problem" in str(exc_info.value)


# ------------------------------------------------------------- provider cascade


def _provider_lists():
    import twin.llm as llm

    return llm._providers()


def test_providers_openrouter_only_without_groq_key(monkeypatch):
    import twin.llm as llm

    monkeypatch.setattr(llm, "API_KEY", "or-key")
    monkeypatch.setattr(llm, "GROQ_API_KEY", "")
    providers = _provider_lists()
    assert [p["name"] for p in providers] == ["openrouter"]


def test_providers_include_groq_when_keyed(monkeypatch):
    import twin.llm as llm

    monkeypatch.setattr(llm, "API_KEY", "or-key")
    monkeypatch.setattr(llm, "GROQ_API_KEY", "gq-key")
    providers = _provider_lists()
    assert [p["name"] for p in providers] == ["openrouter", "groq"]
    assert providers[1]["model"] == llm.GROQ_MODEL
    assert providers[1]["url"] == "https://api.groq.com/openai/v1/chat/completions"


@pytest.mark.asyncio
async def test_generate_reply_falls_back_to_groq(monkeypatch):
    import twin.llm as llm

    monkeypatch.setattr(llm, "API_KEY", "or-key")
    monkeypatch.setattr(llm, "GROQ_API_KEY", "gq-key")

    calls = []

    class FakeResponse:
        def __init__(self, status):
            self.status_code = status

        def json(self):
            return {
                "choices": [{"message": {"content": f"reply-via-{calls[-1][0]}"}}]
            }

    class FakeAsyncClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, **k):
            name = "groq" if "groq" in url else "openrouter"
            calls.append((name, url))
            return FakeResponse(429 if name == "openrouter" else 200)

    monkeypatch.setattr(llm.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(llm.asyncio, "sleep", _noop_sleep)

    reply = await llm.generate_reply([{"role": "user", "content": "hi"}])
    assert reply == "reply-via-groq"
    assert [c[0] for c in calls] == ["openrouter", "openrouter", "groq"]


@pytest.mark.asyncio
async def test_stream_falls_back_before_first_token(monkeypatch):
    import twin.llm as llm

    monkeypatch.setattr(llm, "API_KEY", "or-key")
    monkeypatch.setattr(llm, "GROQ_API_KEY", "gq-key")

    attempted = []

    async def fake_stream_once(provider, messages):
        attempted.append(provider["name"])
        if provider["name"] == "openrouter":
            raise llm.TwinLLMError("Rate limit reached — please wait a moment and try again.")
        yield "hello "
        yield "from groq"

    monkeypatch.setattr(llm, "_stream_once", fake_stream_once)

    out = [d async for d in llm.generate_reply_stream([{"role": "user", "content": "hi"}])]
    assert attempted == ["openrouter", "groq"]
    assert "".join(out) == "hello from groq"


@pytest.mark.asyncio
async def test_stream_does_not_restart_after_partial_output(monkeypatch):
    import twin.llm as llm

    monkeypatch.setattr(llm, "API_KEY", "or-key")
    monkeypatch.setattr(llm, "GROQ_API_KEY", "gq-key")

    attempted = []

    async def fake_stream_once(provider, messages):
        attempted.append(provider["name"])
        yield "partial "
        raise llm.TwinLLMError("The model returned an empty response. Please try again.")

    monkeypatch.setattr(llm, "_stream_once", fake_stream_once)

    out = []
    with pytest.raises(TwinLLMError):
        async for d in llm.generate_reply_stream([{"role": "user", "content": "hi"}]):
            out.append(d)
    # Fallback must NOT be attempted once tokens were already shown.
    assert attempted == ["openrouter"]
    assert "".join(out) == "partial "


# ---------------------------------------------------------------- deploy mode


def _reload_cfg(monkeypatch, **env):
    import importlib

    import twin.config as cfg

    for key in ("SPACE_ID", "TWIN_DEPLOY_MODE"):
        monkeypatch.delenv(key, raising=False)
    for key, val in env.items():
        monkeypatch.setenv(key, val)
    importlib.reload(cfg)
    return cfg


def test_deploy_mode_off_locally(monkeypatch):
    import importlib

    cfg = _reload_cfg(monkeypatch)
    assert cfg.DEPLOY_MODE is False
    importlib.reload(cfg)  # restore real environment for other tests


def test_deploy_mode_on_for_space(monkeypatch):
    import importlib

    cfg = _reload_cfg(monkeypatch, SPACE_ID="Padmanav/AiTwin")
    assert cfg.DEPLOY_MODE is True
    importlib.reload(cfg)  # restore


def test_deploy_mode_env_override(monkeypatch):
    import importlib

    cfg = _reload_cfg(monkeypatch, TWIN_DEPLOY_MODE="1")
    assert cfg.DEPLOY_MODE is True
    importlib.reload(cfg)  # restore


@pytest.mark.asyncio
async def test_deploy_mode_skips_contact_save_and_shows_public_email(monkeypatch):
    monkeypatch.setattr(app_mod, "api_key_configured", lambda: True)
    monkeypatch.setattr(app_mod, "DEPLOY_MODE", True)

    saves = []

    def _spy_save(history):
        saves.append(history)
        return {"saved": True}

    monkeypatch.setattr(app_mod, "_maybe_save_contact", _spy_save)

    async def fake_stream(messages):
        yield "Happy to collaborate!"

    monkeypatch.setattr(app_mod, "generate_reply_stream", fake_stream)

    history = [
        {"role": "user", "content": "I'd like to collaborate! My email is eve@example.com"},
    ]
    yielded = await collect(bot_stream(history))
    content = yielded[-1][-1]["content"]

    assert not saves, "contact saving must be disabled in deploy mode"
    assert "padmanav.mohanty26@gmail.com" in content
    assert "saved locally" not in content.lower()


@pytest.mark.asyncio
async def test_local_mode_still_saves_contacts(monkeypatch):
    monkeypatch.setattr(app_mod, "api_key_configured", lambda: True)
    monkeypatch.setattr(app_mod, "DEPLOY_MODE", False)

    import twin.contact as contact

    monkeypatch.setattr(contact, "CONTACTS_JSON", app_mod.Path("_t3.json"))
    monkeypatch.setattr(contact, "CONTACTS_CSV", app_mod.Path("_t3.csv"))

    async def fake_stream(messages):
        yield "Noted!"

    monkeypatch.setattr(app_mod, "generate_reply_stream", fake_stream)

    history = [
        {"role": "user", "content": "Please contact me — eve@example.com"},
    ]
    yielded = await collect(bot_stream(history))
    content = yielded[-1][-1]["content"]
    assert "saved" in content.lower()

    app_mod.Path("_t3.json").unlink(missing_ok=True)
    app_mod.Path("_t3.csv").unlink(missing_ok=True)
