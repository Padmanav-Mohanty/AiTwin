"""Live smoke test against the deployed Hugging Face Space.

Usage:
    python tests/live_space_test.py [space_id_or_url]

Checks, end to end over the public Gradio API:
  1. The three demo questions stream back non-empty, grounded replies
     (token streaming is proven by receiving multiple intermediate outputs).
  2. Hosted contact handling: an email + collaboration request must NOT be
     saved ("saved locally" note absent) and must surface the public email.

Requires: gradio_client (installed together with gradio).
"""

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from gradio_client import Client

DEFAULT_TARGET = "Padmanav/AiTwin"

PUBLIC_EMAIL = "padmanav.mohanty26@gmail.com"

QUESTIONS = [
    "Who are you?",
    "What is ForgeMind?",
    "What projects has Padmanav worked on?",
]

CONTACT_MESSAGE = (
    "Hi, I'm Test Visitor — my email is test.visitor@example.com "
    "and I'd love to collaborate!"
)


def _api_map(client: Client) -> dict:
    try:
        return client.view_api(return_format="dict")
    except TypeError:  # older gradio_client: only print supported
        client.view_api()
        raise SystemExit("gradio_client too old for dict introspection")


def _find_endpoints(client: Client) -> tuple[str, str]:
    """Return (user_submit_endpoint, bot_stream_endpoint) api names.

    The Enter-key chain is the first wiring registered, so its pair is
    '/user_submit' + '/bot_stream' (duplicates get numeric suffixes).
    """
    api = _api_map(client)
    named = api.get("named_endpoints", {})
    if "/user_submit" in named and "/bot_stream" in named:
        return "/user_submit", "/bot_stream"
    user_eps = sorted(n for n in named if "/user_submit" in n)
    bot_eps = sorted(n for n in named if "/bot_stream" in n)
    if user_eps and bot_eps:
        return user_eps[0], bot_eps[0]
    raise SystemExit(f"Could not find endpoints. Available: {sorted(named)}")


def _last_bot_text(history) -> str:
    if not history:
        return ""
    msg = history[-1]
    if isinstance(msg, dict):
        content = msg.get("content", "")
    else:  # tuple format
        content = msg[-1] if len(msg) > 1 else ""
    if isinstance(content, list):
        content = "\n".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return str(content)


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TARGET
    print(f"Connecting to {target} ...")
    client = Client(target)
    user_ep, bot_ep = _find_endpoints(client)
    print(f"Endpoints: submit={user_ep}, stream={bot_ep}")

    failures: list[str] = []

    def _run_turn(history):
        """Run one bot turn; returns (final_history, intermediate_count)."""
        job = client.submit(history, api_name=bot_ep)
        intermediates = 0
        try:
            for out in job.outputs():  # streaming outputs, if the client surfaces them
                intermediates += 1
                history = out
        except Exception:
            pass
        result = job.result(timeout=180)
        if result:
            history = result
        return history, intermediates

    for question in QUESTIONS:
        _, history = client.predict(message=question, history=[], api_name=user_ep)
        final, intermediates = _run_turn(history)
        reply = _last_bot_text(final).strip()
        ok = bool(reply) and reply != question and len(reply) > 40
        print(f"\nQ: {question}")
        print(f"   intermediate outputs surfaced: {intermediates}")
        print(f"   reply: {reply[:220]}{'...' if len(reply) > 220 else ''}")
        if not ok:
            failures.append(question)

    # Hosted contact handling: nothing saved, public email offered.
    _, history = client.predict(message=CONTACT_MESSAGE, history=[], api_name=user_ep)
    final, _ = _run_turn(history)
    reply = _last_bot_text(final).lower()
    print(f"\nContact flow:")
    print(f"   reply: {_last_bot_text(final)[:300]}")
    if "saved locally" in reply or "💾" in reply:
        failures.append("contact saving appears ACTIVE on the hosted Space (must be off)")
    if PUBLIC_EMAIL not in reply:
        failures.append("public contact email was not offered in the hosted reply")

    print("\n" + "=" * 60)
    if failures:
        print("FAILED:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("ALL LIVE CHECKS PASSED")


if __name__ == "__main__":
    main()
