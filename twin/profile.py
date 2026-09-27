"""Loads profile/profile.json and renders it as compact LLM context.

Profile data is intentionally separate from system-prompt logic so it can be
updated (e.g. from a fresh LinkedIn export) without touching behavior code.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

PROFILE_PATH = Path(__file__).resolve().parent.parent / "profile" / "profile.json"


@lru_cache(maxsize=1)
def load_profile() -> dict[str, Any]:
    """Load the profile JSON (cached). Raises if missing or invalid."""
    with open(PROFILE_PATH, encoding="utf-8") as f:
        return json.load(f)


def render_profile_context() -> str:
    """Render the structured profile as the factual context block."""
    p = load_profile()
    identity = p.get("identity", {})
    skills = p.get("skills", {})

    lines: list[str] = []

    lines.append("IDENTITY")
    lines.append(f"Name: {identity.get('name', '')}")
    lines.append(f"Location: {identity.get('location', '')}")
    lines.append(f"Headline: {identity.get('headline', '')}")
    contact = identity.get("contact", {})
    if contact.get("email"):
        lines.append(f"Email: {contact['email']}")
    if contact.get("linkedin"):
        lines.append(f"LinkedIn: {contact['linkedin']}")

    if p.get("about"):
        lines += ["", "ABOUT", p["about"]]

    if p.get("education"):
        lines += ["", "EDUCATION"]
        for e in p["education"]:
            lines.append(
                f"- {e.get('degree', '')} at {e.get('institution', '')} "
                f"({e.get('period', '')})"
            )

    if skills.get("ai_ml") or skills.get("foundations"):
        lines += ["", "SKILLS"]
        for s in skills.get("ai_ml", []):
            lines.append(f"- {s} (AI/ML)")
        for s in skills.get("foundations", []):
            lines.append(f"- {s} (foundations)")

    if p.get("projects"):
        lines += ["", "PROJECTS"]
        for proj in p["projects"]:
            lines.append(f"- {proj.get('name', '')}: {proj.get('description', '')}")
            for h in proj.get("highlights", []):
                lines.append(f"  * {h}")

    if p.get("experience"):
        lines += ["", "EXPERIENCE"]
        for e in p["experience"]:
            lines.append(f"- {e}")

    if p.get("achievements"):
        lines += ["", "ACHIEVEMENTS"]
        for a in p["achievements"]:
            lines.append(f"- {a}")

    if p.get("certifications"):
        lines += ["", "CERTIFICATIONS"]
        for c in p["certifications"]:
            lines.append(f"- {c}")

    if p.get("interests"):
        lines += ["", "INTERESTS"]
        lines += [", ".join(p["interests"])]

    if p.get("career_goals"):
        lines += ["", "CAREER GOALS", p["career_goals"]]

    return "\n".join(lines)
