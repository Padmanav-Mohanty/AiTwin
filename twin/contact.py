"""Contact-saving tool for the Digital Twin.

When a visitor asks to be contacted (collaboration, internship, job, ...),
the Digital Twin asks for their email, then records the lead locally on
Padmanav's PC. Data is appended to both JSON and CSV files next to the app.
"""

import csv
import json
import re
import threading
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent
CONTACTS_JSON = DATA_DIR / "contacts.json"
CONTACTS_CSV = DATA_DIR / "contacts.csv"

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_lock = threading.Lock()

_CSV_FIELDS = ["timestamp", "name", "email", "message"]


def extract_email(text: str) -> str | None:
    """Return the first email address found in text, or None."""
    if not isinstance(text, str):
        return None
    match = _EMAIL_RE.search(text)
    return match.group(0) if match else None


def save_contact_info(email: str, name: str = "", message: str = "") -> dict:
    """Persist one contact entry to contacts.json and contacts.csv.

    Returns a small dict summary (never raises for storage issues — storage
    failures are reported in the returned dict so the twin can mention them).
    """
    email = (email or "").strip()
    name = (name or "").strip() or "Not provided"
    message = (message or "").strip() or "Not provided"

    if not extract_email(email):
        return {"saved": False, "reason": "invalid_email"}

    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "name": name,
        "email": email,
        "message": message,
    }

    with _lock:
        try:
            # JSON -----------------------------------------------------------
            entries: list = []
            if CONTACTS_JSON.exists():
                try:
                    entries = json.loads(CONTACTS_JSON.read_text(encoding="utf-8"))
                    if not isinstance(entries, list):
                        entries = []
                except (ValueError, OSError):
                    entries = []
            entries.append(entry)
            CONTACTS_JSON.write_text(
                json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            # CSV ------------------------------------------------------------
            new_file = not CONTACTS_CSV.exists()
            with open(CONTACTS_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
                if new_file:
                    writer.writeheader()
                writer.writerow(entry)
        except OSError as exc:
            entry["saved"] = False
            entry["reason"] = f"storage_error: {exc.__class__.__name__}"
            return entry

    return {"saved": True, **entry}


def load_contacts() -> list:
    """Read all saved contacts (used by tests and potential admin views)."""
    if not CONTACTS_JSON.exists():
        return []
    try:
        data = json.loads(CONTACTS_JSON.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (ValueError, OSError):
        return []
