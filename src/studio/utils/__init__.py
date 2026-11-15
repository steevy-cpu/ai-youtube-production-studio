"""Small shared helpers: ids, time, text, safe paths, sanitised logging."""
from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path


def new_id(prefix: str = "") -> str:
    short = uuid.uuid4().hex[:12]
    return f"{prefix}_{short}" if prefix else short


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].strip("-") or "project"


def word_count(text: str) -> int:
    return len(text.split())


def split_sentences(text: str) -> list[str]:
    """Deterministic sentence splitter used to map narration onto shots."""
    parts = re.split(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])", " ".join(text.split()))
    return [p.strip() for p in parts if p.strip()]


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def extract_json(text: str) -> dict:
    """Parse a JSON object from untrusted model text (tolerates code fences / preamble)."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("response contains no JSON object") from None
        try:
            value = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"response is not valid JSON: {exc}") from None
    if not isinstance(value, dict):
        raise ValueError("response JSON must be an object")
    return value


def safe_path(base: Path, *parts: str) -> Path:
    """Join path parts under `base` and refuse anything that escapes it."""
    base = Path(base).resolve()
    target = base.joinpath(*parts).resolve()
    if target != base and base not in target.parents:
        raise ValueError(f"path escapes its base directory: {target}")
    return target


# ---- logging -------------------------------------------------------------
_SECRET_PATTERNS = [
    re.compile(r"sk-or-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(Bearer\s+)[A-Za-z0-9_\-\.]{8,}", re.I),
    re.compile(r"((?:api[_-]?key|token|authorization)[\"']?\s*[:=]\s*[\"']?)[^\s\"',}&]{6,}", re.I),
]


def redact(text: str, extra: str = "") -> str:
    if extra and len(extra) >= 6:
        text = text.replace(extra, "[REDACTED]")
    text = _SECRET_PATTERNS[0].sub("[REDACTED]", text)
    for pattern in _SECRET_PATTERNS[1:]:
        text = pattern.sub(lambda m: m.group(1) + "[REDACTED]", text)
    return text


class RedactingFilter(logging.Filter):
    def __init__(self, secret: str = ""):
        super().__init__()
        self.secret = secret

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage(), self.secret)
        record.args = ()
        return True


def get_logger(log_file: Path | None = None, secret: str = "", verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("studio")
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    handlers: list[logging.Handler] = []
    if verbose:
        handlers.append(logging.StreamHandler())
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    if not handlers:
        handlers.append(logging.NullHandler())
    for handler in handlers:
        handler.setFormatter(fmt)
        handler.addFilter(RedactingFilter(secret))
        logger.addHandler(handler)
    return logger
