"""Paths and environment-driven settings."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SOURCES_DIR = DATA_DIR / "sources"

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"


def together_api_key() -> str | None:
    key = os.getenv("TOGETHER_API_KEY", "").strip()
    return key or None


def together_model() -> str:
    return os.getenv("TOGETHER_MODEL", "").strip() or DEFAULT_MODEL


def reference_date() -> date:
    """Date used as "now" when computing freshness (override for reproducible demos)."""
    raw = os.getenv("PROVRAG_REFERENCE_DATE", "").strip()
    return date.fromisoformat(raw) if raw else date.today()
