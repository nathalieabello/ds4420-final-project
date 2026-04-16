"""Resolve repo root whether Streamlit is run from project root or elsewhere."""

from __future__ import annotations

from pathlib import Path

# streamlit_app/ -> repo root
REPO_ROOT = Path(__file__).resolve().parent.parent


def predictions_dir() -> Path:
    return REPO_ROOT / "Predictions"


def extra_credit_dir() -> Path:
    return REPO_ROOT / "extra_credit"


def extra_credit_data_dir() -> Path:
    return extra_credit_dir() / "data"
