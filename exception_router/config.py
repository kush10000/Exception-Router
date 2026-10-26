"""Configuration: env loading, model choice, API client."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTBOX_DIR = ROOT / "outbox"
RESULTS_DIR = ROOT / "results"

load_dotenv(ROOT / ".env")  # The SDK only reads os.environ, so .env must be loaded explicitly.

MODEL = os.environ.get("ROUTER_MODEL", "claude-sonnet-5-5")
MAX_TOKENS = 2048


def make_client():
    """Anthropic client. Sends anthropic-workspace-id only if ANTHROPIC_WORKSPACE_ID is set
    (needed for API keys that are not scoped to a workspace)."""
    import anthropic

    headers = {}
    workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    if workspace:
        headers["anthropic-workspace-id"] = workspace
    return anthropic.Anthropic(default_headers=headers or None, max_retries=4)
