"""Shared utilities for llama-* tools."""

import re
import sys
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm

REPO_ROOT  = Path(__file__).parent.parent.resolve()
MODELS_YML = REPO_ROOT / "models.yml"
LOG_DIR    = REPO_ROOT / "logs"

console = Console()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def resolve_str(value: str, cwd: str, models_dir: str) -> str:
    return value.replace("${cwd}", cwd).replace("${models_dir}", models_dir)


def resolve_dict(d: dict, cwd: str, models_dir: str) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, str):
            out[k] = resolve_str(v, cwd, models_dir)
        elif isinstance(v, dict):
            out[k] = resolve_dict(v, cwd, models_dir)
        else:
            out[k] = v
    return out


def load_config() -> tuple[dict, dict]:
    """Return (resolved_defaults, resolved_models)."""
    with open(MODELS_YML) as f:
        raw = yaml.safe_load(f)

    cwd = str(REPO_ROOT)
    defaults_raw = raw.get("defaults", {})
    models_dir_raw = defaults_raw.get("models_dir", "${cwd}/models")
    models_dir = resolve_str(models_dir_raw, cwd, cwd)

    defaults = resolve_dict(defaults_raw, cwd, models_dir)
    defaults["models_dir"] = models_dir

    models: dict[str, dict] = {}
    for name, cfg in (raw.get("models") or {}).items():
        models[name] = resolve_dict(cfg or {}, cwd, models_dir)

    return defaults, models


# ---------------------------------------------------------------------------
# Model selection helpers
# ---------------------------------------------------------------------------

def resolve_model(raw: str, model_names: list[str], models: dict) -> str | None:
    if raw in models:
        return raw
    try:
        return model_names[int(raw) - 1]
    except (ValueError, IndexError):
        return None


def select_model_interactive(models: dict, defaults: dict) -> str | None:
    """Show model table and prompt. Returns model name or None to quit."""
    model_names = list(models.keys())
    console.print()
    console.print(models_table(models, defaults))
    console.print()
    choices = [str(i) for i in range(1, len(model_names) + 1)] + model_names + ["q"]
    raw = Prompt.ask(
        "[bold]Select model[/bold] [dim](q to quit)[/dim]",
        choices=choices, default="1", show_choices=False,
    )
    if raw.lower() == "q":
        return None
    return resolve_model(raw, model_names, models)


# ---------------------------------------------------------------------------
# Rich helpers
# ---------------------------------------------------------------------------

def models_table(models: dict, defaults: dict) -> Table:
    t = Table(show_header=True, header_style="bold cyan", border_style="dim")
    t.add_column("#",    style="dim", width=3, justify="right")
    t.add_column("Name", style="bold white")
    t.add_column("File", style="green")
    t.add_column("Ctx",  justify="right", style="yellow")
    t.add_column("Port", justify="right", style="cyan")
    t.add_column("",     style="dim")

    for i, (name, cfg) in enumerate(models.items(), 1):
        path   = cfg.get("path", "")
        fname  = Path(path).name if path else "?"
        ctx    = str(cfg.get("ctx_size",  defaults.get("ctx_size",  "?")))
        port   = str(cfg.get("port",      defaults.get("port",      "?")))
        exists = Path(path).exists() if path else False
        t.add_row(str(i), name, fname, ctx, port, "" if exists else "[red]not found[/red]")

    return t
