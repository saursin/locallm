#!/usr/bin/env python3
"""llama-serve — Rich frontend for llama-server."""

import argparse
import re
import signal
import sys
import subprocess
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import yaml
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.text import Text

from llama_utils import (
    REPO_ROOT, MODELS_YML, LOG_DIR, console,
    load_config, resolve_model, select_model_interactive, models_table,
)

LLAMA_SERVER = REPO_ROOT / "llama.cpp/build/bin/llama-server"


# ---------------------------------------------------------------------------
# Argument builder
# ---------------------------------------------------------------------------

SERVER_PARAM_MAP: dict[str, str] = {
    "port":          "--port",
    "ctx_size":      "--ctx-size",
    "batch_size":    "--batch-size",
    "ubatch_size":   "--ubatch-size",
    "parallel":      "--parallel",
    "cache_k":       "--cache-type-k",
    "cache_v":       "--cache-type-v",
    "threads":       "--threads",
    "threads_batch": "--threads-batch",
    "n_gpu_layers":  "--n-gpu-layers",
}

SERVER_BOOL_MAP: dict[str, str] = {
    "kv_unified": "--kv-unified",
    "jinja":      "--jinja",
}

SERVER_ONOFF_MAP: dict[str, str] = {
    "flash_attn": "--flash-attn",
}

INFERENCE_PARAM_MAP: dict[str, str] = {
    "temperature":    "--temp",
    "top_p":          "--top-p",
    "repeat_penalty": "--repeat-penalty",
    "max_new_tokens": "--n-predict",
}


def build_server_args(defaults: dict, model_cfg: dict) -> list[str]:
    merged: dict[str, Any] = {**defaults}
    for k, v in model_cfg.items():
        if k not in ("path", "inference"):
            merged[k] = v

    inference: dict[str, Any] = {**defaults.get("inference", {})}
    inference.update(model_cfg.get("inference", {}))

    args: list[str] = []
    for key, flag in SERVER_PARAM_MAP.items():
        if (val := merged.get(key)) is not None:
            args += [flag, str(val)]
    for key, flag in SERVER_BOOL_MAP.items():
        if merged.get(key):
            args.append(flag)
    for key, flag in SERVER_ONOFF_MAP.items():
        if (val := merged.get(key)) is not None:
            args += [flag, "on" if val else "off"]
    for key, flag in INFERENCE_PARAM_MAP.items():
        if (val := inference.get(key)) is not None:
            args += [flag, str(val)]
    return args


# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------

def discover_models(models_dir: str, known_paths: set[str]) -> list[Path]:
    base = Path(models_dir)
    if not base.exists():
        return []
    return sorted(f for f in base.rglob("*.gguf") if str(f) not in known_paths)


def add_model_to_yml(name: str, path: str) -> None:
    with open(MODELS_YML) as f:
        content = f.read()

    m = re.search(r'models_dir:\s*["\']?([^\n"\']+)["\']?', content)
    raw_models_dir = m.group(1).strip() if m else None
    display_path = path
    if raw_models_dir:
        from llama_utils import resolve_str
        resolved_dir = resolve_str(raw_models_dir, str(REPO_ROOT), str(REPO_ROOT / "models"))
        try:
            rel = Path(path).relative_to(resolved_dir)
            display_path = f"${{models_dir}}/{rel}"
        except ValueError:
            pass

    content = content.rstrip() + f"\n  {name}:\n    path: \"{display_path}\"\n"
    with open(MODELS_YML, "w") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Server runner
# ---------------------------------------------------------------------------

def run_server(
    selected: str, cmd: list[str], port: int, log_path: Path,
    model_cfg: dict, defaults: dict,
) -> str:
    """Run llama-server with split status + log live display. Returns 'change' or 'quit'."""
    log_h = max(6, console.height // 2)
    max_w = max(20, console.width - 4)
    lines: deque[str] = deque(maxlen=log_h - 4)
    info: dict[str, Any] = {"requests": 0}
    done = threading.Event()
    action = ["change"]

    def _parse(line: str) -> None:
        if m := re.search(r"n_threads = (\d+) \(n_threads_batch = (\d+)\) / (\d+)", line):
            info["threads"] = f"{m.group(1)} gen  /  {m.group(2)} batch  [dim](of {m.group(3)})[/dim]"
        if m := re.search(r"model params\s*=\s*([\d.]+ \w+)", line):
            info["model_params"] = m.group(1).strip()
        if m := re.search(r"file size\s*=\s*([\d.]+ \w+)", line):
            info["file_size"] = m.group(1).strip()
        if m := re.search(r"general\.name\s*=\s*(.+)", line):
            info["model_name"] = m.group(1).strip()
        if "CPU_Mapped" in line:
            info.setdefault("device", "CPU")
        elif "GPU" in line and "buffer size" in line:
            info["device"] = "GPU"
        for flag, label in [("AVX2", "AVX2"), ("AVX_VNNI", "AVX-VNNI"), ("FMA", "FMA")]:
            if re.search(rf"{flag} = 1", line):
                info.setdefault("cpu_flags", set()).add(label)
        if m := re.search(r"llama_kv_cache: size =\s*([\d.]+ \w+)", line):
            info["kv_size"] = m.group(1).strip()
        if "server is listening" in line:
            info["listening"] = True
        if re.search(r"slot\s+launch_slot_.*processing task", line):
            info["busy"] = True
        if "update_slots: all slots are idle" in line:
            info["busy"] = False
        if re.search(r"slot\s+release:", line):
            info["requests"] = info.get("requests", 0) + 1
        if "eval time" in line and "prompt eval" not in line:
            if m := re.search(r"([\d.]+) tokens per second", line):
                info["tok_per_sec"] = float(m.group(1))

    def _reader(proc: subprocess.Popen, log_file: Any) -> None:
        for raw in proc.stdout:  # type: ignore[union-attr]
            line = raw.rstrip("\n")
            _parse(line)
            lines.append(line[:max_w - 1] + "…" if len(line) > max_w else line)
            log_file.write(raw)
            log_file.flush()
        done.set()

    def _status_panel() -> Panel:
        tok_s = info.get("tok_per_sec")
        speed = f"  [dim]{tok_s:.1f} tok/s[/dim]" if tok_s else ""
        if info.get("busy"):
            status = f"[green]● busy{speed}[/green]"
        elif info.get("listening"):
            status = f"[green]● running{speed}[/green]"
        elif done.is_set():
            status = "[red]● stopped[/red]"
        else:
            status = "[yellow]● starting…[/yellow]"

        model_detail = f"[bold]{selected}[/bold]"
        if meta_name := info.get("model_name", ""):
            model_detail += f"  [dim]{meta_name}[/dim]"
        if meta := "  ".join(filter(None, [info.get("model_params"), info.get("file_size")])):
            model_detail += f"  [dim]({meta})[/dim]"

        n_gpu  = int(model_cfg.get("n_gpu_layers", defaults.get("n_gpu_layers", 0)) or 0)
        device = info.get("device", "CPU" if n_gpu == 0 else f"GPU ({n_gpu} layers)")
        if "threads" in info:
            device += f"  ·  {info['threads']} threads"

        ctx      = model_cfg.get("ctx_size",   defaults.get("ctx_size",   "?"))
        batch    = model_cfg.get("batch_size", defaults.get("batch_size", "?"))
        parallel = model_cfg.get("parallel",   defaults.get("parallel",   "?"))
        kv_k     = model_cfg.get("cache_k",    defaults.get("cache_k",    "?"))
        kv_v     = model_cfg.get("cache_v",    defaults.get("cache_v",    "?"))
        fa       = model_cfg.get("flash_attn", defaults.get("flash_attn", False))
        kv_size  = info.get("kv_size", "")

        g = Table.grid(padding=(0, 2))
        g.add_column(style="dim", min_width=11)
        g.add_column()
        g.add_row("Status",   status)
        g.add_row("Model",    model_detail)
        g.add_row("URL",      f"[cyan]http://localhost:{port}[/cyan]")
        g.add_row("Device",   device)
        if cpu_flags := info.get("cpu_flags", set()):
            g.add_row("CPU", "  ".join(sorted(cpu_flags)))
        g.add_row("Context",  f"{ctx}   [dim]Batch[/dim] {batch}   [dim]Parallel[/dim] {parallel}")
        kv_str = f"K: {kv_k}   V: {kv_v}"
        if kv_size:
            kv_str += f"  [dim]({kv_size})[/dim]"
        kv_str += f"   [dim]Flash attn:[/dim] {'on' if fa else 'off'}"
        g.add_row("KV Cache", kv_str)
        g.add_row("Requests", str(info.get("requests", 0)))
        return Panel(g, title="[bold]Status[/bold]", border_style="cyan")

    def _log_panel() -> Panel:
        return Panel(
            Text("\n".join(lines) or "…", style="dim", no_wrap=True, overflow="ellipsis"),
            title="[dim]server log[/dim]",
            subtitle=f"[dim]Ctrl+C to stop  ·  {log_path.name}[/dim]",
            border_style="dim",
            height=log_h,
        )

    layout = Layout()
    layout.split_column(Layout(name="status"), Layout(name="log", size=log_h))
    LOG_DIR.mkdir(exist_ok=True)

    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        # Ensure server is killed if this script is terminated via SIGTERM
        _prev_sigterm = signal.getsignal(signal.SIGTERM)
        def _sigterm(signum, frame):
            proc.terminate()
            signal.signal(signal.SIGTERM, _prev_sigterm)
            sys.exit(0)
        signal.signal(signal.SIGTERM, _sigterm)

        threading.Thread(target=_reader, args=(proc, log_file), daemon=True).start()
        try:
            with Live(layout, console=console, refresh_per_second=4, screen=False) as live:
                while not done.is_set():
                    layout["status"].update(_status_panel())
                    layout["log"].update(_log_panel())
                    live.refresh()
                    done.wait(timeout=0.25)
                layout["status"].update(_status_panel())
                layout["log"].update(_log_panel())
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
        finally:
            signal.signal(signal.SIGTERM, _prev_sigterm)

    return action[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="llama-serve", description="Frontend for llama-server.")
    p.add_argument("model",      nargs="?", help="Model name or 1-based index")
    p.add_argument("--list",     "-l", action="store_true", help="List models and exit")
    p.add_argument("--discover", "-d", action="store_true", help="Scan models_dir for new .gguf files")
    p.add_argument("--dry-run",        action="store_true", help="Print command without running")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    console.print(Panel.fit(
        Text("llama-serve", style="bold cyan") + Text("  ·  local model server frontend", style="dim"),
        border_style="cyan",
    ))

    if not LLAMA_SERVER.exists():
        console.print(f"\n[red]llama-server not found:[/red] {LLAMA_SERVER}")
        console.print("[yellow]Run 'make' from the repo root first.[/yellow]\n")
        sys.exit(1)

    defaults, models = load_config()
    models_dir = defaults.get("models_dir", str(REPO_ROOT / "models"))

    if args.list:
        console.print(models_table(models, defaults))
        sys.exit(0)

    known_paths = {cfg.get("path", "") for cfg in models.values()}
    new_files = discover_models(models_dir, known_paths)

    if new_files:
        console.print(f"\n[bold yellow]Discovered {len(new_files)} unregistered model(s):[/bold yellow]")
        for f in new_files:
            console.print(f"  [dim]{f}[/dim]")
        console.print()
        if args.discover or args.model is None:
            for f in new_files:
                if args.discover or Confirm.ask(f"  Add [bold green]{f.name}[/bold green] to models.yml?", default=False):
                    name = Prompt.ask("    Name", default=f.stem)
                    add_model_to_yml(name, str(f))
                    console.print(f"    [green]Added '{name}'[/green]")
                    defaults, models = load_config()
            console.print()
        if args.discover:
            sys.exit(0)

    if not models:
        console.print("[red]No models configured.[/red]")
        sys.exit(1)

    model_names = list(models.keys())
    initial: str | None = None
    if args.model is not None:
        initial = resolve_model(args.model, model_names, models)
        if not initial:
            console.print(f"[red]Unknown model:[/red] {args.model!r}")
            console.print(f"[dim]Available: {', '.join(model_names)}[/dim]")
            sys.exit(1)

    selected = initial
    while True:
        defaults, models = load_config()
        model_names = list(models.keys())

        if selected is None:
            try:
                selected = select_model_interactive(models, defaults)
            except KeyboardInterrupt:
                break
            if selected is None:
                break

        model_cfg  = models.get(selected, {})
        model_path = model_cfg.get("path", "")
        if not model_path or not Path(model_path).exists():
            console.print(f"[red]Model file not found:[/red] {model_path}")
            selected = None
            continue

        extra = build_server_args(defaults, model_cfg)
        cmd   = [str(LLAMA_SERVER), "--model", model_path] + extra
        port  = int(model_cfg.get("port", defaults.get("port", 8080)))

        if args.dry_run:
            console.print(f"[dim]{' '.join(cmd)}[/dim]")
            break

        log_path = LOG_DIR / f"{selected}_{datetime.now():%Y%m%d_%H%M%S}.log"
        action   = run_server(selected, cmd, port, log_path, model_cfg, defaults)
        console.print(f"\n[yellow]Server stopped.[/yellow]  Log: [dim]{log_path.name}[/dim]")

        if action == "quit":
            break
        selected = None

    console.print("[dim]Goodbye.[/dim]")


if __name__ == "__main__":
    main()
