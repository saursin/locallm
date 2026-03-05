#!/usr/bin/env python3
"""llama-sweep — Sweep inference parameters to find optimal tok/s config.

Uses llama-bench's native multi-value syntax to run all combinations in a
single process, then parses the markdown table output.

Usage:
  python llama-sweep.py [model]
  python llama-sweep.py 3 --threads 4,6,8,10 --batch 128,256,512
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rich.live import Live
from rich.panel import Panel
from rich.prompt import Confirm
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from llama_utils import REPO_ROOT, MODELS_YML, console, load_config, resolve_model, select_model_interactive

LLAMA_BENCH = REPO_ROOT / "llama.cpp/build/bin/llama-bench"

DEFAULT_THREADS = "4,6,8,10"
DEFAULT_BATCH   = "128,512"
DEFAULT_UBATCH  = "128,512"


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_bench(model_path: str, threads: str, batch: str, ubatch: str,
              n_prompt: int, n_gen: int, reps: int,
              cache_k: str, cache_v: str) -> str | None:
    """Run llama-bench with comma-separated param lists. Returns raw stdout or None."""
    cmd = [
        str(LLAMA_BENCH),
        "-m",   model_path,
        "-t",   threads,       # e.g. "4,6,8,10" — bench sweeps all values
        "-b",   batch,         # e.g. "128,512"
        "-ub",  ubatch,        # micro-batch size — affects pp throughput
        "-ngl", "0",           # CPU only
        "-p",   str(n_prompt),
        "-n",   str(n_gen),
        "-r",   str(reps),
        "-fa",  "1",           # match server config
        "-ctk", cache_k,
        "-ctv", cache_v,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return result.stdout if result.returncode == 0 else None
    except Exception as e:
        console.print(f"[red]llama-bench error:[/red] {e}")
        return None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_output(output: str) -> list[dict]:
    """Parse llama-bench markdown table into list of result dicts.
    Reads the header row to find column positions dynamically.
    Each row becomes: {threads, n_batch, test, tps, type ('pp'|'tg')}
    """
    col: dict[str, int] = {}
    rows = []

    for line in output.splitlines():
        if not line.startswith("|") or "---" in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        parts = [p for p in parts if p != ""]

        if not parts:
            continue

        # Header row — build column index map
        if parts[0].lower() == "model":
            col = {name.lower(): i for i, name in enumerate(parts)}
            continue

        if not col:
            continue

        try:
            threads  = int(parts[col["threads"]])
            n_batch  = int(parts[col["n_batch"]])
            n_ubatch = int(parts[col["n_ubatch"]]) if "n_ubatch" in col else n_batch
            test     = parts[col["test"]]
            tps_str  = parts[col["t/s"]]
        except (KeyError, ValueError, IndexError):
            continue

        if m := re.match(r"([\d.]+)", tps_str):
            rows.append({
                "threads":  threads,
                "n_batch":  n_batch,
                "n_ubatch": n_ubatch,
                "test":     test,
                "tps":      float(m.group(1)),
                "type":     "pp" if test.startswith("pp") else "tg",
            })
    return rows


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def build_table(rows: list[dict]) -> Table:
    # Group by (threads, n_batch, n_ubatch) — one row in our table per combo
    combos: dict[tuple, dict] = {}
    for r in rows:
        key = (r["threads"], r["n_batch"], r["n_ubatch"])
        combos.setdefault(key, {})
        combos[key][r["type"]] = r["tps"]

    best_tg = max((v.get("tg", 0) for v in combos.values()), default=0.0)

    t = Table(show_header=True, header_style="bold cyan", border_style="dim")
    t.add_column("threads", justify="right")
    t.add_column("batch",   justify="right")
    t.add_column("ubatch",  justify="right")
    t.add_column("pp tok/s", justify="right", style="cyan")
    t.add_column("tg tok/s", justify="right", style="green")
    t.add_column("",         width=2)

    for (threads, n_batch, n_ubatch), vals in sorted(combos.items()):
        tg = vals.get("tg")
        pp = vals.get("pp")
        is_best  = tg is not None and tg == best_tg and best_tg > 0
        star     = "[bold yellow]★[/bold yellow]" if is_best else ""
        tg_color = "bold green" if is_best else "green"
        tg_str   = f"[{tg_color}]{tg:.2f}[/{tg_color}]" if tg is not None else "—"
        pp_str   = f"{pp:.2f}" if pp is not None else "—"
        t.add_row(str(threads), str(n_batch), str(n_ubatch), pp_str, tg_str, star)

    return t


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_to_yml(model_name: str, threads: int, batch_size: int, ubatch_size: int) -> None:
    """Update models.yml in-place using text patching to preserve comments."""
    with open(MODELS_YML) as f:
        lines = f.readlines()

    to_set = {
        "threads":     str(threads),
        "batch_size":  str(batch_size),
        "ubatch_size": str(ubatch_size),
    }

    # Find model block: line matching "  <model_name>:"
    model_re = re.compile(r"^  " + re.escape(model_name) + r"\s*:")
    start = next((i for i, l in enumerate(lines) if model_re.match(l)), None)
    if start is None:
        console.print(f"[red]Could not find model {model_name!r} in models.yml[/red]")
        return

    # Find end of block: next line at ≤2-space indent (another model or section)
    end = len(lines)
    for i in range(start + 1, len(lines)):
        stripped = lines[i].lstrip()
        if stripped and not stripped.startswith("#") and (len(lines[i]) - len(stripped)) <= 2:
            end = i
            break

    block = lines[start:end]
    updated: set[str] = set()

    # Overwrite existing keys
    for i, line in enumerate(block):
        for key, val in to_set.items():
            if re.match(rf"^\s+{re.escape(key)}\s*:", line):
                indent = len(line) - len(line.lstrip())
                block[i] = f"{' ' * indent}{key}: {val}\n"
                updated.add(key)

    # Insert missing keys after the `path:` line (or after model name line)
    insert_at = 1
    for i, line in enumerate(block):
        if re.match(r"^\s+path\s*:", line):
            insert_at = i + 1
            break

    for key in ("threads", "batch_size", "ubatch_size"):
        if key not in updated:
            block.insert(insert_at, f"    {key}: {to_set[key]}\n")
            insert_at += 1

    lines[start:end] = block
    with open(MODELS_YML, "w") as f:
        f.writelines(lines)

    console.print(
        f"[green]Saved:[/green] threads={threads}, batch_size={batch_size}, ubatch_size={ubatch_size}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(prog="llama-sweep", description="Sweep params to maximise tok/s.")
    p.add_argument("model",      nargs="?", help="Model name or index")
    p.add_argument("--threads",  default=DEFAULT_THREADS,
                   help=f"Comma-separated thread counts  (default: {DEFAULT_THREADS})")
    p.add_argument("--batch",    default=DEFAULT_BATCH,
                   help=f"Comma-separated batch sizes  (default: {DEFAULT_BATCH})")
    p.add_argument("--ubatch",   default=DEFAULT_UBATCH,
                   help=f"Comma-separated ubatch sizes  (default: {DEFAULT_UBATCH})")
    p.add_argument("--prompt",   type=int, default=512,  help="Prompt tokens  (default: 512)")
    p.add_argument("--gen",      type=int, default=128,  help="Generation tokens  (default: 128)")
    p.add_argument("--reps",     type=int, default=3,    help="Repetitions per combo  (default: 3)")
    args = p.parse_args()

    if not LLAMA_BENCH.exists():
        console.print(f"[red]llama-bench not found:[/red] {LLAMA_BENCH}")
        console.print("[yellow]Run 'make' from the repo root first.[/yellow]")
        sys.exit(1)

    defaults, models = load_config()

    if args.model:
        selected = resolve_model(args.model, list(models.keys()), models)
        if not selected:
            console.print(f"[red]Unknown model:[/red] {args.model!r}")
            sys.exit(1)
    else:
        try:
            selected = select_model_interactive(models, defaults)
        except KeyboardInterrupt:
            sys.exit(0)
        if not selected:
            sys.exit(0)

    model_cfg  = models[selected]
    model_path = model_cfg.get("path", "")
    if not model_path or not Path(model_path).exists():
        console.print(f"[red]Model file not found:[/red] {model_path}")
        sys.exit(1)

    cache_k = str(model_cfg.get("cache_k", defaults.get("cache_k", "f16")))
    cache_v = str(model_cfg.get("cache_v", defaults.get("cache_v", "f16")))

    n_threads = len(args.threads.split(","))
    n_batch   = len(args.batch.split(","))
    n_ubatch  = len(args.ubatch.split(","))
    n_combos  = n_threads * n_batch * n_ubatch * 2   # ×2 for pp + tg

    console.print(Panel.fit(
        Text("llama-sweep", style="bold cyan")
        + Text(f"  ·  {selected}  ·  {n_threads}×{n_batch}×{n_ubatch} combos", style="dim"),
        border_style="cyan",
    ))
    console.print(
        f"[dim]threads: {args.threads}   batch: {args.batch}   ubatch: {args.ubatch}\n"
        f"pp={args.prompt} tokens  tg={args.gen} tokens  reps={args.reps}  "
        f"kv={cache_k}/{cache_v}[/dim]\n"
    )

    # Run — show spinner while llama-bench works
    output = None
    try:
        with Live(
            Panel(Spinner("dots", text=" Running llama-bench…"), border_style="dim"),
            console=console, refresh_per_second=8, screen=False,
        ):
            output = run_bench(
                model_path, args.threads, args.batch, args.ubatch,
                args.prompt, args.gen, args.reps, cache_k, cache_v,
            )
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted.[/yellow]")
        sys.exit(0)

    if not output:
        console.print("[red]llama-bench produced no output. Check the model path and flags.[/red]")
        sys.exit(1)

    rows = parse_output(output)
    if not rows:
        console.print("[red]Could not parse llama-bench output.[/red]")
        console.print("[dim]Raw output:[/dim]")
        console.print(output)
        sys.exit(1)

    console.print(build_table(rows))

    # Find best tg config
    tg_rows = [r for r in rows if r["type"] == "tg"]
    if not tg_rows:
        console.print("[red]No tg results found.[/red]")
        sys.exit(1)

    best = max(tg_rows, key=lambda r: r["tps"])
    console.print(
        f"\n[bold green]Best for generation:[/bold green]  "
        f"threads={best['threads']}  batch_size={best['n_batch']}  ubatch_size={best['n_ubatch']}  →  "
        f"[bold]{best['tps']:.2f} tok/s[/bold]"
    )

    try:
        if Confirm.ask("\nSave to models.yml?", default=True):
            save_to_yml(selected, best["threads"], best["n_batch"], best["n_ubatch"])
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
