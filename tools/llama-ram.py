#!/usr/bin/env python3
"""llama-ram — Estimate RAM usage before loading a model."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from gguf import GGUFReader
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / "llama.cpp" / "gguf-py"))
    from gguf import GGUFReader

from rich.panel import Panel
from rich.table import Table

from llama_utils import REPO_ROOT, console, load_config, resolve_model, select_model_interactive, models_table

# Bytes per element for KV cache quant types
# Formula: (bits_per_element * block_size + scale_bytes * 8) / block_size / 8
QUANT_BYTES: dict[str, float] = {
    "f32":  4.000,
    "f16":  2.000,
    "bf16": 2.000,
    "q8_0": 1.0625,   # (8*32 + 16) / 32 / 8 * 8  → 34 bytes / 32 elements
    "q4_0": 0.5625,   # (4*32 + 16) / 32 / 8 * 8  → 18 bytes / 32 elements
    "q4_1": 0.6250,   # (4*32 + 32) / 32 / 8 * 8  → 20 bytes / 32 elements
    "q5_0": 0.6875,   # (5*32 + 16) / 32 / 8 * 8  → 22 bytes / 32 elements
    "q5_1": 0.7500,   # (5*32 + 32) / 32 / 8 * 8  → 24 bytes / 32 elements
}


def quant_bytes(name: str) -> float:
    return QUANT_BYTES.get(name.lower().replace("-", "_"), 2.0)


def read_gguf_meta(path: str) -> dict:
    reader = GGUFReader(path, "r")
    meta: dict = {}
    for field in reader.fields.values():
        try:
            if field.types and field.types[0].name == "STRING":
                meta[field.name] = bytes(field.parts[-1]).decode("utf-8")
            else:
                val = field.parts[-1].tolist()
                meta[field.name] = val[0] if isinstance(val, list) and len(val) == 1 else val
        except Exception:
            pass
    return meta


def estimate(model_path: str, ctx_size: int, cache_k: str, cache_v: str, n_gpu_layers: int = 0) -> dict:
    file_size = Path(model_path).stat().st_size
    meta      = read_gguf_meta(model_path)
    arch      = meta.get("general.architecture", "llama")

    n_layers   = int(meta.get(f"{arch}.block_count",          32))
    n_embd     = int(meta.get(f"{arch}.embedding_length",    4096))
    n_head     = int(meta.get(f"{arch}.attention.head_count",  32))
    n_head_kv  = int(meta.get(f"{arch}.attention.head_count_kv", n_head))
    head_dim_k = int(meta.get(f"{arch}.attention.key_length",   n_embd // max(n_head, 1)))
    head_dim_v = int(meta.get(f"{arch}.attention.value_length", n_embd // max(n_head, 1)))

    kv_k = ctx_size * n_layers * n_head_kv * head_dim_k * quant_bytes(cache_k)
    kv_v = ctx_size * n_layers * n_head_kv * head_dim_v * quant_bytes(cache_v)

    # Rough compute buffer: scales with batch × embd; llama.cpp typically uses ~100-300 MiB
    compute = max(100 * 1024**2, n_embd * 512 * 4)

    cpu_frac       = max(0.0, (n_layers - n_gpu_layers) / n_layers) if n_layers else 1.0
    cpu_model_size = file_size * cpu_frac

    return {
        "meta":           meta,
        "arch":           arch,
        "n_layers":       n_layers,
        "n_head_kv":      n_head_kv,
        "head_dim_k":     head_dim_k,
        "head_dim_v":     head_dim_v,
        "file_size":      file_size,
        "cpu_model_size": cpu_model_size,
        "kv_k":           kv_k,
        "kv_v":           kv_v,
        "kv_total":       kv_k + kv_v,
        "compute":        compute,
        "total":          cpu_model_size + kv_k + kv_v + compute,
    }


def available_ram() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def fmt(n: float) -> str:
    return f"{n / 1024**3:.2f} GiB" if n >= 1024**3 else f"{n / 1024**2:.0f} MiB"


def show(model_name: str, model_cfg: dict, defaults: dict) -> None:
    model_path = model_cfg.get("path", "")
    if not model_path or not Path(model_path).exists():
        console.print(f"[red]File not found:[/red] {model_path}")
        return

    ctx    = int(model_cfg.get("ctx_size",    defaults.get("ctx_size",    4096)))
    kv_k   = str(model_cfg.get("cache_k",     defaults.get("cache_k",     "f16")))
    kv_v   = str(model_cfg.get("cache_v",     defaults.get("cache_v",     "f16")))
    n_gpu  = int(model_cfg.get("n_gpu_layers", defaults.get("n_gpu_layers", 0)) or 0)
    avail  = available_ram()

    try:
        e = estimate(model_path, ctx, kv_k, kv_v, n_gpu)
    except Exception as ex:
        console.print(f"[red]Could not read GGUF metadata:[/red] {ex}")
        return

    total = e["total"]
    if avail:
        ratio = total / avail
        color = "green" if ratio < 0.70 else "yellow" if ratio < 0.90 else "red"
    else:
        color = "white"

    t = Table(show_header=True, header_style="bold cyan", border_style="dim")
    t.add_column("Component",   style="dim")
    t.add_column("Size",        justify="right")
    t.add_column("Detail",      style="dim")

    cpu_note = f"mmap'd ({n_gpu} layers on GPU)" if n_gpu else "mmap'd"
    t.add_row("Model weights",          fmt(e["cpu_model_size"]), cpu_note)
    t.add_row(f"KV cache K  ({kv_k})",  fmt(e["kv_k"]),
              f"ctx {ctx}  ×  {e['n_layers']} layers  ×  {e['n_head_kv']} heads  ×  dim {e['head_dim_k']}")
    t.add_row(f"KV cache V  ({kv_v})",  fmt(e["kv_v"]), "")
    t.add_row("Compute buffer",         fmt(e["compute"]), "estimate")
    t.add_section()
    t.add_row(
        f"[bold {color}]Total[/bold {color}]",
        f"[bold {color}]{fmt(total)}[/bold {color}]",
        f"[{color}]{fmt(avail)} available[/{color}]" if avail else "",
    )

    console.print(Panel(t, title=f"[bold]RAM estimate — {model_name}[/bold]", border_style="cyan"))

    if avail and total > avail * 0.90:
        console.print(
            f"[bold red]Warning:[/bold red] estimated usage is {ratio*100:.0f}% of available RAM.\n"
            f"[dim]Consider lowering ctx_size or using a smaller quant.[/dim]"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(prog="llama-ram", description="Estimate RAM usage for GGUF models.")
    p.add_argument("model", nargs="?", help="Model name or index (omit to show all)")
    p.add_argument("--ctx", type=int, help="Override context size for the estimate")
    args = p.parse_args()

    console.print(Panel.fit("[bold cyan]llama-ram[/bold cyan]  ·  RAM estimator", border_style="cyan"))

    defaults, models = load_config()
    if args.ctx:
        defaults = {**defaults, "ctx_size": args.ctx}

    if args.model:
        selected = resolve_model(args.model, list(models.keys()), models)
        if not selected:
            console.print(f"[red]Unknown model:[/red] {args.model!r}")
            sys.exit(1)
        show(selected, models[selected], defaults)
    else:
        # Show all models
        for name, cfg in models.items():
            show(name, cfg, defaults)
            console.print()


if __name__ == "__main__":
    main()
