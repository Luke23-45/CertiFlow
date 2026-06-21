"""Batch benchmark evaluator for CertiQIndexModel across all QGym environments.

Produces comparison tables matching the format in the QGym paper (arXiv:2410.06170).

Usage
-----
    # Single checkpoint evaluated on all envs (generalization test)
    python -m certiq_net.studies.run_benchmark ^
        --checkpoint path/to/model.pt ^
        --output-dir benchmark_results

    # Per-environment checkpoints
    python -m certiq_net.studies.run_benchmark ^
        --checkpoint-dir checkpoints/ ^
        --output-dir benchmark_results

    # Generate LaTeX table from existing results (no re-evaluation)
    python -m certiq_net.studies.run_benchmark ^
        --results-dir benchmark_results/results ^
        --output-dir benchmark_results ^
        --table-only
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from certiq_net.studies.qgym_eval._qgym_paths import (
    certiq_model_config_path,
    resolve_qgym_root,
)

# ---------------------------------------------------------------------------
# Benchmark environment definitions
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkEnv:
    """A single benchmark environment with metadata for table display."""

    env_name: str
    display_name: str
    group: str
    service_type: str = "exponential"
    lam_type: str = "constant"
    has_env_data: bool = True


BENCHMARK_ENVS: list[BenchmarkEnv] = [
    BenchmarkEnv("criss_cross_bh", "Criss Cross BH", "Criss Cross"),
    # Reentrant-1 [Exponential]
    *[BenchmarkEnv(f"reentrant_{level}", f"Reentrant-1 (L={level})",
                    "Reentrant-1 [Exponential]") for level in range(2, 11)],
    # Reentrant-2 [Exponential]
    *[BenchmarkEnv(f"re-reentrant_{level}", f"Reentrant-2 (L={level})",
                    "Reentrant-2 [Exponential]") for level in range(2, 11)],
    # Reentrant-1 [Hyperexponential]
    *[BenchmarkEnv(f"reentrant_{level}_hyper", f"Reentrant-1 (L={level})",
                    "Reentrant-1 [Hyperexponential]",
                    service_type="hyper", lam_type="hyper") for level in range(2, 8)],
    # Reentrant-2 [Hyperexponential]
    *[BenchmarkEnv(f"re-reentrant_{level}_hyper", f"Reentrant-2 (L={level})",
                    "Reentrant-2 [Hyperexponential]",
                    service_type="hyper", lam_type="hyper") for level in range(2, 8)],
    # Parallel server
    BenchmarkEnv("n_model_5x5", "N Model (5x5)", "Parallel Server"),
    BenchmarkEnv("n_model", "N Model (basic)", "Parallel Server"),
    # Real-world
    BenchmarkEnv("input_switch", "Input Switch", "Real World"),
    BenchmarkEnv("hospital", "Hospital", "Real World"),
]

# Table groups for layout
TABLE_GROUPS: list[tuple[str, list[str]]] = [
    ("Criss Cross", ["Criss Cross BH"]),
    ("Reentrant-1 [Exponential]", [f"Reentrant-1 (L={level})" for level in range(2, 11)]),
    ("Reentrant-2 [Exponential]", [f"Reentrant-2 (L={level})" for level in range(2, 11)]),
    ("Reentrant-1 [Hyperexponential]", [f"Reentrant-1 (L={level})" for level in range(2, 8)]),
    ("Reentrant-2 [Hyperexponential]", [f"Reentrant-2 (L={level})" for level in range(2, 8)]),
    ("Parallel Server", ["N Model (5x5)"]),
    ("Real World", ["Input Switch", "Hospital"]),
]

BASELINE_COLUMNS = ["c-mu", "MW", "MP", "FP", "PPO", "PPO BC", "PPO WC"]
ALL_COLUMNS = list(BASELINE_COLUMNS) + ["CertiQ (ours)"]


def _resolve_checkpoint(
    env_name: str,
    checkpoint: Path | None,
    checkpoint_dir: Path | None,
) -> Path | None:
    """Resolve the checkpoint path for *env_name*.

    Priority:
      1. ``checkpoint`` (single checkpoint used for all envs)
      2. ``checkpoint_dir / {env_name}.pt``
      3. ``checkpoint_dir / {env_name}_final_model_state.pt``
    """
    if checkpoint is not None:
        return checkpoint if checkpoint.exists() else None

    if checkpoint_dir is not None:
        for candidate in (
            checkpoint_dir / f"{env_name}.pt",
            checkpoint_dir / f"{env_name}_final_model_state.pt",
            checkpoint_dir / f"{env_name}_checkpoint.pt",
        ):
            if candidate.exists():
                return candidate
    return None


# ---------------------------------------------------------------------------
# Single-environment evaluation
# ---------------------------------------------------------------------------


def evaluate_single(
    env_name: str,
    checkpoint_path: Path,
    *,
    qgym_root: Path,
    device: str = "cpu",
    test_batch: int = 100,
    test_steps: int | None = None,
    seed: int = 42,
    model_config_path: Path | None = None,
    timeout: float = 600.0,  # noqa: ARG001
) -> dict:
    """Evaluate one environment using the RL/PPO pipeline."""
    from certiq_net.studies.qgym_eval.evaluate import evaluate_checkpoint

    return evaluate_checkpoint(
        env_name=env_name,
        checkpoint_path=checkpoint_path,
        qgym_root=qgym_root,
        device=device,
        test_batch=test_batch,
        test_steps=test_steps,
        seed=seed,
        model_config_path=model_config_path,
    )


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------


def save_result(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    env_name = result.get("env_name", "unknown")
    path = output_dir / f"{env_name}_results.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


def load_result(env_name: str, results_dir: Path) -> dict | None:
    path = results_dir / f"{env_name}_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_all_results(results_dir: Path, envs: list[BenchmarkEnv]) -> dict[str, dict | None]:
    return {be.env_name: load_result(be.env_name, results_dir) for be in envs}


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------


def _fmt_loss(val: float | None, std: float | None = None) -> str:
    """Format a loss value for LaTeX display."""
    if val is None:
        return "--"
    if abs(val) >= 1e4:
        exp = int(np.floor(np.log10(abs(val))))
        mantissa = val / (10**exp)
        if std is not None and std > 0:
            return f"${mantissa:.1f}\\text{{E+}}{exp} \\pm {std:.1f}$"
        return f"${mantissa:.1f}\\text{{E+}}{exp}$"
    if std is not None and std > 0:
        return f"${val:.2f} \\pm {std:.2f}$"
    return f"${val:.2f}$"


def _bold_best(results: list[dict | None], metric_key: str = "test_loss") -> list[str]:
    """Identify and bold the best (lowest) value in a row."""
    valid = [(i, r[metric_key]) for i, r in enumerate(results)
             if r is not None and r.get(metric_key) is not None]
    if not valid:
        return ["--"] * len(results)
    best_idx = min(valid, key=lambda x: x[1])[0]
    out: list[str] = []
    for i, r in enumerate(results):
        if r is not None and r.get(metric_key) is not None:
            val = r[metric_key]
            std = r.get("test_loss_std")
            formatted = _fmt_loss(val, std)
            if i == best_idx:
                formatted = r"\mathbf{" + formatted.strip("$") + "}"
            out.append(formatted)
        else:
            out.append("--")
    return out


def generate_latex_table(
    all_results: dict[str, dict | None],
    *,
    caption: str = "Benchmark results.",
    label: str = "tab:benchmark",
    metric_key: str = "test_loss",
) -> str:
    """Generate a LaTeX table."""
    display_results: OrderedDict[str, dict | None] = OrderedDict()
    for be in BENCHMARK_ENVS:
        display_results[be.display_name] = all_results.get(be.env_name)

    n_cols = len(ALL_COLUMNS)

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\sisetup{round-mode=places,round-precision=2}")
    lines.append(r"\begin{tabular}{l" + "c" * n_cols + "}")
    lines.append(r"\toprule")
    lines.append("Network & " + " & ".join(ALL_COLUMNS) + r" \\")
    lines.append(r"\midrule")

    for group_name, group_displays in TABLE_GROUPS:
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{" + str(n_cols + 1) + r"}{l}{\textbf{" + group_name + r"}} \\")
        lines.append(r"\midrule")

        for disp in group_displays:
            row: list[str] = [disp]
            row.extend(["--"] * (len(ALL_COLUMNS) - 1))

            res = display_results.get(disp)
            if res is not None and res.get(metric_key) is not None:
                val = res[metric_key]
                std = res.get("test_loss_std")
                row[-1] = _fmt_loss(val, std)

            lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_markdown_table(
    all_results: dict[str, dict | None],
    *,
    metric_key: str = "test_loss",
) -> str:
    """Generate a Markdown table for terminal/README."""
    display_results: OrderedDict[str, dict | None] = OrderedDict()
    for be in BENCHMARK_ENVS:
        display_results[be.display_name] = all_results.get(be.env_name)

    n_cols = len(ALL_COLUMNS)

    lines: list[str] = []
    lines.append("| Network | " + " | ".join(ALL_COLUMNS) + " |")
    lines.append("|" + "---|" * (n_cols + 1))

    for group_name, group_displays in TABLE_GROUPS:
        sep = " | ".join(["---"] * n_cols)
        lines.append(f"**{group_name}** | {sep} |")
        for disp in group_displays:
            row: list[str] = [disp]
            row.extend(["--"] * (len(ALL_COLUMNS) - 1))

            res = display_results.get(disp)
            if res is not None and res.get(metric_key) is not None:
                val = res[metric_key]
                std = res.get("test_loss_std")
                row[-1] = f"{val:.2f} \u00b1 {std:.2f}" if std else f"{val:.2f}"

            lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch benchmark evaluator for CertiQIndexModel"
    )
    ckpt_group = parser.add_mutually_exclusive_group()
    ckpt_group.add_argument("--checkpoint", type=str, default=None,
                            help="Single checkpoint used for all envs")
    ckpt_group.add_argument("--checkpoint-dir", type=str, default=None,
                            help="Directory with per-env checkpoints (e.g. reentrant_2.pt)")

    parser.add_argument("--envs", type=str, nargs="*", default=None,
                        help="Specific envs to benchmark (default: all)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device (cpu, cuda, mps)")
    parser.add_argument("--test-batch", type=int, default=100,
                        help="Parallel environments per evaluation")
    parser.add_argument("--test-steps", type=int, default=None,
                        help="Override simulation horizon")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    parser.add_argument("--qgym-root", type=str, default=None,
                        help="Path to QGym (default: <project_root>/extern/QGym)")
    parser.add_argument("--model-config", type=str, default=None,
                        help="Path to model config YAML")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Output directory for results and tables")

    parser.add_argument("--table-only", action="store_true",
                        help="Skip evaluation; regenerate tables from existing results")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory with per-env _results.json files (for --table-only)")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="Per-environment timeout in seconds")

    parser.add_argument("--format", type=str, default="latex",
                        choices=["latex", "markdown", "both"],
                        help="Table output format")
    parser.add_argument("--resume", action="store_true",
                        help="Skip envs that already have result files")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.qgym_root:
        qgym_root = Path(args.qgym_root).resolve()
    else:
        qgym_root = resolve_qgym_root()

    if not (qgym_root / "RL" / "PPO" / "trainer.py").exists():
        print(f"QGym root not found or invalid: {qgym_root}", file=sys.stderr)
        sys.exit(1)

    if args.model_config:
        model_config_path = Path(args.model_config).resolve()
    else:
        model_config_path = certiq_model_config_path()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir) if args.results_dir else output_dir / "results"

    checkpoint: Path | None = Path(args.checkpoint).resolve() if args.checkpoint else None
    checkpoint_dir: Path | None = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else None

    if args.envs is not None:
        env_names = set(args.envs)
        envs = [be for be in BENCHMARK_ENVS if be.env_name in env_names]
        missing = env_names - {be.env_name for be in envs}
        if missing:
            print(f"Warning: unknown envs: {missing}", file=sys.stderr)
    else:
        envs = list(BENCHMARK_ENVS)

    if not args.table_only:
        if checkpoint is None and checkpoint_dir is None:
            print("Error: specify --checkpoint or --checkpoint-dir (or --table-only)", file=sys.stderr)
            sys.exit(1)

        results_dir.mkdir(parents=True, exist_ok=True)
        all_results: dict[str, dict | None] = {}

        from tqdm import tqdm

        for be in (pbar := tqdm(envs, desc="Benchmarking")):
            pbar.set_description(f"Benchmarking {be.env_name}")

            if args.resume and load_result(be.env_name, results_dir) is not None:
                tqdm.write(f"  [skip] {be.env_name} (already evaluated)")
                all_results[be.env_name] = load_result(be.env_name, results_dir)
                continue

            ckpt_path = _resolve_checkpoint(be.env_name, checkpoint, checkpoint_dir)
            if ckpt_path is None:
                tqdm.write(f"  [skip] No checkpoint for {be.env_name}")
                all_results[be.env_name] = None
                continue

            try:
                result = evaluate_single(
                    be.env_name, ckpt_path,
                    qgym_root=qgym_root,
                    device=args.device,
                    test_batch=args.test_batch,
                    test_steps=args.test_steps,
                    seed=args.seed,
                    model_config_path=model_config_path,
                    timeout=args.timeout,
                )
                save_result(result, results_dir)
                all_results[be.env_name] = result
                tqdm.write(
                    f"  {be.env_name}: test_loss={result.get('test_loss', 'N/A'):.4f} "
                    f"\u00b1 {result.get('test_loss_std', 'N/A')}"
                )
            except Exception as exc:
                tqdm.write(f"  [error] {be.env_name}: {exc}")
                all_results[be.env_name] = None
    else:
        print(f"Loading results from {results_dir}", file=sys.stderr)
        all_results = {}
        for be in envs:
            all_results[be.env_name] = load_result(be.env_name, results_dir)

    if args.format in ("latex", "both"):
        latex = generate_latex_table(all_results,
                                     caption="CertiQ benchmark results.",
                                     label="tab:benchmark")
        latex_path = output_dir / "benchmark_table.tex"
        latex_path.write_text(latex)
        print(f"LaTeX table: {latex_path}", file=sys.stderr)

    if args.format in ("markdown", "both"):
        md = generate_markdown_table(all_results)
        md_path = output_dir / "benchmark_table.md"
        md_path.write_text(md)
        print(f"Markdown table: {md_path}", file=sys.stderr)

        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)
        print(md)

    total = len(envs)
    completed = sum(1 for r in all_results.values() if r is not None)
    print(f"\n{completed}/{total} environments evaluated successfully.", file=sys.stderr)
