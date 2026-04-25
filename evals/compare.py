"""Diff two benchmark runs to spot regressions.

Usage:
    python -m evals.compare <baseline_run> <candidate_run>
    python -m evals.compare evals/runs/20260416-...-abc123 evals/runs/20260416-...-def456

Can also pass just run names (the dir under evals/runs/).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = _ROOT / "evals" / "runs"


def _resolve(run_arg: str) -> Path:
    p = Path(run_arg)
    if p.exists():
        return p
    p = RUNS_DIR / run_arg
    if p.exists():
        return p
    print(f"ERROR: run not found: {run_arg}")
    sys.exit(1)


def _load_results(run_dir: Path) -> dict[str, dict]:
    results_file = run_dir / "results.json"
    if not results_file.exists():
        print(f"ERROR: {results_file} does not exist")
        sys.exit(1)
    results = json.loads(results_file.read_text())
    return {r["id"]: r for r in results}


def _delta(a: float, b: float) -> str:
    """Format a -> b with sign and colored arrow (no colors, just text)."""
    diff = b - a
    arrow = "=" if abs(diff) < 1e-9 else ("▲" if diff > 0 else "▼")
    return f"{a} {arrow} {b} ({diff:+.3f})"


# Metrics where higher is better; regression means candidate < baseline
_HIGHER_IS_BETTER = {"citation_coverage.coverage_ratio", "citation_density.per_100_words"}
# Metrics where lower is better; regression means candidate > baseline
_LOWER_IS_BETTER = {
    "invalid_citations.count",
    "orphan_sources.ratio",
    "snippet_ratio.ratio",
    "banned_phrases.total",
}


def _get_path(d: dict, path: str):
    for part in path.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def _is_regression(metric: str, baseline: float, candidate: float, tol: float = 0.01) -> bool:
    if baseline is None or candidate is None:
        return False
    if metric in _HIGHER_IS_BETTER:
        return candidate < baseline - tol
    if metric in _LOWER_IS_BETTER:
        return candidate > baseline + tol
    return False


def _compare_one(bench_id: str, a: dict, b: dict) -> list[str]:
    lines = [f"\n=== {bench_id} ==="]
    regressions = 0

    if a.get("error") or b.get("error"):
        lines.append(f"  baseline: {a.get('error') or 'ok'}")
        lines.append(f"  candidate: {b.get('error') or 'ok'}")
        if b.get("error") and not a.get("error"):
            lines.append("  [REGRESSION] new failure in candidate")
            regressions = 1
        lines.append(f"__regressions__={regressions}")
        return lines

    # Timings + source count
    lines.append(
        f"  elapsed:      {a['elapsed_seconds']}s -> {b['elapsed_seconds']}s"
        f"   sources: {a['source_count']} -> {b['source_count']}"
    )

    a_checks, b_checks = a.get("checks", {}), b.get("checks", {})

    metrics = [
        ("citation_density.per_100_words", "cite_density/100w"),
        ("citation_coverage.coverage_ratio", "cite_coverage    "),
        ("invalid_citations.count", "invalid_refs     "),
        ("orphan_sources.ratio", "orphan_ratio     "),
        ("snippet_ratio.ratio", "snippet_ratio    "),
        ("banned_phrases.total", "banned_phrases   "),
    ]
    for path, label in metrics:
        av = _get_path(a_checks, path)
        bv = _get_path(b_checks, path)
        if av is None or bv is None:
            continue
        regression_marker = ""
        if _is_regression(path, av, bv):
            regression_marker = "  [REGRESSION]"
            regressions += 1
        lines.append(f"  {label}: {_delta(av, bv)}{regression_marker}")

    # Structure pass/fail
    a_struct = _get_path(a_checks, "structure.pass")
    b_struct = _get_path(b_checks, "structure.pass")
    if a_struct is not None and b_struct is not None:
        lines.append(f"  structure_pass   : {a_struct} -> {b_struct}")
        if a_struct and not b_struct:
            missing = _get_path(b_checks, "structure.missing") or []
            lines.append(f"  [REGRESSION] structure broke — missing: {missing}")
            regressions += 1

    lines.append(f"__regressions__={regressions}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Diff two benchmark runs")
    parser.add_argument("baseline", help="Baseline run dir or name")
    parser.add_argument("candidate", help="Candidate run dir or name")
    args = parser.parse_args()

    a_dir = _resolve(args.baseline)
    b_dir = _resolve(args.candidate)
    a = _load_results(a_dir)
    b = _load_results(b_dir)

    print(f"Baseline:  {a_dir.name}")
    print(f"Candidate: {b_dir.name}")

    shared = sorted(set(a) & set(b))
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))

    total_regressions = 0
    for bid in shared:
        lines = _compare_one(bid, a[bid], b[bid])
        reg_line = lines[-1]
        if reg_line.startswith("__regressions__="):
            total_regressions += int(reg_line.split("=", 1)[1])
            lines = lines[:-1]
        print("\n".join(lines))

    if only_a:
        print(f"\nOnly in baseline: {', '.join(only_a)}")
    if only_b:
        print(f"\nOnly in candidate: {', '.join(only_b)}")

    print("\n" + "=" * 60)
    if total_regressions:
        print(f"REGRESSIONS DETECTED: {total_regressions} across {len(shared)} benchmarks")
        sys.exit(1)
    else:
        print(f"No regressions across {len(shared)} benchmarks")


if __name__ == "__main__":
    main()
