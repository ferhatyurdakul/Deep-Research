"""Benchmark runner for Deep Research evals.

Runs a subset of benchmarks, saves reports + mechanical checks to
evals/runs/<timestamp>/, and prints a summary.

Usage:
    python -m evals.run --tag smoke            # fast subset
    python -m evals.run --tag full             # everything
    python -m evals.run --id wasm-vs-js        # one benchmark
    python -m evals.run --tag smoke --label pre-tone-fix
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Make src/ importable without installing
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from deep_research.config import settings
from deep_research.logging_config import setup_logging
from deep_research.models import ReportStyle, ResearchConfig, ResearchDepth
from deep_research.pipeline.agent import arun_agent_research
from deep_research.pipeline.orchestrator import arun_research

from evals.benchmarks import select
from evals.checks import run_all, summary_line

RUNS_DIR = _ROOT / "evals" / "runs"


def _git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], cwd=_ROOT, stderr=subprocess.DEVNULL
        )
        return f"{sha}{'-dirty' if dirty else ''}"
    except Exception:
        return "unknown"


def _build_config(spec: dict) -> ResearchConfig:
    config = ResearchConfig.from_depth(ResearchDepth(spec["depth"]))
    config.report_style = ReportStyle(spec.get("style", "standard"))
    if spec.get("template"):
        config.template = spec["template"]
    config.fresh = True  # evals always start clean, never inherit skip list
    config.load_model_routes()
    return config


async def _run_one(spec: dict, out_dir: Path) -> dict:
    config = _build_config(spec)
    bench_id = spec["id"]
    bench_dir = out_dir / bench_id
    bench_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    error = None
    report = None
    try:
        if spec.get("agent"):
            report = await arun_agent_research(spec["query"], config)
        else:
            report = await arun_research(spec["query"], config)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    elapsed = round(time.time() - started, 2)

    result: dict = {
        "id": bench_id,
        "query": spec["query"],
        "depth": spec["depth"],
        "template": spec.get("template"),
        "style": spec.get("style", "standard"),
        "agent": spec.get("agent", False),
        "elapsed_seconds": elapsed,
        "error": error,
    }

    if report is not None:
        (bench_dir / "report.md").write_text(
            f"# {report.query}\n\n{report.executive_summary}\n\n{report.detailed_findings}"
        )
        (bench_dir / "report.json").write_text(report.model_dump_json(indent=2))
        checks = run_all(report, config.report_style)
        (bench_dir / "checks.json").write_text(json.dumps(checks, indent=2))
        result["source_count"] = len(report.sources)
        result["iteration_count"] = len(report.iterations)
        result["checks"] = checks
        result["summary"] = summary_line(checks)
    else:
        (bench_dir / "error.txt").write_text(error or "unknown error")

    return result


async def _run_all(specs: list[dict], out_dir: Path) -> list[dict]:
    results = []
    for i, spec in enumerate(specs, 1):
        print(f"\n[{i}/{len(specs)}] Running {spec['id']}: {spec['query'][:70]}...")
        res = await _run_one(spec, out_dir)
        if res.get("error"):
            print(f"  [FAIL] {res['error']}")
        else:
            print(f"  [OK] {res['elapsed_seconds']}s — {res['summary']}")
        results.append(res)
    return results


def _print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for r in results:
        if r.get("error"):
            print(f"  {r['id']:30s}  FAIL  {r['error']}")
        else:
            print(f"  {r['id']:30s}  {r['summary']}")

    failed = [r for r in results if r.get("error")]
    if failed:
        print(f"\n{len(failed)}/{len(results)} benchmarks failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Deep Research evaluation benchmarks")
    parser.add_argument("--tag", default="smoke", help="Run benchmarks matching this tag (default: smoke)")
    parser.add_argument("--id", nargs="*", help="Run specific benchmarks by id")
    parser.add_argument("--label", default="", help="Optional label appended to run directory")
    parser.add_argument("--list", action="store_true", help="List matching benchmarks and exit")
    args = parser.parse_args()

    setup_logging()

    if not settings.active_api_key:
        print(f"ERROR: {settings.active_api_key_env_name} not set (LLM_PROVIDER={settings.active_provider})")
        sys.exit(1)

    specs = select(tag=None if args.id else args.tag, ids=args.id)
    if not specs:
        print(f"No benchmarks matched tag={args.tag!r} ids={args.id!r}")
        sys.exit(1)

    if args.list:
        for s in specs:
            print(f"  {s['id']:30s} depth={s['depth']:8s} template={s.get('template') or '-':12s} style={s.get('style', 'standard')}")
        return

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    sha = _git_sha()
    run_name = f"{timestamp}-{sha}"
    if args.label:
        run_name += f"-{args.label}"
    out_dir = RUNS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "timestamp": timestamp,
        "git_sha": sha,
        "label": args.label,
        "tag": None if args.id else args.tag,
        "ids": args.id,
        "benchmark_count": len(specs),
        "provider": settings.active_provider,
        "model": settings.active_model,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Run: {out_dir.relative_to(_ROOT)}")
    print(f"Benchmarks: {len(specs)}  Provider: {settings.active_provider}  Model: {settings.active_model}")

    results = asyncio.run(_run_all(specs, out_dir))

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    _print_summary(results)
    print(f"\nRun saved to: {out_dir.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
