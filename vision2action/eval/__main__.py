"""CLI entry point: run the evaluation suite and print metrics.

    python -m vision2action.eval                      # oracle backend, default matrix
    python -m vision2action.eval --seeds 0 1 2 --json out.json --overlays runs/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from vision2action.config import Config
from vision2action.eval.evaluator import Evaluator, summarize
from vision2action.eval.scenarios import DEFAULT_INSTRUCTIONS, Scenario, default_scenarios

_HEADER = f"{'instruction':<22}{'seed':>5}{'det':>5}{'conf':>6}{'locErr_cm':>10}{'nav':>5}{'finalD_m':>9}{'steps':>7}{'phase':>8}"


def _print_table(results) -> None:
    print(_HEADER)
    print("-" * len(_HEADER))
    for r in results:
        loc = f"{r.localization_error_m * 100:.1f}" if r.localization_error_m == r.localization_error_m else "  nan"
        print(
            f"{r.instruction:<22}{r.seed:>5}"
            f"{('Y' if r.detected else '.'):>5}{r.confidence:>6.2f}{loc:>10}"
            f"{('Y' if r.nav_success else '.'):>5}{r.final_distance_m:>9.2f}{r.steps:>7}{r.phase:>8}"
        )


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vision2Action evaluation suite")
    p.add_argument("--backend", choices=("oracle",), default="oracle",
                   help="oracle ground-truth projection backend")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="layout seeds to evaluate")
    p.add_argument("--instructions", nargs="+", default=None, help="override the instruction list")
    p.add_argument("--stop-distance", type=float, default=None, help="override stop distance (m)")
    p.add_argument("--max-steps", type=int, default=6000, help="per-episode step cap")
    p.add_argument("--json", type=str, default=None, help="write full per-episode results to this JSON file")
    p.add_argument("--overlays", type=str, default=None, help="save final overlay PNGs to this directory")
    p.add_argument("--log", default="WARNING", help="logging level")
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=args.log.upper(), format="%(levelname)s %(name)s: %(message)s")

    instructions = tuple(args.instructions) if args.instructions else DEFAULT_INSTRUCTIONS
    if args.seeds:
        scenarios = [Scenario(i, s) for s in args.seeds for i in instructions]
    else:
        scenarios = default_scenarios(instructions=instructions)

    cfg = Config()
    if args.stop_distance is not None:
        cfg.nav.stop_distance_m = args.stop_distance

    evaluator = Evaluator(
        config=cfg,
        backend=args.backend,
        max_steps=args.max_steps,
        save_overlays_to=Path(args.overlays) if args.overlays else None,
    )
    try:
        results = evaluator.run_suite(scenarios)
    finally:
        evaluator.close()

    print(f"\nVision2Action evaluation - backend={args.backend}, "
          f"stop={cfg.nav.stop_distance_m:.2f}m, {len(results)} episodes\n")
    _print_table(results)

    summary = summarize(results)
    print("\nsummary:")
    for k, v in summary.items():
        print(f"  {k:<28} {v}")

    if args.json:
        payload = {
            "backend": args.backend,
            "stop_distance_m": cfg.nav.stop_distance_m,
            "summary": summary,
            "episodes": [r.as_row() for r in results],
        }
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json}")

    # Non-zero exit if the suite regressed to no successful navigation.
    return 0 if summary.get("navigation_success_rate", 0) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
