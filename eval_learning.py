"""
eval_learning.py — Evaluation of the restaurant agent's learning capacity.

Measures whether --learn mode (injecting past strategy rules) improves search
behaviour compared to baseline (no injection), across 25 requests in 5 groups.

Three phases:
  Phase 1 — Seed runs: run each group's seed request with reflection to populate
             learnings.md with rules for each category.
  Phase 2 — Baseline runs: run all 25 requests with no learning injection.
  Phase 3 — Learning mode runs: run all 25 requests with relevant rules injected.

Results are saved to eval_results.json and a summary table is printed.

Usage:
    python eval_learning.py
    python eval_learning.py --output my_results.json
    python eval_learning.py --dry-run   # print dataset only, no API calls
"""

import argparse
import contextlib
import dataclasses
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

# Ensure the restaurant_agent module is importable when running from any cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from restaurant_agent import load_relevant_learnings, reflect_and_learn, run_agent


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class RunSpec:
    group: str        # "A"–"E"
    group_name: str   # human label
    index: int        # 0–4 within group; 0 = seed
    request: str
    is_seed: bool


def _make_group(group: str, name: str, requests: list[str]) -> list[RunSpec]:
    return [
        RunSpec(group=group, group_name=name, index=i, request=r, is_seed=(i == 0))
        for i, r in enumerate(requests)
    ]


DATASET: list[RunSpec] = [
    # Group A — Budget/price-sensitive dining in Paris
    # Tests: cluster-anchoring, French-language price queries, listicle-first rules
    *_make_group("A", "Budget Paris", [
        "cheap ramen in Paris under 15 euros",           # seed
        "affordable pho in Paris under 12 euros",
        "budget sushi in Paris under 20 euros",
        "cheap Vietnamese food in Paris under 15 euros",
        "inexpensive Japanese noodles in Paris under 18 euros",
    ]),

    # Group B — Romantic/occasion dining
    # Tests: vibe-tiered discovery, editorial-roundup-first, ambiance-keyword queries
    *_make_group("B", "Romantic Dining", [
        "romantic dinner for two in Amsterdam",          # seed
        "intimate anniversary dinner in Barcelona",
        "date night restaurant in Paris with candlelit atmosphere",
        "special occasion dinner in Vienna",
        "romantic restaurant in Copenhagen for a proposal",
    ]),

    # Group C — Niche dietary restrictions
    # Tests: editorial-roundup-first for niche categories (vegan/halal/GF)
    *_make_group("C", "Niche Dietary", [
        "vegan friendly dinner in London for a date night",  # seed
        "halal fine dining in Paris",
        "gluten-free restaurant in Berlin",
        "vegan fine dining in Amsterdam",
        "halal sushi in London",
    ]),

    # Group D — Location-constrained search
    # Tests: editorial-roundup for filter phrases, Reddit pivot when results are off
    *_make_group("D", "Location Constrained", [
        "restaurant in SF very close to 101 and with easy parking",  # seed
        "restaurant near Eiffel Tower Paris with easy parking",
        "restaurant walking distance from Sagrada Familia in Barcelona",
        "restaurant near Tokyo Station with late night hours",
        "restaurant near Grand Central Station New York open late",
    ]),

    # Group E — Cheap authentic ethnic cuisine in a city cluster
    # Tests: geographic-cluster anchoring, listicle-first, price-cap strategies
    *_make_group("E", "Cheap Ethnic Cuisine", [
        "cheap authentic Thai food in London under 15 pounds",   # seed
        "budget Korean BBQ in London under 20 pounds",
        "cheap authentic Mexican food in Los Angeles under 15 dollars",
        "inexpensive authentic Chinese food in NYC under 15 dollars",
        "cheap authentic Ethiopian food in Washington DC under 15 dollars",
    ]),
]

SEEDS = [s for s in DATASET if s.is_seed]


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    run_id: str             # e.g. "A_2_baseline"
    group: str
    group_name: str
    request: str
    is_seed: bool
    mode: str               # "baseline" | "learn"
    injected_rules: str     # "" if baseline or no match found
    queries: list[str]
    search_count: int
    strategy_labels: list[str]  # unique labels from [strategy] stdout markers
    answer: str
    answer_length: int
    timestamp: str          # ISO 8601
    error: str | None


# ---------------------------------------------------------------------------
# Stdout capture + strategy label parsing
# ---------------------------------------------------------------------------

class _Tee(io.StringIO):
    """Writes to both an in-memory buffer and the real stdout simultaneously."""
    def write(self, s: str) -> int:
        super().write(s)
        sys.__stdout__.write(s)
        return len(s)

    def flush(self):
        super().flush()
        sys.__stdout__.flush()


_STRATEGY_RE = re.compile(r'\[Search \d+/\d+\] \[([^\]]+)\]')


def _parse_strategy_labels(stdout: str) -> list[str]:
    """Extract unique strategy labels from captured agent stdout, preserving order."""
    seen: dict[str, None] = {}
    for match in _STRATEGY_RE.finditer(stdout):
        seen[match.group(1)] = None
    return list(seen)


def _capture_run(request: str, injected_learnings: str = "") -> tuple[str, list[str], str]:
    """
    Run the agent while capturing stdout. Returns (answer, queries, captured_stdout).
    The captured stdout is also mirrored to the real stdout via the Tee.
    """
    tee = _Tee()
    with contextlib.redirect_stdout(tee):
        answer, queries = run_agent(request, injected_learnings=injected_learnings)
    return answer, queries, tee.getvalue()


# ---------------------------------------------------------------------------
# Single-run executor with retry
# ---------------------------------------------------------------------------

def _execute_run(spec: RunSpec, mode: str) -> RunRecord:
    """
    Execute one evaluation run. Retries once on failure (15 s sleep).
    Returns a RunRecord; sets error field if both attempts fail.
    """
    run_id = f"{spec.group}_{spec.index}_{mode}"
    injected_rules = ""
    if mode == "learn":
        injected_rules = load_relevant_learnings(spec.request)

    for attempt in range(2):
        try:
            answer, queries, captured = _capture_run(spec.request, injected_rules)

            if not answer:
                raise RuntimeError("run_agent returned empty answer (unexpected stop_reason)")

            return RunRecord(
                run_id=run_id,
                group=spec.group,
                group_name=spec.group_name,
                request=spec.request,
                is_seed=spec.is_seed,
                mode=mode,
                injected_rules=injected_rules,
                queries=queries,
                search_count=len(queries),
                strategy_labels=_parse_strategy_labels(captured),
                answer=answer,
                answer_length=len(answer),
                timestamp=datetime.now(timezone.utc).isoformat(),
                error=None,
            )
        except Exception as exc:
            if attempt == 0:
                print(f"\n  [Retry in 15s after error: {exc}]")
                time.sleep(15)
            else:
                print(f"\n  [Run {run_id} failed after retry: {exc}]")
                return RunRecord(
                    run_id=run_id,
                    group=spec.group,
                    group_name=spec.group_name,
                    request=spec.request,
                    is_seed=spec.is_seed,
                    mode=mode,
                    injected_rules=injected_rules,
                    queries=[],
                    search_count=0,
                    strategy_labels=[],
                    answer="",
                    answer_length=0,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    error=str(exc),
                )

    # unreachable, but satisfies type checker
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def run_phase_1(seeds: list[RunSpec]) -> None:
    """
    Seed learnings.md by running each group's seed request with simple reflection.
    Results are not recorded — this phase only populates learnings.md.
    """
    print("\n" + "═" * 70)
    print("PHASE 1 — Seeding learnings.md (5 seed requests + reflection)")
    print("═" * 70)

    for seed in seeds:
        print(f"\n[Seed {seed.group}] {seed.request}")
        print("-" * 60)
        answer, queries, _ = _capture_run(seed.request)
        if answer:
            reflect_and_learn(seed.request, queries, answer, mode="simple")
        else:
            print(f"  [Warning] Seed {seed.group} returned empty answer; reflection skipped]")


def run_phase_2(dataset: list[RunSpec]) -> list[RunRecord]:
    """Baseline runs — all 25 requests with no learning injection."""
    print("\n" + "═" * 70)
    print("PHASE 2 — Baseline runs (25 requests, no learning injection)")
    print("═" * 70)

    records: list[RunRecord] = []
    for i, spec in enumerate(dataset, 1):
        print(f"\n[{i}/25] [{spec.group}{spec.index}] baseline | {spec.request}")
        print("-" * 60)
        records.append(_execute_run(spec, mode="baseline"))
    return records


def run_phase_3(dataset: list[RunSpec]) -> list[RunRecord]:
    """Learning mode runs — all 25 requests with relevant learnings injected."""
    print("\n" + "═" * 70)
    print("PHASE 3 — Learning mode runs (25 requests, learnings injected where applicable)")
    print("═" * 70)

    records: list[RunRecord] = []
    for i, spec in enumerate(dataset, 1):
        print(f"\n[{i}/25] [{spec.group}{spec.index}] learn    | {spec.request}")
        print("-" * 60)
        records.append(_execute_run(spec, mode="learn"))
    return records


# ---------------------------------------------------------------------------
# Results serialization
# ---------------------------------------------------------------------------

def save_results(records: list[RunRecord], path: str) -> None:
    """Write all RunRecords to a JSON file atomically (tmp file + rename)."""
    data = [dataclasses.asdict(r) for r in records]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    print(f"\nResults saved to {path} ({len(records)} records)")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(records: list[RunRecord]) -> None:
    baseline = {(r.group, r.index): r for r in records if r.mode == "baseline"}
    learn = {(r.group, r.index): r for r in records if r.mode == "learn"}

    groups = ["A", "B", "C", "D", "E"]
    group_names = {s.group: s.group_name for s in DATASET}

    def group_stats(g: str) -> dict:
        base_g = [v for (grp, _), v in baseline.items() if grp == g and not v.error]
        learn_g = [v for (grp, _), v in learn.items() if grp == g and not v.error]

        avg_base = mean(r.search_count for r in base_g) if base_g else 0.0
        avg_learn = mean(r.search_count for r in learn_g) if learn_g else 0.0

        injected = sum(1 for r in learn_g if r.injected_rules)
        total_learn = len([v for (grp, _), v in learn.items() if grp == g])

        # Strategy delta: pairs where strategy label sets differ
        pairs = [(baseline.get((g, i)), learn.get((g, i))) for i in range(5)]
        strategy_delta = sum(
            1 for b, l in pairs
            if b and l and not b.error and not l.error
            and set(b.strategy_labels) != set(l.strategy_labels)
        )
        return {
            "avg_base": avg_base,
            "avg_learn": avg_learn,
            "injected": injected,
            "total": total_learn,
            "strategy_delta": strategy_delta,
        }

    all_stats = {g: group_stats(g) for g in groups}

    total_base_valid = [v for v in baseline.values() if not v.error]
    total_learn_valid = [v for v in learn.values() if not v.error]
    overall_avg_base = mean(r.search_count for r in total_base_valid) if total_base_valid else 0.0
    overall_avg_learn = mean(r.search_count for r in total_learn_valid) if total_learn_valid else 0.0
    overall_injected = sum(1 for r in learn.values() if r.injected_rules)
    overall_delta = sum(s["strategy_delta"] for s in all_stats.values())

    base_errors = sum(1 for r in baseline.values() if r.error)
    learn_errors = sum(1 for r in learn.values() if r.error)

    col_w = 20
    print("\n" + "═" * 75)
    print(f"EVALUATION SUMMARY  (50 runs: 25 baseline + 25 learning mode)")
    print("═" * 75)
    header = (
        f"{'Group':<{col_w}} │ {'Searches (base)':^15} │ {'Searches (learn)':^16} │"
        f" {'Rules injected':^14} │ {'Strategy Δ':^10}"
    )
    print(header)
    print("─" * 75)

    for g in groups:
        s = all_stats[g]
        name = f"{g}  {group_names[g]}"
        injected_pct = f"{s['injected']}/{s['total']} ({100*s['injected']//s['total'] if s['total'] else 0}%)"
        print(
            f"{name:<{col_w}} │ {s['avg_base']:^15.1f} │ {s['avg_learn']:^16.1f} │"
            f" {injected_pct:^14} │ {s['strategy_delta']:^3}/5"
        )

    print("─" * 75)
    overall_injected_pct = f"{overall_injected}/25 ({100*overall_injected//25}%)"
    print(
        f"{'OVERALL':<{col_w}} │ {overall_avg_base:^15.1f} │ {overall_avg_learn:^16.1f} │"
        f" {overall_injected_pct:^14} │ {overall_delta:^3}/25"
    )
    print("═" * 75)
    print(f"Errors: {base_errors} baseline, {learn_errors} learn")

    # Per-request detail for runs where strategy changed
    changed_pairs = [
        (baseline[(g, i)], learn[(g, i)])
        for g in groups for i in range(5)
        if (g, i) in baseline and (g, i) in learn
        and not baseline[(g, i)].error and not learn[(g, i)].error
        and set(baseline[(g, i)].strategy_labels) != set(learn[(g, i)].strategy_labels)
    ]
    if changed_pairs:
        print(f"\nStrategy changed in {len(changed_pairs)} runs:")
        for b, l in changed_pairs:
            print(f"  [{b.group}] {b.request[:55]}")
            print(f"    base:  {b.strategy_labels}")
            print(f"    learn: {l.strategy_labels}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the restaurant agent's learning capacity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output", default="eval_results.json",
        help="Path for the JSON results file (default: eval_results.json)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the dataset and exit without making any API calls.",
    )
    args = parser.parse_args()

    if args.dry_run:
        print(f"Dataset: {len(DATASET)} requests across {len(SEEDS)} groups\n")
        for spec in DATASET:
            tag = "[SEED]" if spec.is_seed else "      "
            print(f"  {spec.group}{spec.index} {tag} {spec.request}")
        return

    output_path = str(Path(__file__).parent / args.output)

    # Phase 1: populate learnings.md with seed reflections
    run_phase_1(SEEDS)

    # Phase 2: baseline (no injection)
    baseline_records = run_phase_2(DATASET)

    # Phase 3: learning mode (inject relevant rules)
    learn_records = run_phase_3(DATASET)

    # Save + summarise
    all_records = baseline_records + learn_records
    save_results(all_records, output_path)
    print_summary(all_records)


if __name__ == "__main__":
    main()
