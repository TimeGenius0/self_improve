"""
eval_learning.py — Evaluation of the restaurant agent's learning capacity.

Measures whether learning mode improves search strategy through progressive
accumulation: each request is run as a baseline/learn pair, then reflection
and scoring happen immediately so the next request benefits from what was
just learned. Consolidation runs every CONSOLIDATION_INTERVAL learn-mode runs
to prune and merge the ruleset.

Execution order for each of the 25 requests:
  1. Baseline run   — no learning injection, no reflection
  2. Learn run      — injects rules accumulated from all prior requests
  3. Reflect        — reflects on the learn run, appends new rules to learnings.md
  4. Score          — updates rule scores using the baseline vs. learn search delta
  5. Consolidate    — every CONSOLIDATION_INTERVAL learn runs, prunes the ruleset

This interleaved design means later requests benefit from a progressively richer
and higher-quality ruleset than earlier ones.

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
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from restaurant_agent import (
    CONSOLIDATION_INTERVAL,
    LEARNINGS_FILE,
    consolidate_learnings,
    load_relevant_learnings,
    reflect_and_learn,
    run_agent,
    score_injected_rules,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class RunSpec:
    group: str        # "A"–"E"
    group_name: str
    index: int        # 0–4 within group; 0 = first request in group
    request: str


def _make_group(group: str, name: str, requests: list[str]) -> list[RunSpec]:
    return [RunSpec(group=group, group_name=name, index=i, request=r)
            for i, r in enumerate(requests)]


DATASET: list[RunSpec] = [
    # Group A — Budget/price-sensitive dining in Paris
    *_make_group("A", "Budget Paris", [
        "cheap ramen in Paris under 15 euros",
        "affordable pho in Paris under 12 euros",
        "budget sushi in Paris under 20 euros",
        "cheap Vietnamese food in Paris under 15 euros",
        "inexpensive Japanese noodles in Paris under 18 euros",
    ]),

    # Group B — Romantic/occasion dining
    *_make_group("B", "Romantic Dining", [
        "romantic dinner for two in Amsterdam",
        "intimate anniversary dinner in Barcelona",
        "date night restaurant in Paris with candlelit atmosphere",
        "special occasion dinner in Vienna",
        "romantic restaurant in Copenhagen for a proposal",
    ]),

    # Group C — Niche dietary restrictions
    *_make_group("C", "Niche Dietary", [
        "vegan friendly dinner in London for a date night",
        "halal fine dining in Paris",
        "gluten-free restaurant in Berlin",
        "vegan fine dining in Amsterdam",
        "halal sushi in London",
    ]),

    # Group D — Location-constrained search
    *_make_group("D", "Location Constrained", [
        "restaurant in SF very close to 101 and with easy parking",
        "restaurant near Eiffel Tower Paris with easy parking",
        "restaurant walking distance from Sagrada Familia in Barcelona",
        "restaurant near Tokyo Station with late night hours",
        "restaurant near Grand Central Station New York open late",
    ]),

    # Group E — Cheap authentic ethnic cuisine in a city cluster
    *_make_group("E", "Cheap Ethnic Cuisine", [
        "cheap authentic Thai food in London under 15 pounds",
        "budget Korean BBQ in London under 20 pounds",
        "cheap authentic Mexican food in Los Angeles under 15 dollars",
        "inexpensive authentic Chinese food in NYC under 15 dollars",
        "cheap authentic Ethiopian food in Washington DC under 15 dollars",
    ]),
]


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    run_id: str
    group: str
    group_name: str
    request: str
    sequential_index: int    # position in the 25-request evaluation order (0–24)
    mode: str                # "baseline" | "learn"
    injected_rules: str      # "" if baseline or no match found
    queries: list[str]
    search_count: int
    strategy_labels: list[str]
    answer: str
    answer_length: int
    timestamp: str
    error: str | None


# ---------------------------------------------------------------------------
# Stdout capture + strategy label parsing
# ---------------------------------------------------------------------------

class _Tee(io.StringIO):
    """Writes to both an in-memory buffer and real stdout simultaneously."""
    def write(self, s: str) -> int:
        super().write(s)
        sys.__stdout__.write(s)
        return len(s)

    def flush(self):
        super().flush()
        sys.__stdout__.flush()


_STRATEGY_RE = re.compile(r'\[Search \d+/\d+\] \[([^\]]+)\]')


def _parse_strategy_labels(stdout: str) -> list[str]:
    seen: dict[str, None] = {}
    for match in _STRATEGY_RE.finditer(stdout):
        seen[match.group(1)] = None
    return list(seen)


def _capture_run(request: str, injected_learnings: str = "") -> tuple[str, list[str], str]:
    tee = _Tee()
    with contextlib.redirect_stdout(tee):
        answer, queries = run_agent(request, injected_learnings=injected_learnings)
    return answer, queries, tee.getvalue()


# ---------------------------------------------------------------------------
# Single-run executor
# ---------------------------------------------------------------------------

def _execute_run(spec: RunSpec, mode: str, seq_index: int) -> RunRecord:
    """Execute one run with a single retry on failure."""
    run_id = f"{spec.group}_{spec.index}_{mode}"
    injected_rules = ""
    if mode == "learn":
        injected_rules = load_relevant_learnings(spec.request)

    for attempt in range(2):
        try:
            answer, queries, captured = _capture_run(spec.request, injected_rules)
            if not answer:
                raise RuntimeError("run_agent returned empty answer")

            return RunRecord(
                run_id=run_id,
                group=spec.group,
                group_name=spec.group_name,
                request=spec.request,
                sequential_index=seq_index,
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
                print(f"\n  [Run {run_id} failed: {exc}]")
                return RunRecord(
                    run_id=run_id,
                    group=spec.group,
                    group_name=spec.group_name,
                    request=spec.request,
                    sequential_index=seq_index,
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

    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Interleaved evaluation loop
# ---------------------------------------------------------------------------

def run_interleaved_evaluation(dataset: list[RunSpec], output_path: str) -> list[RunRecord]:
    """
    For each request in order:
      1. Baseline run  — no injection, no reflection
      2. Learn run     — inject rules accumulated from all prior requests
      3. Reflect       — reflect on learn run, append new rules
      4. Score         — update rule scores using the search delta
      5. Consolidate   — every CONSOLIDATION_INTERVAL learn runs

    Results are saved to disk after every request pair so progress is preserved.
    """
    all_records: list[RunRecord] = []
    learn_run_count = 0

    print("\n" + "═" * 70)
    print("INTERLEAVED EVALUATION — 25 requests × (baseline + learn)")
    print("Each learn run reflects and scores immediately; later requests")
    print("benefit from rules accumulated and consolidated over prior runs.")
    print("═" * 70)

    for seq_index, spec in enumerate(dataset):
        print(f"\n{'─'*70}")
        print(f"[{seq_index+1:02d}/25] {spec.group}{spec.index} | {spec.request}")
        print(f"{'─'*70}")

        # 1. Baseline
        print("\n  → Baseline run")
        baseline = _execute_run(spec, mode="baseline", seq_index=seq_index)
        all_records.append(baseline)

        # 2. Learn run
        print("\n  → Learn run")
        if not baseline.error:
            # Indicate how many rules are in play
            rule_count = sum(
                1 for line in LEARNINGS_FILE.read_text().splitlines()
                if line.strip().startswith("-")
            ) if LEARNINGS_FILE.exists() else 0
            print(f"  [Learning pool: {rule_count} rules in learnings.md]")

        learn = _execute_run(spec, mode="learn", seq_index=seq_index)
        all_records.append(learn)

        # 3. Reflect on learn run → save new rules
        if not learn.error:
            print("\n  → Reflecting on learn run")
            tee = _Tee()
            with contextlib.redirect_stdout(tee):
                reflect_and_learn(learn.request, learn.queries, learn.answer, mode="simple")
            learn_run_count += 1

            # 4. Score the injected rules
            if not baseline.error and learn.injected_rules:
                delta = baseline.search_count - learn.search_count
                score_injected_rules(learn.injected_rules, delta)

            # 5. Consolidate every CONSOLIDATION_INTERVAL learn runs
            if learn_run_count % CONSOLIDATION_INTERVAL == 0:
                consolidate_learnings()

        # Save incrementally after each pair
        save_results(all_records, output_path)
        print(
            f"\n  [Pair complete] baseline={baseline.search_count} searches, "
            f"learn={learn.search_count} searches, "
            f"delta={baseline.search_count - learn.search_count:+d}"
        )

    return all_records


# ---------------------------------------------------------------------------
# Results serialization
# ---------------------------------------------------------------------------

def save_results(records: list[RunRecord], path: str) -> None:
    data = [dataclasses.asdict(r) for r in records]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(records: list[RunRecord]) -> None:
    baseline_map = {(r.group, r.request): r for r in records if r.mode == "baseline"}
    learn_map    = {(r.group, r.request): r for r in records if r.mode == "learn"}

    groups = ["A", "B", "C", "D", "E"]
    group_names = {s.group: s.group_name for s in DATASET}

    def group_stats(g: str) -> dict:
        specs_g = [s for s in DATASET if s.group == g]
        base_g  = [baseline_map[(g, s.request)] for s in specs_g if (g, s.request) in baseline_map]
        learn_g = [learn_map[(g, s.request)]    for s in specs_g if (g, s.request) in learn_map]

        valid_base  = [r for r in base_g  if not r.error]
        valid_learn = [r for r in learn_g if not r.error]

        avg_base  = mean(r.search_count for r in valid_base)  if valid_base  else 0.0
        avg_learn = mean(r.search_count for r in valid_learn) if valid_learn else 0.0

        injected = sum(1 for r in valid_learn if r.injected_rules)
        strategy_delta = sum(
            1 for s in specs_g
            if (g, s.request) in baseline_map and (g, s.request) in learn_map
            and not baseline_map[(g, s.request)].error
            and not learn_map[(g, s.request)].error
            and set(baseline_map[(g, s.request)].strategy_labels)
               != set(learn_map[(g, s.request)].strategy_labels)
        )
        return {
            "avg_base": avg_base, "avg_learn": avg_learn,
            "injected": injected, "total": len(valid_learn),
            "strategy_delta": strategy_delta,
        }

    # ── per-group table ──────────────────────────────────────────────────────
    col_w = 22
    print("\n" + "═" * 77)
    print("EVALUATION SUMMARY  (50 runs: 25 baseline + 25 learning mode)")
    print("═" * 77)
    print(
        f"{'Group':<{col_w}} │ {'Base searches':^13} │ {'Learn searches':^14} │"
        f" {'Injected':^10} │ {'Strategy Δ':^10}"
    )
    print("─" * 77)
    all_stats = {g: group_stats(g) for g in groups}
    for g in groups:
        s = all_stats[g]
        name = f"{g}  {group_names[g]}"
        inj  = f"{s['injected']}/{s['total']}"
        print(
            f"{name:<{col_w}} │ {s['avg_base']:^13.1f} │ {s['avg_learn']:^14.1f} │"
            f" {inj:^10} │ {s['strategy_delta']:^3}/5"
        )

    valid_b = [r for r in records if r.mode == "baseline" and not r.error]
    valid_l = [r for r in records if r.mode == "learn"    and not r.error]
    ov_b  = mean(r.search_count for r in valid_b) if valid_b else 0.0
    ov_l  = mean(r.search_count for r in valid_l) if valid_l else 0.0
    ov_inj = sum(1 for r in valid_l if r.injected_rules)
    ov_dlt = sum(s["strategy_delta"] for s in all_stats.values())
    print("─" * 77)
    print(
        f"{'OVERALL':<{col_w}} │ {ov_b:^13.1f} │ {ov_l:^14.1f} │"
        f" {ov_inj:^3}/25      │ {ov_dlt:^3}/25"
    )
    print("═" * 77)

    # ── learning trend across thirds ─────────────────────────────────────────
    # Split the *active* dataset into thirds by sequential_index so the trend
    # reflects actual run order rather than fixed DATASET positions.
    active_pairs = sorted(
        [
            (baseline_map[(s.group, s.request)], learn_map[(s.group, s.request)])
            for s in DATASET
            if (s.group, s.request) in baseline_map and (s.group, s.request) in learn_map
        ],
        key=lambda pair: pair[0].sequential_index,
    )
    n = len(active_pairs)
    if n >= 3:
        t1, t2 = n // 3, 2 * (n // 3)
        thirds = [
            (active_pairs[:t1],        f"Early  (1–{t1})"),
            (active_pairs[t1:t2],      f"Mid    ({t1+1}–{t2})"),
            (active_pairs[t2:],        f"Late   ({t2+1}–{n})"),
        ]
        print("\nLearning trend — does later benefit more from accumulated rules?")
        print(f"  {'Segment':<18} │ {'Avg Δ (base−learn)':^20} │ {'Injected':^10}")
        print(f"  {'─'*18}─┼─{'─'*20}─┼─{'─'*10}")
        for pairs, label in thirds:
            valid = [(b, l) for b, l in pairs if not b.error and not l.error]
            if valid:
                avg_delta = mean(b.search_count - l.search_count for b, l in valid)
                inj_count = sum(1 for _, l in valid if l.injected_rules)
                print(f"  {label:<18} │ {avg_delta:^+20.2f} │ {inj_count}/{len(valid)}")

    # ── errors ───────────────────────────────────────────────────────────────
    base_errors  = sum(1 for r in records if r.mode == "baseline" and r.error)
    learn_errors = sum(1 for r in records if r.mode == "learn"    and r.error)
    print(f"\nErrors: {base_errors} baseline, {learn_errors} learn")


# ---------------------------------------------------------------------------
# Learnings file management
# ---------------------------------------------------------------------------

def backup_and_reset_learnings() -> None:
    """
    Back up the current learnings.md and start fresh for the evaluation.
    A clean slate ensures all learning during the eval is from the eval itself.
    """
    if LEARNINGS_FILE.exists() and LEARNINGS_FILE.stat().st_size > 0:
        backup = LEARNINGS_FILE.with_name("learnings_pre_eval.md")
        backup.write_text(LEARNINGS_FILE.read_text())
        print(f"[Backed up existing learnings to {backup.name}]")
    LEARNINGS_FILE.write_text("# Restaurant Agent — Strategy Learnings\n\n")
    print("[learnings.md reset for fresh evaluation]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the restaurant agent's learning capacity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            examples:
              python eval_learning.py                      # all 25 requests
              python eval_learning.py --groups A           # Group A only (5 requests)
              python eval_learning.py --groups A C E       # three groups (15 requests)
              python eval_learning.py --groups A --dry-run # preview group A
        """),
    )
    parser.add_argument(
        "--output", default="eval_results.json",
        help="Path for the JSON results file (default: eval_results.json)",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        metavar="GROUP",
        choices=["A", "B", "C", "D", "E"],
        help="Run only the specified group(s). Choices: A B C D E. Default: all.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the selected dataset and exit without making any API calls.",
    )
    args = parser.parse_args()

    active = [s for s in DATASET if args.groups is None or s.group in args.groups]
    n_groups = len(set(s.group for s in active))

    if args.dry_run:
        print(f"Dataset: {len(active)} requests in {n_groups} group(s)\n")
        for spec in active:
            print(f"  {spec.group}{spec.index}  {spec.request}")
        consolidations = len(active) // CONSOLIDATION_INTERVAL
        print(f"\nConsolidation every {CONSOLIDATION_INTERVAL} learn runs "
              f"→ {consolidations} consolidation(s) during this eval")
        return

    output_path = str(Path(__file__).parent / args.output)

    backup_and_reset_learnings()

    records = run_interleaved_evaluation(active, output_path)

    save_results(records, output_path)
    print(f"\nFinal results saved to {output_path} ({len(records)} records)")

    print_summary(records)


if __name__ == "__main__":
    main()
