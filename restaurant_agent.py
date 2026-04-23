"""
Restaurant suggestion agent with self-reflection and learning.

This script has three components that build on each other:

  1. AGENT  (always active)
     Searches the web up to MAX_SEARCHES times to find 3 restaurants that
     match the user's request, then presents them with the specific details
     the user asked for (prices, ambiance, menu items, etc.).

  2. REFLECTION  (runs after every request)
     Analyses the search strategy used, proposes alternatives, and writes
     genuinely better rules to learnings.md. Two modes:
       --reflect simple    Pure reasoning — fast, no extra searches (default).
       --reflect advanced  Actually executes the 3 alternatives via live searches,
                           compares real results, then extracts learnings.

  3. LEARNING MODE  (opt-in via --learn)
     Before running, reads learnings.md and injects the rules relevant to
     the current request into the agent's system prompt so it applies past
     lessons automatically.

Usage:
    python restaurant_agent.py                           # basic
    python restaurant_agent.py --learn                   # apply past learnings
    python restaurant_agent.py --reflect advanced        # deep reflection
    python restaurant_agent.py --learn --reflect advanced  # both

Install:
    pip install anthropic ddgs
"""

import argparse
import json
import re
import textwrap
from datetime import datetime
from pathlib import Path

import anthropic

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS


# ---------------------------------------------------------------------------
# Shared infrastructure
# ---------------------------------------------------------------------------

MAX_SEARCHES = 10
MAX_REFLECTION_SEARCHES = 15  # total budget across the 3 alternative strategies
CONSOLIDATION_INTERVAL = 10   # consolidate learnings every N learn-mode runs
LEARNINGS_FILE = Path(__file__).parent / "learnings.md"

# Matches the score annotation appended to every rule: "  [u:3 h:2]"
_SCORE_RE = re.compile(r'\s*\[u:(\d+) h:(\d+)\]$')

client = anthropic.Anthropic()


def _strip_score(text: str) -> str:
    """Remove the [u:N h:M] annotation from a rule bullet, returning clean rule text."""
    return _SCORE_RE.sub('', text).strip()


def _parse_score(text: str) -> tuple[int, int]:
    """Return (used_count, helped_count) from a rule bullet, or (0, 0) if not annotated."""
    m = _SCORE_RE.search(text)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def _search(query: str, strategy: str = "default") -> str:
    """
    Execute a DuckDuckGo search and return results as a JSON string.

    The `strategy` label is included in the response so the reflection
    agent can tell which strategy produced which results when comparing
    multiple approaches side-by-side.
    """
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=6))
        results = [
            {"title": r.get("title", ""), "snippet": r.get("body", ""), "url": r.get("href", "")}
            for r in raw
        ]
        return json.dumps(
            {"strategy": strategy, "query": query, "results": results},
            ensure_ascii=False,
            indent=2,
        )
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# Tool definition exposed to Claude. `strategy` is optional — the main agent
# can ignore it; the reflection agent uses it to tag searches by approach.
TOOLS = [
    {
        "name": "search_restaurants",
        "description": (
            "Search the internet for restaurants matching a query. "
            "Returns a list of results with title, snippet, and URL."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search expression.",
                },
                "strategy": {
                    "type": "string",
                    "description": (
                        "Label for the search strategy being executed "
                        "(e.g. 'broad_discovery', 'price_verification', 'reddit_forum'). "
                        "Used by the reflection agent to track which strategy produced which results."
                    ),
                },
            },
            "required": ["query"],
        },
    }
]


def _run_tool_call(block, search_count: int, max_searches: int, limit_message: str) -> tuple[dict, int, str | None]:
    """
    Execute a single tool_use block. Returns (tool_result_dict, new_count, query_or_None).
    If the search limit is already reached, returns an error result without searching.
    """
    if search_count >= max_searches:
        result = json.dumps({"error": limit_message})
        query = None
    else:
        search_count += 1
        query = block.input["query"]
        strategy = block.input.get("strategy", "default")
        result = _search(query, strategy)

    tool_result = {"type": "tool_result", "tool_use_id": block.id, "content": result}
    return tool_result, search_count, query


# ---------------------------------------------------------------------------
# Component 1 — Agent: answer a restaurant request
# ---------------------------------------------------------------------------

_AGENT_PROMPT = """You are a restaurant recommendation agent. Your job is to help the
user choose between three good restaurants that fit their request.

## Phase 1 — Understand what the user actually needs

Before searching, read the user's request carefully and identify the specific criteria
that matter to them. Examples:
- "cheap" or "budget" → they need concrete prices
- "vegan" / "gluten-free" / specific dish → they need menu items and dietary info
- "romantic" / "for a date" → they need ambiance, noise level, lighting details
- "quick lunch" → they need service pace, average time, lunch deals
- "good wine" → they need wine list quality or by-the-glass options

Keep these criteria in mind throughout — they determine what you must find out.

## Phase 2 — Find 3 candidates

Use the search tool to identify exactly 3 distinct, high-quality restaurants that
match the user's request. Be strategic with queries. After each search, evaluate what
you have: name, location, cuisine, quality signals. Discard weak results and refine.
Stop as soon as you have 3 solid, distinct candidates.

## Phase 3 — Gather the specific information the user needs

Once you have 3 candidates, use further searches to find the concrete details that
directly answer the user's criteria:
- If price matters: search for actual prices, average spend, or menu costs.
- If a specific dish or diet matters: search for menu items, ingredients, dietary labels.
- If ambiance/occasion matters: search for atmosphere descriptions, photos, setting reviews.
- If convenience matters: search for hours, wait times, reservation policies.

Only search for information that is genuinely relevant to the user's stated need.
You may search at most 10 times in total across all phases.

## Output format

Be concise. No preamble, no commentary about your search process.

For each restaurant, one tight block:
  **Name** — neighbourhood
  The one fact that makes it stand out.
  Then only the fields directly relevant to the user's request — one line each, label: value.

End with a compact "pick X if…" line per option — one sentence, no padding.

If a piece of information was not found, omit the field entirely."""


def run_agent(user_request: str, injected_learnings: str = "") -> tuple[str, list[str]]:
    """
    Run the restaurant recommendation agent.

    If `injected_learnings` is provided (learning mode), the relevant past
    search-strategy rules are appended to the system prompt before the first
    API call so the agent can apply them.

    Returns (final_answer, list_of_queries_used).
    """
    print(f"\nRequest: {user_request}")
    print("=" * 60)

    system = _AGENT_PROMPT
    if injected_learnings:
        system += textwrap.dedent(f"""

            ## Search strategy rules from past experience
            Apply these rules when planning your searches for this request:
            {injected_learnings}
        """)
        print("[Learning mode] Injected relevant past learnings.")

    messages = [{"role": "user", "content": user_request}]
    search_count = 0
    queries_used: list[str] = []
    final_answer = ""

    while True:
        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=4096,
            system=system,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    final_answer = block.text
                    print("\nRESULTS")
                    print("-" * 40)
                    print(final_answer)
            print(f"\n[Searches used: {search_count}/{MAX_SEARCHES}]")
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                tool_result, search_count, query = _run_tool_call(
                    block, search_count, MAX_SEARCHES,
                    limit_message="Search limit reached. Give your final answer.",
                )
                if query:
                    strategy = block.input.get("strategy", "default")
                    queries_used.append(query)
                    print(f"[Search {search_count}/{MAX_SEARCHES}] [{strategy}] {query}")
                tool_results.append(tool_result)

            messages.append({"role": "user", "content": tool_results})
            if search_count >= MAX_SEARCHES:
                messages.append({"role": "user", "content": "Search limit reached. Give your final answer now."})
        else:
            print(f"[Unexpected stop_reason: {response.stop_reason}]")
            break

    return final_answer, queries_used


# ---------------------------------------------------------------------------
# Component 2 — Reflection: learn from a completed run
# ---------------------------------------------------------------------------
#
# After every run, the reflection component analyses whether the search strategy
# used was optimal. It proposes alternatives and saves any genuinely better rules
# to learnings.md as reusable "if/then" guidelines.
#
# Two modes:
#   simple   — pure reasoning, no extra API searches (fast, cheap)
#   advanced — actually executes the 3 alternatives via live searches, compares
#              real results, then extracts evidence-backed learnings
# ---------------------------------------------------------------------------

_SIMPLE_REFLECTION_PROMPT = """You are a search-strategy analyst for a restaurant recommendation agent.

You will be given the user's request, the queries the agent ran, and its final answer.

Your job:

1. Describe the strategy used in one sentence.

2. Propose 3 alternative strategies — each in one sentence. Vary entry point, query
   type, or search budget split in a meaningful way.

3. For each alternative, reason briefly (2–3 sentences) on whether it would have
   performed better, worse, or the same for THIS specific request.

4. Decide whether any alternative is genuinely better — not just different, but
   concretely better (fewer searches, stronger information, fewer missed candidates).
   If none clears that bar, output LEARNINGS: none and stop.

5. If there is a genuine improvement, extract 1–3 reusable rules in this format:
   "if [condition about user request], then [specific search strategy action]"
   Rules must apply across future requests, not just this one.

Output format — use exactly these section headers:

STRATEGY USED
<one sentence>

ALTERNATIVE 1
<one sentence description>
<reasoning>

ALTERNATIVE 2
<one sentence description>
<reasoning>

ALTERNATIVE 3
<one sentence description>
<reasoning>

LEARNINGS
- <learning>
(or "none" if no alternative is genuinely better)
"""

_ADVANCED_REFLECTION_PROMPT = """You are a search-strategy analyst for a restaurant recommendation agent.
You have access to the same search tool the agent uses.

You will be given the user's request, the queries the original agent ran (labelled
strategy "original"), and its final answer.

## Step 1 — Devise 3 alternative strategies

Think of 3 meaningfully different approaches. Dimensions to vary:
- Entry point: geographic cluster vs. broad city query vs. forum/Reddit query
- Budget split: heavy discovery vs. heavy per-candidate verification
- Query type: dish-anchored vs. location-anchored vs. editorial-roundup-anchored

## Step 2 — Execute each strategy

Run each alternative using the search tool. Label every search call with the strategy
name (e.g. "cluster_first", "reddit_first", "editorial_roundup") via the `strategy`
parameter. Aim for 3–5 searches per strategy; stop early if a strategy clearly fails.

## Step 3 — Compare all 4 strategies on real results

- What candidates did each strategy surface?
- How many searches did each need to reach 3 solid candidates?
- Which found the most relevant detail (price / menu / ambiance) with fewest searches?
- Did any alternative surface candidates the original missed?

## Step 4 — Extract learnings

If any alternative is genuinely better (fewer searches, stronger candidates, fewer
information gaps), write 2–4 reusable rules:
"if [condition about user request], then [specific search strategy action]"

If no alternative clears that bar, output LEARNINGS: none.

Output format — use exactly these section headers:

STRATEGY USED
<one sentence>

ALTERNATIVE 1: <strategy name>
<one sentence description>
<2–3 sentence reasoning based on actual search results>

ALTERNATIVE 2: <strategy name>
<one sentence description>
<2–3 sentence reasoning>

ALTERNATIVE 3: <strategy name>
<one sentence description>
<2–3 sentence reasoning>

COMPARISON
<3–6 sentences comparing all four strategies on candidate quality, search count, and information relevance>

LEARNINGS
- <learning>
(or "none" if no alternative is genuinely better)
"""


def _run_simple_reflection(user_request: str, queries_used: list[str], final_answer: str) -> str:
    """Ask Claude to reason about alternative strategies without running any searches."""
    prompt = textwrap.dedent(f"""
        USER REQUEST: {user_request}

        QUERIES USED (in order):
        {chr(10).join(f"  {i+1}. {q}" for i, q in enumerate(queries_used))}

        FINAL ANSWER:
        {final_answer}
    """).strip()

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=_SIMPLE_REFLECTION_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return next((block.text for block in response.content if hasattr(block, "text")), "")


def _run_advanced_reflection(user_request: str, queries_used: list[str], final_answer: str) -> str:
    """
    Execute the 3 alternative strategies via live searches, then compare all four
    strategies (original + 3 alternatives) on real results before extracting learnings.
    """
    prompt = textwrap.dedent(f"""
        USER REQUEST: {user_request}

        ORIGINAL QUERIES USED (strategy="original"):
        {chr(10).join(f"  {i+1}. {q}" for i, q in enumerate(queries_used))}

        ORIGINAL FINAL ANSWER:
        {final_answer}
    """).strip()

    messages = [{"role": "user", "content": prompt}]
    search_count = 0

    while True:
        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=4096,
            system=_ADVANCED_REFLECTION_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return next((block.text for block in response.content if hasattr(block, "text")), "")

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                tool_result, search_count, query = _run_tool_call(
                    block, search_count, MAX_REFLECTION_SEARCHES,
                    limit_message="Reflection search limit reached. Proceed to comparison and learnings.",
                )
                if query:
                    strategy = block.input.get("strategy", "reflection")
                    print(f"  [Reflection search {search_count}/{MAX_REFLECTION_SEARCHES}] [{strategy}] {query}")
                tool_results.append(tool_result)

            messages.append({"role": "user", "content": tool_results})
            if search_count >= MAX_REFLECTION_SEARCHES:
                messages.append({
                    "role": "user",
                    "content": "Reflection search limit reached. Write your COMPARISON and LEARNINGS now.",
                })
        else:
            break

    return ""


def _save_learnings(user_request: str, reflection: str) -> bool:
    """
    Parse the LEARNINGS section of a reflection and append bullet rules to
    learnings.md. Returns True if at least one rule was saved, False otherwise
    (including when the reflection explicitly concludes LEARNINGS: none).
    """
    if "LEARNINGS" not in reflection:
        return False

    raw = reflection.split("LEARNINGS", 1)[1].strip()
    bullets = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower() == "none":
            return False
        if stripped.startswith("-"):
            # Initialise score annotation so every rule is trackable from birth
            bullets.append(f"{stripped}  [u:0 h:0]")

    if not bullets:
        return False

    if not LEARNINGS_FILE.exists():
        LEARNINGS_FILE.write_text("# Restaurant Agent — Strategy Learnings\n\n")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    block = f"\n## {timestamp} — {user_request}\n" + "\n".join(bullets) + "\n"
    with LEARNINGS_FILE.open("a") as f:
        f.write(block)

    return True


def reflect_and_learn(
    user_request: str,
    queries_used: list[str],
    final_answer: str,
    mode: str = "simple",
) -> None:
    """
    Reflect on the search strategy used in a completed run and save any
    reusable learnings to learnings.md.

    mode="simple"   — fast, no extra searches; Claude reasons hypothetically
    mode="advanced" — slower; Claude actually runs the 3 alternatives via
                      live searches before comparing and extracting learnings
    """
    print(f"\n[Reflecting on strategy... mode={mode}]")

    if mode == "advanced":
        reflection = _run_advanced_reflection(user_request, queries_used, final_answer)
    else:
        reflection = _run_simple_reflection(user_request, queries_used, final_answer)

    print("\n[REFLECTION]")
    print("-" * 40)
    print(reflection)

    saved = _save_learnings(user_request, reflection)
    if saved:
        print(f"\n[Learnings saved to {LEARNINGS_FILE}]")
    else:
        print("\n[No new learnings — no strategy was meaningfully better; nothing saved]")


# ---------------------------------------------------------------------------
# Component 3 — Learning mode: apply past learnings to a new request
# ---------------------------------------------------------------------------
#
# Before the agent runs, this component reads all "if/then" rules from
# learnings.md, filters them to those relevant to the current request using
# a fast Haiku call, and returns them ready to inject into the system prompt.
# ---------------------------------------------------------------------------

def load_relevant_learnings(user_request: str) -> str:
    """
    Return the subset of past learnings from learnings.md that are applicable
    to the current request, ready to be injected into the agent's system prompt.

    Uses claude-haiku for fast, cheap relevance filtering.
    Returns an empty string if the file is empty or no rules apply.
    """
    if not LEARNINGS_FILE.exists():
        return ""

    # Strip score annotations — the agent only needs the clean rule text
    all_bullets_clean = [
        _strip_score(line.strip())
        for line in LEARNINGS_FILE.read_text().splitlines()
        if line.strip().startswith("-")
    ]
    if not all_bullets_clean:
        return ""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": textwrap.dedent(f"""
                User request: "{user_request}"

                Below are search-strategy rules learned from past restaurant queries.
                Return only the rules directly applicable to THIS request, verbatim
                as a bullet list. If none apply, reply with exactly: none

                Rules:
                {chr(10).join(all_bullets_clean)}
            """).strip(),
        }],
    )
    result = next(
        (block.text for block in response.content if hasattr(block, "text")), ""
    ).strip()

    return "" if (not result or result.lower() == "none") else result


# ---------------------------------------------------------------------------
# Component 4 — Rule scoring and consolidation
# ---------------------------------------------------------------------------
#
# After each learning-mode run, score_injected_rules updates the [u:N h:M]
# counters on every rule that was injected: u (used) always increments;
# h (helped) increments only when learning mode used fewer searches than the
# paired baseline run (positive search_delta).
#
# consolidate_learnings periodically asks Claude to merge near-duplicate rules,
# remove consistently unhelpful ones (h/u < 0.3 with u >= 3), and produce a
# tighter ruleset. Called every CONSOLIDATION_INTERVAL learn-mode runs.
# ---------------------------------------------------------------------------

def score_injected_rules(injected_rules: str, search_delta: int) -> None:
    """
    Update used/helped counts on the rules that were injected in a learning run.

    injected_rules  — the string returned by load_relevant_learnings (clean, no annotations)
    search_delta    — baseline_search_count minus learn_search_count; positive means
                      learning mode used fewer searches (helped)
    """
    if not LEARNINGS_FILE.exists() or not injected_rules.strip():
        return

    injected_clean = {
        _strip_score(line.strip())
        for line in injected_rules.splitlines()
        if line.strip().startswith("-")
    }
    if not injected_clean:
        return

    helped = search_delta > 0
    lines = LEARNINGS_FILE.read_text().splitlines()
    updated = 0
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("-") and _strip_score(stripped) in injected_clean:
            u, h = _parse_score(stripped)
            u += 1
            h += 1 if helped else 0
            line = f"{_strip_score(stripped)}  [u:{u} h:{h}]"
            updated += 1
        new_lines.append(line)

    LEARNINGS_FILE.write_text("\n".join(new_lines))
    if updated:
        verdict = "helped" if helped else "no improvement"
        print(f"  [Scored {updated} rule(s): {verdict}, delta={search_delta:+d}]")


_CONSOLIDATION_PROMPT = """You are maintaining a search-strategy knowledge base for a restaurant recommendation agent.

You will receive a list of "if/then" rules, each annotated with [u:used h:helped].

Your job — produce a tighter, higher-quality ruleset:

1. MERGE near-duplicate rules into one. Keep the more specific/actionable phrasing.
   Set the merged rule's score to the sum of both rules' counts.

2. REMOVE rules where u >= 3 AND (h / u) < 0.30 — used at least 3 times but helped
   fewer than 30% of the time. These rules are actively misleading.

3. KEEP all rules where u < 3 — not enough data to judge yet.

4. If two rules contradict each other, keep the one with the higher h/u ratio.

Return ONLY the consolidated bullet list in this exact format (no headers, no explanation):
- rule text  [u:N h:M]
- rule text  [u:N h:M]
"""


def consolidate_learnings() -> None:
    """
    Merge near-duplicate rules and prune consistently unhelpful ones.
    Rewrites learnings.md with the consolidated ruleset.
    """
    if not LEARNINGS_FILE.exists():
        return

    all_bullets = [
        line.strip()
        for line in LEARNINGS_FILE.read_text().splitlines()
        if line.strip().startswith("-")
    ]
    if len(all_bullets) < 3:
        return

    print(f"\n[Consolidating {len(all_bullets)} rules...]")

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=_CONSOLIDATION_PROMPT,
        messages=[{"role": "user", "content": "\n".join(all_bullets)}],
    )
    result = next(
        (block.text for block in response.content if hasattr(block, "text")), ""
    )

    new_bullets = [
        line.strip()
        for line in result.splitlines()
        if line.strip().startswith("-")
    ]
    if not new_bullets:
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    content = (
        "# Restaurant Agent — Strategy Learnings\n\n"
        f"## Consolidated: {timestamp} ({len(all_bullets)} → {len(new_bullets)} rules)\n"
        + "\n".join(new_bullets) + "\n"
    )
    LEARNINGS_FILE.write_text(content)
    print(f"[Consolidation complete: {len(all_bullets)} → {len(new_bullets)} rules]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Restaurant suggestion agent with self-reflection and learning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            examples:
              python restaurant_agent.py
              python restaurant_agent.py --learn
              python restaurant_agent.py --reflect advanced
              python restaurant_agent.py --learn --reflect advanced
        """),
    )
    parser.add_argument(
        "--learn",
        action="store_true",
        help="Apply relevant past learnings from learnings.md to this run.",
    )
    parser.add_argument(
        "--reflect",
        choices=["simple", "advanced"],
        default="simple",
        help=(
            "Reflection mode after the run. "
            "'simple' reasons hypothetically (default); "
            "'advanced' executes alternatives via live searches."
        ),
    )
    args = parser.parse_args()

    request = input("What kind of restaurant are you looking for? ").strip()
    if not request:
        exit()

    # Learning mode: load and inject relevant past rules before running
    injected = ""
    if args.learn:
        injected = load_relevant_learnings(request)
        if not injected:
            print("[Learning mode] No relevant past learnings found.")

    # Run the agent
    answer, queries = run_agent(request, injected_learnings=injected)
    if not answer:
        exit()

    # Reflect on the strategy and save any new learnings
    reflect_and_learn(request, queries, answer, mode=args.reflect)
