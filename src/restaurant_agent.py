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
LEARNINGS_FILE = Path(__file__).parent.parent / "data" / "learnings.md"

# Matches the score annotation: "  [u:3 h:2]"
_SCORE_RE = re.compile(r'\s*\[u:(\d+) h:(\d+)\]')
# Matches the source annotation: "  [src: "req1" | "req2"]"
_SRC_RE = re.compile(r'\s*\[src:[^\]]*\]')

client = anthropic.Anthropic()


def _strip_score(text: str) -> str:
    """Remove [u:N h:M] and [src: ...] annotations, returning clean rule text."""
    text = _SCORE_RE.sub('', text)
    text = _SRC_RE.sub('', text)
    return text.strip()


def _strip_src(text: str) -> str:
    """Remove only the [src: ...] annotation, keeping the score annotation."""
    return _SRC_RE.sub('', text).strip()


def _strip_score_only(text: str) -> str:
    """Remove only the [u:N h:M] score annotation, keeping the [src: ...] annotation."""
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


def _run_tool_call(
    block, search_count: int, max_searches: int, limit_message: str
) -> tuple[dict, int, str | None, str | None, str | None]:
    """
    Execute a single tool_use block.
    Returns (tool_result_dict, new_count, query_or_None, strategy_or_None, result_json_or_None).
    If the search limit is already reached, returns an error result without searching.
    """
    if search_count >= max_searches:
        result = json.dumps({"error": limit_message})
        tool_result = {"type": "tool_result", "tool_use_id": block.id, "content": result}
        return tool_result, search_count, None, None, None
    else:
        search_count += 1
        query = block.input["query"]
        strategy = block.input.get("strategy", "default")
        result = _search(query, strategy)
        tool_result = {"type": "tool_result", "tool_use_id": block.id, "content": result}
        return tool_result, search_count, query, strategy, result


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


def run_agent(
    user_request: str, injected_learnings: str = ""
) -> tuple[str, list[str], list[dict]]:
    """
    Run the restaurant recommendation agent.

    If `injected_learnings` is provided (learning mode), the relevant past
    search-strategy rules are appended to the system prompt before the first
    API call so the agent can apply them.

    Returns (final_answer, list_of_queries_used, search_log).
    search_log is a list of dicts: {n, query, strategy, result_json}.
    """
    print(f"\nRequest: {user_request}")
    print("=" * 60)

    system = _AGENT_PROMPT
    if injected_learnings:
        system += textwrap.dedent(f"""

            ## Proven search constraints — read before planning any searches

            These rules were learned from real runs where the default broad-discovery
            approach wasted searches or missed better candidates. Each rule identifies
            a specific condition where the normal instinct is wrong.

            Before you decide on your first query, check whether any rule below applies
            to this request. If one does, follow it instead of defaulting to broad
            discovery — the "because" clause explains what goes wrong if you don't.

            {injected_learnings}
        """)
        print("[Learning mode] Injected relevant past learnings.")

    messages = [{"role": "user", "content": user_request}]
    search_count = 0
    queries_used: list[str] = []
    search_log: list[dict] = []
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
                tool_result, search_count, query, strategy, result_json = _run_tool_call(
                    block, search_count, MAX_SEARCHES,
                    limit_message="Search limit reached. Give your final answer.",
                )
                if query:
                    queries_used.append(query)
                    search_log.append({
                        "n": search_count,
                        "query": query,
                        "strategy": strategy,
                        "result_json": result_json,
                    })
                    print(f"[Search {search_count}/{MAX_SEARCHES}] [{strategy}] {query}")
                tool_results.append(tool_result)

            messages.append({"role": "user", "content": tool_results})
            if search_count >= MAX_SEARCHES:
                messages.append({"role": "user", "content": "Search limit reached. Give your final answer now."})
        else:
            print(f"[Unexpected stop_reason: {response.stop_reason}]")
            break

    return final_answer, queries_used, search_log


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

_SIMPLE_REFLECTION_PROMPT = """You are a search-efficiency analyst for a restaurant recommendation agent.

You receive the user's request, every search in order with its full results, and the final answer.

Your only goal: find searches that could have been skipped or merged without losing any information.

For each search ask: did this search surface anything that wasn't already in a prior search result, or that couldn't have been obtained in fewer total calls by a smarter first query?

If no search was clearly wasteful, output LEARNINGS: none.

If a more efficient path exists, write at most 2 rules. Each rule must:
- Be a concrete if/then instruction (not a general tip)
- Explain exactly which search(es) it would eliminate and why those searches returned redundant or low-value results
- Generalise to similar future requests, not just this one
- Clearly save at least 1 search call when applied

Format:
LEARNINGS
- if [condition], then [action] — saves [N] call(s) because [specific redundancy or failure observed in this run]

If no rule meets that bar: LEARNINGS: none"""

_ADVANCED_REFLECTION_PROMPT = """You are a search-efficiency analyst for a restaurant recommendation agent.
You have access to the same search tool.

You receive the original request, the queries run (labelled strategy "original"), and the final answer.

## Step 1 — Identify the bottleneck

Look at the original queries. Where did the agent spend the most calls? Was it discovery (finding candidates), verification (prices/menu/ambiance), or recovery (re-searching because an earlier query failed)? Name the costliest phase in one sentence.

## Step 2 — Test one targeted alternative

Pick the single most promising alternative that directly addresses the bottleneck. Run it (3–5 searches max, label strategy accordingly). Stop early if it clearly underperforms.

## Step 3 — Compare on search count only

Did the alternative reach the same quality answer in fewer calls? Count searches required by each approach. If the alternative is not strictly fewer calls for equivalent output, output LEARNINGS: none.

## Step 4 — Extract learnings

If the alternative saves calls, write at most 2 rules. Each rule must:
- State a concrete condition and action (not a general tip)
- Name exactly which call(s) it eliminates and why those calls were wasteful
- Generalise to similar future requests

Format:
LEARNINGS
- if [condition], then [action] — saves [N] call(s) because [specific wasteful pattern observed]

If no rule meets that bar: LEARNINGS: none"""


def _run_simple_reflection(
    user_request: str,
    queries_used: list[str],
    final_answer: str,
    search_log: list[dict] | None = None,
) -> str:
    """Analyse the actual search results to identify wasted searches and extract learnings."""
    if search_log:
        searches_block = "\n\n".join(
            f"Search {entry['n']} [{entry['strategy']}]: {entry['query']}\n"
            + entry["result_json"]
            for entry in search_log
        )
    else:
        # Fallback when no search log available (e.g. old callers)
        searches_block = "\n".join(
            f"Search {i+1}: {q}" for i, q in enumerate(queries_used)
        )

    prompt = textwrap.dedent(f"""
        USER REQUEST: {user_request}

        SEARCHES (in order, with full results):
        {searches_block}

        FINAL ANSWER:
        {final_answer}
    """).strip()

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=2048,
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
                tool_result, search_count, query, strategy, _ = _run_tool_call(
                    block, search_count, MAX_REFLECTION_SEARCHES,
                    limit_message="Reflection search limit reached. Proceed to comparison and learnings.",
                )
                if query:
                    strategy = strategy or block.input.get("strategy", "reflection")
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


_LEARNING_SYNTHESIS_PROMPT = """You maintain the rule library for a restaurant recommendation agent. The only purpose of every rule is to reduce the number of search tool calls the agent makes.

You receive a reflection (with proposed rules under LEARNINGS:) and the current rule library.

A rule earns its place only if:
1. It saves at least 1 search call on the class of requests it targets
2. It is specific enough to act on (not a general tip like "use aggregators")
3. It generalises beyond the single request that produced it

For each EXISTING rule decide:
- KEEP   — still valid, no better phrasing; preserve score and [src: ...] exactly
- UPDATE — evidence supports better phrasing; reset score to [u:0 h:0]; append current request to [src: ...]
- MERGE  — overlaps with a proposed rule; combine into one tighter rule; keep higher score; union [src: ...]
- REMOVE — contradicted by evidence, or can't plausibly save a call

For each PROPOSED rule (from LEARNINGS:):
- ADD   — passes the 3 criteria above; add [u:0 h:0] [src: "CURRENT_REQUEST"]
- MERGE — already covered by an existing rule
- SKIP  — vague, untestable, or unlikely to save a call

Rule format (every bullet must follow this exactly):
- if [condition], then [action] — saves ~N call(s) because [specific wasteful pattern]  [u:N h:M] [src: "req"]

Output:
CHANGES
- <action> rule: <one-line reason> (omit KEEPs)

RULES
- <complete updated list>

If LEARNINGS: none → CHANGES: (no changes), then existing rules verbatim."""


def _extract_proposed_learnings(reflection: str) -> list[str]:
    """Extract bullet lines from the LEARNINGS section of a reflection."""
    if "LEARNINGS" not in reflection:
        return []
    raw = reflection.split("LEARNINGS", 1)[1].strip()
    bullets = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower() == "none":
            return []
        if stripped.startswith("-"):
            bullets.append(stripped)
    return bullets


def update_learnings_from_reflection(user_request: str, reflection: str) -> bool:
    """
    Holistic learnings update: sends the full reflection evidence plus all
    existing rules to Opus, which decides keep/update/merge/remove/add for
    every rule, then rewrites learnings.md as a single living document.

    Falls back to bootstrap mode when the file is empty or missing.
    Returns True if the file was written, False if there was nothing to save.
    """
    proposed = _extract_proposed_learnings(reflection)

    existing_text = LEARNINGS_FILE.read_text() if LEARNINGS_FILE.exists() else ""
    existing_bullets = [
        line.strip()
        for line in existing_text.splitlines()
        if line.strip().startswith("-")
    ]

    # Bootstrap path: no existing rules yet — just write proposed rules directly
    if not existing_bullets:
        if not proposed:
            return False
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        bullets = [f'{b}  [u:0 h:0] [src: "{user_request}"]' for b in proposed]
        content = (
            "# Restaurant Agent — Strategy Learnings\n\n"
            f"*Last updated: {timestamp} — {user_request}*\n\n"
            + "\n".join(bullets) + "\n"
        )
        LEARNINGS_FILE.write_text(content)
        return True

    user_content = textwrap.dedent(f"""
        REFLECTION FOR REQUEST: "{user_request}"

        {reflection}

        ───
        CURRENT RULE LIBRARY ({len(existing_bullets)} rules):
        {chr(10).join(existing_bullets)}
    """).strip()

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=2048,
        thinking={"type": "adaptive"},
        system=_LEARNING_SYNTHESIS_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    result = next(
        (block.text for block in response.content if hasattr(block, "text")), ""
    )

    # Parse CHANGES and RULES sections
    changes_text = ""
    rules_text = result
    if "CHANGES" in result and "RULES" in result:
        parts = result.split("RULES", 1)
        changes_text = parts[0].split("CHANGES", 1)[1].strip()
        rules_text = parts[1]
    elif "CHANGES" in result:
        changes_text = result.split("CHANGES", 1)[1].strip()
        rules_text = ""

    new_bullets = [
        line.strip()
        for line in rules_text.splitlines()
        if line.strip().startswith("-")
    ]
    if not new_bullets:
        return False

    # Print before/after diff
    print("\n[RULE LIBRARY UPDATE]")
    print("─" * 60)
    print(f"BEFORE ({len(existing_bullets)} rules):")
    for b in existing_bullets:
        print(f"  {b}")
    print(f"\nAFTER ({len(new_bullets)} rules):")
    for b in new_bullets:
        print(f"  {b}")
    if changes_text and changes_text.lower() != "(no changes)":
        print("\nCHANGES:")
        for line in changes_text.splitlines():
            if line.strip():
                print(f"  {line.strip()}")
    print("─" * 60)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    content = (
        "# Restaurant Agent — Strategy Learnings\n\n"
        f"*Last updated: {timestamp} — {user_request}*\n\n"
        + "\n".join(new_bullets) + "\n"
    )
    tmp = LEARNINGS_FILE.with_suffix(".tmp")
    tmp.write_text(content)
    tmp.replace(LEARNINGS_FILE)

    delta = len(new_bullets) - len(existing_bullets)
    sign = "+" if delta >= 0 else ""
    print(f"\n  [Synthesis complete: {len(existing_bullets)} → {len(new_bullets)} rules ({sign}{delta})]")
    return True


def reflect_and_learn(
    user_request: str,
    queries_used: list[str],
    final_answer: str,
    mode: str = "simple",
    search_log: list[dict] | None = None,
) -> None:
    """
    Reflect on the search strategy used in a completed run and save any
    reusable learnings to learnings.md.

    mode="simple"   — analyses the actual search results to find wasted searches;
                      no extra API searches needed
    mode="advanced" — executes 3 alternative strategies via live searches, compares
                      real results, then extracts evidence-backed learnings
    """
    print(f"\n[Reflecting on strategy... mode={mode}]")

    if mode == "advanced":
        reflection = _run_advanced_reflection(user_request, queries_used, final_answer)
    else:
        reflection = _run_simple_reflection(user_request, queries_used, final_answer, search_log)

    print("\n[REFLECTION]")
    print("-" * 40)
    print(reflection)

    saved = update_learnings_from_reflection(user_request, reflection)
    if saved:
        print(f"\n[Learnings updated in {LEARNINGS_FILE}]")
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

    # Pass rule text + [src: ...] to Haiku so it can reason about source similarity.
    # Strip only the score annotation — keep the src tag visible for matching.
    all_bullets_with_src = [
        _strip_score_only(line.strip())  # removes score, keeps src
        for line in LEARNINGS_FILE.read_text().splitlines()
        if line.strip().startswith("-")
    ]
    if not all_bullets_with_src:
        return ""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": textwrap.dedent(f"""
                Current request: "{user_request}"

                Below are search-strategy rules. Each rule shows which past request(s)
                shaped it in a [src: ...] tag.

                Select only rules applicable to the current request. For each candidate:
                1. Is the rule abstractly relevant (condition matches the request type)?
                2. How similar are the source request(s) to the current request?
                   - CLOSE: same city, same cuisine type, same constraint
                   - RELATED: same constraint type or occasion, different city/cuisine
                   - DISTANT: different domain entirely

                Only include a rule if it is abstractly relevant AND source similarity
                is CLOSE or RELATED. Skip DISTANT rules even if abstractly relevant.

                If none qualify, reply with exactly: none

                Otherwise respond in this exact two-section format:

                WHY
                - rule N: [abstract relevance] | source similarity: CLOSE/RELATED — <one sentence on why>

                RULES
                - verbatim rule text only (no score annotations, no [src: ...])

                Rules:
                {chr(10).join(all_bullets_with_src)}
            """).strip(),
        }],
    )
    result = next(
        (block.text for block in response.content if hasattr(block, "text")), ""
    ).strip()

    if not result or result.lower() == "none":
        print("  [Learning injection: no relevant rules found]")
        return ""

    # Parse WHY and RULES sections
    why_text = ""
    rules_text = result
    if "WHY" in result and "RULES" in result:
        parts = result.split("RULES", 1)
        why_text = parts[0].split("WHY", 1)[1].strip()
        rules_text = parts[1].strip()
    elif "RULES" in result:
        rules_text = result.split("RULES", 1)[1].strip()

    injected_rules = "\n".join(
        line.strip()
        for line in rules_text.splitlines()
        if line.strip().startswith("-")
    )

    if not injected_rules:
        print("  [Learning injection: no relevant rules found]")
        return ""

    rule_count = injected_rules.count("\n") + 1
    print(f"\n[LEARNING INJECTION — {rule_count} rule(s) retrieved]")
    print("─" * 60)
    if why_text:
        print("Why these rules apply:")
        for line in why_text.splitlines():
            if line.strip():
                print(f"  {line.strip()}")
        print("\nInjected rules:")
    for line in injected_rules.splitlines():
        if line.strip():
            print(f"  {line.strip()}")
    print("─" * 60)

    return injected_rules


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
            # Preserve the [src: ...] annotation; only replace the score part
            src_match = _SRC_RE.search(stripped)
            src_tag = src_match.group(0).strip() if src_match else ""
            clean = _strip_score(stripped)
            line = f"{clean}  [u:{u} h:{h}]" + (f"  {src_tag}" if src_tag else "")
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
    answer, queries, search_log = run_agent(request, injected_learnings=injected)
    if not answer:
        exit()

    # Reflect on the strategy and save any new learnings
    reflect_and_learn(request, queries, answer, mode=args.reflect, search_log=search_log)
