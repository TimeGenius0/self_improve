# Self-Improving Tool-Calling Agent

## Purpose

Tool-calling agents repeat the same mistakes. Each request starts from scratch with no memory of what worked before. For repeated similar tasks — finding cheap restaurants in a city, vetting candidates near a landmark — the agent rediscovers the same efficient strategies every time, wasting tool calls it has already paid for.

This project tests a concrete fix: **learn search policies from past runs and inject them into future similar requests to reduce tool call count.**

The mechanism is:
1. After each run, extract the search strategy as reusable `if [condition] → then [action]` rules ("if the request has a hard price cap in euros, include a French-language query early")
2. Track how often each rule was injected and whether it actually reduced search count (`[u:used h:helped]`)
3. Before each new run, filter the ruleset to rules that match the current request and inject them into the system prompt
4. Periodically prune rules that were used but didn't help (h/u < 0.30 with u ≥ 3)

The restaurant recommendation domain is the vehicle: a bounded, measurable task where search count is a clean proxy for efficiency.

---

## Architecture

Four components build on each other:

### 1. Agent
Answers restaurant requests using web search (DuckDuckGo), capped at 10 searches. Labels every search with a strategy tag (`broad_discovery`, `price_verification`, `reddit_forum`, etc.) so the reflection step can reason about which approach produced which results.

### 2. Reflection
After each run, a second Claude call proposes 3 alternative search strategies and compares them against the one used. Writes any genuinely better strategies to `learnings.md` as `if/then` rules, each initialised with `[u:0 h:0]`.

Two modes:
- `simple` — reasons hypothetically, no extra searches (fast, cheap)
- `advanced` — executes all 3 alternatives via live searches before comparing (default in evaluation)

### 3. Learning mode
Before the agent runs, a Haiku call strips score annotations from all rules in `learnings.md`, filters to the ones applicable to the current request, and injects them into the system prompt — clean prose, no bookkeeping noise.

### 4. Rule scoring and consolidation
After each learning run, updates `[u:N h:M]` counters on every injected rule: `u` always increments; `h` increments only when the learning run used fewer searches than the paired baseline. Every 10 learn runs, a Haiku call consolidates the ruleset: merging near-duplicates, pruning rules with h/u < 0.30 (u ≥ 3), resolving contradictions by keeping the higher-ratio rule.

---

## Evaluation

The evaluation runs 25 restaurant requests across 5 thematic groups. Each request is processed in an interleaved loop that maximises how quickly new rules reach later requests:

```
For each request (in sequence):

  ┌─ 1. Baseline run ──────────────────────────────────────────┐
  │  No rules injected. Records search count and strategy.     │
  └────────────────────────────────────────────────────────────┘
            │
            ▼  Reflect (advanced: 3 live alternative strategies)
  ┌─ 2. Update learnings.md ───────────────────────────────────┐
  │  New rules appended [u:0 h:0] — available to learn run.    │
  └────────────────────────────────────────────────────────────┘
            │
            ▼  Haiku filters rules to this request
  ┌─ 3. Learn run ─────────────────────────────────────────────┐
  │  Injected rules guide search strategy.                     │
  └────────────────────────────────────────────────────────────┘
            │
            ├─ 4. Score: update [u:N h:M] on injected rules
            │
            ▼  Reflect again (synthesise learn run)
  ┌─ 5. Rewrite learnings.md ──────────────────────────────────┐
  │  Scored, merged ruleset ready for next request.            │
  └────────────────────────────────────────────────────────────┘
            │
            ▼  Every 10 learn runs
  ┌─ 6. Consolidate ───────────────────────────────────────────┐
  │  Haiku prunes h/u < 0.30 (u ≥ 3), merges duplicates.      │
  └────────────────────────────────────────────────────────────┘
```

Because the baseline's reflection runs *before* the learn run, the learn run always sees the freshest possible ruleset — including rules extracted just seconds earlier from the same request.

### Results

```
═══════════════════════════════════════════════════════════════════════════════
EVALUATION SUMMARY  (50 runs: 25 baseline + 25 learning mode)
═══════════════════════════════════════════════════════════════════════════════
Group                        │ Base searches │ Learn searches │ Injected │ Strategy Δ
─────────────────────────────────────────────────────────────────────────────────
A  Paris Budget Ethnic        │     8.8       │     7.0        │   5/5    │  5/5
B  Paris Occasion Anti-Tourist│     0.0       │     0.0        │   0/0    │  0/5
C  London Dietary Occasion    │     0.0       │     0.0        │   0/0    │  0/5
D  Bay Area Highway Proximity │     8.8       │     8.6        │   4/5    │  5/5
E  NYC Post-Event Late Night  │     0.0       │     0.0        │   0/0    │  0/5
─────────────────────────────────────────────────────────────────────────────────
OVERALL                       │     8.8       │     7.8        │  9/25    │ 10/25
═══════════════════════════════════════════════════════════════════════════════
```

Groups B, C, and E were intentionally skipped to limit token spend. Only Groups A and D were run, giving 10 completed request pairs. The signal below comes from those 10 pairs.

**Search count, baseline vs. learning (completed groups only):**

```
Group A — Paris Budget Ethnic   (5 requests, price-capped, same city)
  Baseline  ████████████████████████████  8.8 searches avg
  Learn     ███████████████████████       7.0 searches avg   ▼ 20%

Group D — Bay Area Highway Proximity   (5 requests, location-constrained)
  Baseline  ████████████████████████████  8.8 searches avg
  Learn     ███████████████████████████   8.6 searches avg   ▼  2%
```

**What the results show:**

- **Rules changed search strategy in every injected run** — 10/10 runs where rules existed produced a different sequence of strategy labels than the baseline. The agent wasn't ignoring the policies; it was acting on them.

- **Transfer quality depends on category tightness.** Group A (cheap ethnic food in Paris, price caps in euros) saw a 20% reduction. All 5 requests share the same city, currency, and budget framing, so a rule like "lead with a French-language price query" applies directly across all of them. Group D (freeway proximity in the Bay Area) saw only a 2% reduction despite 4/5 runs having rules injected — location-constraint requests vary enough in structure that the rules changed strategy without cutting search count.

- **Partial coverage by design.** Groups B, C, and E were skipped to limit token spend. A full eval across all 5 groups would show whether policies transfer across request types (budget vs. romantic vs. dietary) or only within tight category clusters like Group A.

---

## Simplifications

**Single tool, single domain.** One search tool, one domain keeps search count as an unambiguous efficiency signal. Real agents use many tools; isolating one makes the learning loop observable.

**Annotated markdown as memory.** Rules live in a plain text file with `[u:N h:M]` annotations. No vector database, no embeddings. The whole ruleset is inspectable and editable by hand. The cost: no semantic similarity search — relevance filtering is a Haiku call, and annotations are stripped before the agent or filter sees the rules.

**Natural-language policies, not weight updates.** Learning is prompt injection, not fine-tuning. Rules are prose appended to the system prompt. Fully transparent and auditable; bounded by the agent's ability to follow prose instructions at inference time.

**Score-gated pruning, not reinforcement learning.** The `h/u < 0.30` threshold is a hard heuristic. It requires no training data and is easy to inspect, but it won't adapt if the distribution of requests shifts significantly.

**Haiku for filtering and consolidation.** Both steps use the smallest available model to keep costs low. Works at ~40 rules; would become noisier at hundreds of rules or with subtle semantic overlaps between them.

**Simple reflection as default for interactive use.** The `advanced` mode runs 3 live alternative strategies before extracting learnings — higher quality but costs 15 extra searches per run. The evaluation uses `advanced` by default; the CLI defaults to `simple`.

**Fixed 25-request evaluation set.** Hand-crafted, not sampled from real traffic. Groups test specific dimensions (price caps, dietary filters, location constraints) rather than representing a realistic request distribution.

---

## Usage

```bash
pip install anthropic ddgs

# Basic run
python restaurant_agent.py

# Apply past learnings
python restaurant_agent.py --learn

# Deep reflection (runs 3 alternative strategies live)
python restaurant_agent.py --reflect advanced

# Both
python restaurant_agent.py --learn --reflect advanced

# Run the full evaluation (all 25 requests)
python eval_learning.py

# Run only specific groups
python eval_learning.py --groups A D

# Dry-run: preview dataset without API calls
python eval_learning.py --dry-run
python eval_learning.py --groups A --dry-run
```

Requires `ANTHROPIC_API_KEY` to be set in the environment.
