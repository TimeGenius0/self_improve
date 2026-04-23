# Self-Improving Tool-Calling Agent

A testbed for one question: **can an LLM agent get better at using tools over time by learning from its own past runs?**

The hypothesis is that after each run the agent reflects on its tool-calling strategy, extracts reusable if/then rules, injects those rules into future runs, scores whether they actually helped, and prunes the ones that didn't. If the hypothesis holds, later runs in the evaluation should use fewer tool calls than earlier ones as the ruleset matures.

---

## Why a restaurant agent?

Restaurant search is a concrete, observable proxy for any tool-calling task that involves a budget of API/search calls and a quality target (find 3 good candidates matching specific criteria). It avoids the complexity of a real production domain while still having enough variety across price, cuisine, occasion, and location to surface meaningful strategy differences.

---

## How it works

Four components build on each other:

### 1. Agent
Answers restaurant requests using web search (DuckDuckGo), capped at 10 searches. It plans its queries, discards weak results, and stops once it has 3 solid candidates with the details the user asked for.

### 2. Reflection
After every run, a second Claude call analyses the search strategy used, proposes 3 alternatives, and writes any genuinely better rules to `learnings.md` as `if [condition] → then [action]` bullets. Each new rule is initialised with a score annotation `[u:0 h:0]` (used: 0, helped: 0).

Two modes:
- `simple` — pure reasoning, no extra searches (fast, cheap, default)
- `advanced` — actually executes the 3 alternatives via live searches before comparing

### 3. Learning mode
Before the agent runs, a fast Haiku call reads all rules in `learnings.md`, strips their score annotations, filters down to the ones applicable to the current request, and injects them into the agent's system prompt.

### 4. Rule scoring and consolidation
After each learning-mode run, the `[u:N h:M]` counters on every injected rule are updated: `u` (used) always increments; `h` (helped) increments only when the learning run used fewer searches than the paired baseline. Every `CONSOLIDATION_INTERVAL` (default: 10) learn runs, a Haiku call consolidates the ruleset: merging near-duplicates, removing rules that have been used 3+ times but helped fewer than 30% of the time, and keeping contradicting rules only when one clearly outperforms the other.

---

## Evaluation (`eval_learning.py`)

Runs 50 total runs (25 baseline + 25 learning) across 5 thematic groups (budget dining, romantic occasions, niche dietary, location-constrained, cheap ethnic cuisine) using an **interleaved loop** rather than separate phases.

For each of the 25 requests, in order:

1. **Baseline run** — no learning injection, no reflection
2. **Learn run** — inject rules accumulated from all prior requests
3. **Reflect** — reflect on the learn run, append new rules to `learnings.md`
4. **Score** — update rule scores using the baseline vs. learn search delta
5. **Consolidate** — every 10 learn runs, prune and merge the ruleset

This interleaved design means later requests benefit from a progressively richer and higher-quality ruleset than earlier ones — which is what the learning trend section of the summary measures.

Results are saved to `eval_results.json` **incrementally after each request pair**, so a crashed run can be resumed. The summary table reports average search count and strategy diversity per group, plus a learning-trend breakdown (early / mid / late thirds) to show whether the ruleset compounds over time.

At the start of each evaluation run, the existing `learnings.md` is backed up to `learnings_pre_eval.md` and reset, so all rules learned during the eval come from the eval itself.

---

## Simplifications

The design deliberately cuts scope to keep the loop tight and the signal clear.

**Single tool, single domain.** One search tool, one domain. Real agents use many tools across shifting domains; here the only variable is search strategy, which makes learning signals easier to isolate.

**Annotated markdown as memory.** Past rules are stored as bullet points in a plain text file, each with a `[u:N h:M]` score annotation. No vector database, no embeddings — the store stays inspectable and editable by hand. The cost is no semantic similarity search; relevance filtering is done by a language model call, and score annotations are stripped before rules are shown to the agent or the relevance filter.

**Natural-language rules, not weight updates.** Learning is prompt injection, not fine-tuning. Rules are written in plain English and appended to the system prompt. This makes the learning mechanism fully transparent and auditable, but it means the agent's "memory" is only as good as its ability to apply prose instructions.

**Haiku for relevance filtering and consolidation.** Both filtering applicable rules and consolidating the ruleset use cheap, fast Haiku calls rather than embeddings or a separate classifier. This works at the current scale (~40 rules); it would become noisier with hundreds of rules or subtle semantic distinctions between them.

**Score-gated pruning, not reinforcement learning.** The `h/u < 0.30 with u ≥ 3` threshold is a hard heuristic, not a learned policy. It's transparent and requires no training data, but it won't adapt if the distribution of requests changes significantly.

**Simple reflection as default.** The `simple` reflection mode reasons hypothetically about alternative strategies without running them. It is fast and cheap but can propose improvements that don't actually hold up against real search results. The `advanced` mode corrects this but costs 15 extra searches per run.

**Fixed 25-request evaluation set.** The evaluation dataset is hand-crafted, not sampled from real traffic. The groups are chosen to test specific strategy dimensions (price-capped queries, niche dietary filters, location constraints) rather than to be statistically representative.

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
python eval_learning.py --groups A C E

# Dry-run: preview the dataset without making API calls
python eval_learning.py --dry-run
python eval_learning.py --groups A --dry-run
```

Requires `ANTHROPIC_API_KEY` to be set in the environment.
