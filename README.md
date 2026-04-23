# Self-Improving Tool-Calling Agent

A testbed for one question: **can an LLM agent get better at using tools over time by learning from its own past runs?**

The hypothesis is that after each run the agent reflects on its tool-calling strategy, extracts reusable if/then rules, and injects those rules into future runs. If the hypothesis holds, the learning-mode runs should use fewer tool calls and surface better results than baseline runs on the same requests.

---

## Why a restaurant agent?

Restaurant search is a concrete, observable proxy for any tool-calling task that involves a budget of API/search calls and a quality target (find 3 good candidates matching specific criteria). It avoids the complexity of a real production domain while still having enough variety across price, cuisine, occasion, and location to surface meaningful strategy differences.

---

## How it works

Three components build on each other:

### 1. Agent
Answers restaurant requests using web search (DuckDuckGo), capped at 10 searches. It plans its queries, discards weak results, and stops once it has 3 solid candidates with the details the user asked for.

### 2. Reflection
After every run, a second Claude call analyses the search strategy used, proposes 3 alternatives, and writes any genuinely better rules to `learnings.md` as `if [condition] → then [action]` bullets.

Two modes:
- `simple` — pure reasoning, no extra searches (fast, cheap, default)
- `advanced` — actually executes the 3 alternatives via live searches before comparing

### 3. Learning mode
Before the agent runs, a fast Haiku call reads all rules in `learnings.md` and filters down to the ones applicable to the current request. Those rules are injected into the system prompt so the agent applies past lessons automatically.

---

## Evaluation (`eval_learning.py`)

Runs 50 requests across 5 thematic groups (budget dining, romantic occasions, niche dietary, location-constrained, cheap ethnic cuisine) in three phases:

1. **Seed phase** — run each group's seed request with reflection to populate `learnings.md`
2. **Baseline phase** — run all 25 requests with no learning injection
3. **Learning phase** — run all 25 requests with relevant rules injected

Results are saved to `eval_results.json`. The summary table reports average search count and strategy diversity (how often the agent's search pattern changed) per group, baseline vs. learning mode.

---

## Simplifications

The design deliberately cuts scope to keep the loop tight and the signal clear.

**Single tool, single domain.** One search tool, one domain. Real agents use many tools across shifting domains; here the only variable is search strategy, which makes learning signals easier to isolate.

**Flat markdown as memory.** Past rules are stored as bullet points in a plain text file. No vector database, no embeddings. This makes the store inspectable and editable by hand and removes infrastructure as a failure mode. The cost is no semantic similarity search — relevance filtering is done by a language model call instead.

**Natural-language rules, not weight updates.** Learning is prompt injection, not fine-tuning. Rules are written in plain English and appended to the system prompt. This makes the learning mechanism fully transparent and auditable, but it means the agent's "memory" is only as good as its ability to apply prose instructions.

**Haiku for relevance filtering.** Filtering ~40 rules down to the applicable subset is done with a cheap, fast Haiku call rather than embeddings. This works when rules are short and categorical; it would break down at thousands of rules or with subtle semantic distinctions.

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

# Run the full evaluation
python eval_learning.py

# Dry-run: print the dataset without making API calls
python eval_learning.py --dry-run
```

Requires `ANTHROPIC_API_KEY` to be set in the environment.
