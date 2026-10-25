# Deep Research

AI research agent that iteratively searches, analyzes, and synthesizes comprehensive reports with evidence-backed citations from web and academic sources.

Built on [Z.AI](https://z.ai) GLM and [OpenCode Go](https://opencode.ai/go) models. Fully async. CLI tool.

## What It Does

Give it a question. It breaks it into sub-questions, searches the web and academic databases in parallel, reads and analyzes each source, identifies what's missing, searches again, and produces a structured report with inline citations traced back to specific sources.

```
deep-research "What are the latest advances in LLM reasoning?"
```

```
Query
  --> Decompose into sub-questions
  --> Search (Tavily + DuckDuckGo + arXiv + Semantic Scholar)
  --> Deduplicate results across providers
  --> Extract content (HTML + PDF)
  --> Analyze each source, extract evidence quotes
  --> Identify knowledge gaps
  --> Search again (repeat until sufficient)
  --> Synthesize report with verified citations
```

## Getting Started

### Requirements

- Python 3.9+
- [Z.AI API key](https://z.ai)

That's it. Web search works out of the box with DuckDuckGo (free, no key needed). Add a [Tavily](https://tavily.com) key for higher quality web results.

### Install

**Recommended: pipx** — installs `deep-research` on your global PATH so you can run it from any directory without activating a venv.

```bash
pipx install git+https://github.com/ferhatyurdakul/Deep-Research.git
```

**For development** — editable install in a venv:

```bash
git clone https://github.com/ferhatyurdakul/Deep-Research.git
cd Deep-Research
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Configure

The recommended way is the interactive setup wizard — keys are stored in your OS keychain, no files to edit:

```bash
deep-research setup
```

This walks you through entering your `ZAI_API_KEY` (or `OPENCODE_API_KEY`) and optional Tavily / Semantic Scholar keys, then stores them in:

- **macOS:** Keychain
- **Linux:** Secret Service (GNOME Keyring / KWallet)
- **Windows:** Credential Manager

Other credential management commands:

```bash
deep-research keys                  # show which keys are stored (never prints values)
deep-research login                 # set just the active provider's API key
deep-research login TAVILY_API_KEY  # set a specific key
deep-research logout                # clear the active provider's key
deep-research logout all            # clear every stored credential
```

Inside the REPL, the same commands are available as slash commands: `/setup`, `/login`, `/logout`, `/keys`.

**File-based config still works** — handy for CI, scripted overrides, or air-gapped boxes. Either set environment variables directly (`export ZAI_API_KEY=…`), or put them in `./.env` (per-project, wins when present) or `~/.config/deep-research/.env` (global fallback). The resolution order is: **env var > project `.env` > global `.env` > OS keyring**.

**Diagnostics.** The CLI shows warnings and errors only (retries, failed fetches, rate limits). Set `DEEP_RESEARCH_LOG_LEVEL=INFO` — or `DEBUG` — to also see the per-call route details and result counts. Colors follow `NO_COLOR` and are dropped automatically when output is piped.

### Where files live

| Path | Override env var | Purpose |
|------|------------------|---------|
| `~/.local/share/deep-research/research.db` | `DEEP_RESEARCH_DATA_DIR` | SQLite session store (XDG_DATA_HOME respected) |
| `<cwd>/outputs/` | `DEEP_RESEARCH_OUTPUTS_DIR` | Markdown reports — created lazily where you run from |
| `~/.config/deep-research/.env` | `XDG_CONFIG_HOME` respected | Global credentials fallback |
| `~/.config/deep-research/repl_history` | — | REPL command history |

### Run

```bash
# Interactive REPL (slash commands + persistent settings)
deep-research

# One-shot query
deep-research "your research question"
```

The bare `deep-research` command launches an interactive prompt where you can type questions directly or use slash commands like `/depth deep`, `/template science`, `/agent`, `/history`, `/continue 3`. See **Interactive Mode** below for the full command list.

Everything is discoverable from the terminal: `deep-research --help` lists every option (with what each value means) and every command, `deep-research COMMAND --help` details one command, and `deep-research status` shows the resolved model route, credentials and file locations.

## CLI Reference

```bash
# Interactive prompt (asks for the query)
deep-research

# Direct query
deep-research "your research question"
```

### Flags

```bash
# Depth presets
deep-research --depth quick "topic"             # 1 pass, fast
deep-research --depth standard "topic"          # 2 passes, balanced (default)
deep-research --depth deep "topic"              # 3 passes, thorough, academic sources

# Report style
deep-research --style brief "topic"             # ~500-1000 words, summary + key findings
deep-research --style standard "topic"          # ~2000-4000 words, themed sections (default)
deep-research --style academic "topic"          # ~3000-5000 words, numbered sections + abstract + methodology

# Pipeline features
deep-research --agent "topic"                   # autonomous agent mode (decides when to stop)
deep-research --academic "topic"                # include arXiv + Semantic Scholar
deep-research --template science "topic"        # domain-specific strategy
deep-research --fresh "topic"                   # ignore URLs already analyzed in past sessions

# Combine flags freely
deep-research --agent --template science --style academic --depth deep "topic"

# Session management (subcommands)
deep-research history                           # list past sessions (-n N for more)
deep-research continue 3                        # deepen a previous session
deep-research continue 3 --depth deep           # deepen with a deeper shape
deep-research fork 3                            # copy session 3 into a new session (original untouched)
deep-research fork 3 "narrower angle on X"      # fork with a different query
deep-research resynthesize 3                    # re-run only synthesis on session 3's sources, save as fork
deep-research resynthesize 3 --style academic   # same, but with a different report style
deep-research inspect 3                         # show metadata, sub-questions, sources, iterations, summary
deep-research inspect 3 --full                  # also print the full detailed findings
deep-research inspect 3 --source 5              # deep dive on source #5 (findings + evidence quotes)
deep-research delete 5                          # remove a session (confirms on a terminal)
```

Credential subcommands: `deep-research setup`, `login [KEY]`, `logout [KEY|all]`, `keys`.
System subcommands: `deep-research status` (resolved model route, credentials, file paths), `migrate [PATH]`, `help [COMMAND]`.
`continue` and `resynthesize` accept the same shape flags as one-shot runs (`--depth`, `--academic`, `--template`, `--style`, `--fresh`).

### Getting help

```bash
deep-research --help               # all options (with value meanings), commands, examples
deep-research continue --help      # one command: usage, arguments, options, examples
deep-research help continue        # same, as a command
deep-research --version
```

Help is generated from the same command registry and parsers the CLI runs on, so it can't drift from what is actually accepted. Usage mistakes print the error, the command's usage line and a pointer to `--help` (exit code 2).

### Flag reference

| Flag                     | Values                                        | Default    | Notes                                                    |
|--------------------------|-----------------------------------------------|------------|----------------------------------------------------------|
| `--depth`                | `quick` \| `standard` \| `deep`               | `standard` | Controls sub-questions, results, and iteration count     |
| `--style`                | `brief` \| `standard` \| `academic`           | `standard` | Shape and length of the final report                     |
| `--template`             | `technology` \| `science` \| `market` \| `literature` | —    | Domain-specific prompts and defaults                     |
| `--agent`                | —                                             | off        | Autonomous decision loop instead of fixed iterations     |
| `--academic`             | —                                             | off        | Adds arXiv + Semantic Scholar                            |
| `--fresh`                | —                                             | off        | Disables cross-session URL deduplication                 |

## Search Sources

Up to four providers searched in parallel, deduplicated automatically. Web providers always run; the academic providers (arXiv, Semantic Scholar) are opt-in — enabled with `--academic`, `--depth deep`, or the science/literature templates:

| Provider         | Type     | Key Required | Notes                           |
|-----------------|----------|:------------:|---------------------------------|
| Tavily          | Web      | Optional     | High quality, needs API key     |
| DuckDuckGo      | Web      | No           | Free fallback, always available |
| arXiv           | Academic | No           | Full metadata, always available |
| Semantic Scholar | Academic | No           | API key optional for rate limits|

When no Tavily key is set, DuckDuckGo is the primary web search. Academic providers join with `--academic` / `--depth deep` / the science or literature template.

## Depth Presets

| Depth      | Sub-questions | Results/query | Pages scraped | Iterations | Academic | Extraction |
|------------|:------------:|:-------------:|:-------------:|:----------:|:--------:|:----------:|
| `quick`    | 3            | 3             | 2             | 1          |          |            |
| `standard` | 5            | 5             | 3             | 2          |          |            |
| `deep`     | 7            | 7             | 5             | 3          | yes      | yes        |

Thinking mode is always on — models that don't support the thinking payload are gated off automatically (or reason natively), so no flag is needed.

Pages beyond the scrape budget fall back to search-snippet sources instead of being fetched.

## Research Templates

| Template     | Best for                               | What it enables       |
|--------------|----------------------------------------|-----------------------|
| `technology` | Software, AI, infrastructure           | data extraction       |
| `science`    | Biology, physics, chemistry, medicine  | academic + extraction |
| `market`     | Industries, competitors, market sizing | data extraction       |
| `literature` | Systematic literature review           | academic search       |

Each template provides domain-tuned prompts for decomposition, analysis, and synthesis.

## Agent Mode

With `--agent`, the fixed iteration loop is replaced by an autonomous decision loop. After each pass, the agent evaluates research quality and decides:

- **search_more** -- findings are insufficient
- **go_deeper** -- broad but shallow on a key area
- **broaden** -- missing related perspectives
- **sufficient** -- done

The agent's iteration budget is independent of the fixed-loop count and scales with `--depth` — `quick` 3, `standard` 5, `deep` 8 — capped at a hard safety limit of **8 iterations**. Every depth runs at least one decision cycle, and the agent stops early as soon as it judges coverage sufficient.

## Interactive Mode

Run `deep-research` with no arguments to drop into the REPL:

```
> /depth deep
  Depth: deep
> What are the latest advances in LLM reasoning?
[research runs at depth=deep, report displayed]
  Saved as session 4 -> outputs/research-2026-05-08.md
> /history
  4  2026-05-08  What are the latest advances in LLM reasoning?
  3  2026-05-07  CRISPR base editing 2026
> /continue 3
[continues session 3 with current settings]
> /quit
```

Anything not starting with `/` runs as a research query using the current settings. The input is a Claude-Code-style rounded box whose top border carries the run state — depth, template, non-default style, active toggles (`academic`, `agent`, `fresh`), provider/route — colored and refreshed on every prompt; an empty box shows a hint. Type `/` and a completion menu opens *below* the box with each command's usage and description (Tab/arrows to pick and scroll, keep typing to filter fuzzily); after `/depth ` the menu offers that command's values with what each means. Plain queries never trigger it. History persists in `~/.config/deep-research/repl_history` (↑/↓ to navigate, Ctrl-R to search, first Ctrl-C clears the line).

Seeing your options:

- `/help` lists every command grouped by purpose (Research, Model, Sessions, Credentials, General), with the allowed values under each setting; `/help depth` explains one command.
- A setting command with options is a picker: type `/depth` and press Enter, and an arrow-key list opens (`quick`, `standard`, `deep`, each with what it does, the current one marked). ↑/↓ (or j/k) move, Enter selects, 1-9 jump straight to an item, Esc cancels. The same works for `/style`, `/template`, `/provider` and `/route`. Typing the value yourself (`/depth deep`) skips the picker, and setting a value echoes its meaning.
- **Nothing is left behind.** Choosing or changing a setting prints no confirmation — the status line above the input box already shows the new state, and the picker disappears when you choose. Screens you read (`/help`, `/status`, `/history`, `/inspect`, `/keys`, `/fork`) open in a viewer that vanishes when you press Enter, Esc or `q` (↑/↓ scroll when it's taller than the screen); if a setting command hits a problem (a bad value, an invalid route) that message appears in the same kind of viewer. `/delete <id>` asks with a Yes/No picker. Commands whose output is the point — runs (`/continue`, `/resynthesize`, your questions) and the credential wizards — print normally.
- The status line and banner show academic sources whenever a run will use them — switched on with `/academic`, or implied by the `science` / `literature` templates or `deep` depth — and `/status` says which. Only a submitted research question leaves its input box in the scrollback; settings changes don't stack up boxes.
- A mistyped command suggests the closest match (`/histroy` → `/history`).

| Command | Description |
|---|---|
| `/depth [quick\|standard\|deep]` | Show or set research depth |
| `/style [brief\|standard\|academic]` | Show or set report style |
| `/template [technology\|science\|market\|literature\|none]` | Set domain template |
| `/academic` | Toggle academic sources (arXiv + Semantic Scholar) |
| `/agent` | Toggle autonomous agent loop |
| `/fresh` | Toggle fresh mode (revisit URLs from past sessions) |
| `/provider [zai\|opencode-go]` | Show or switch LLM provider |
| `/route [balanced\|budget\|quality]` | Show or set route profile |
| `/model [stage] [model]` | Pick the model for each pipeline stage (`all`, `decompose`, `analyze`, `gap_analysis`, `synthesize`; `default` returns to the profile) |
| `/setup` | Run the credential setup wizard |
| `/login [KEY]` | Set a single API key in the OS keyring |
| `/logout [KEY\|all]` | Clear stored credentials |
| `/keys` | Show which credentials are stored (never shows values) |
| `/status` | Show current settings and resolved per-stage models |
| `/history [count]` | List recent research sessions |
| `/inspect <id> [--full] [--source N]` | Show a stored session; `--full` adds the detailed findings, `--source N` one source's evidence |
| `/continue <id>` | Deepen a stored session with current settings |
| `/delete <id>` | Delete a stored session (asks for confirmation) |
| `/fork <id> [query]` | Copy a session into a new branch (optionally with a different query) |
| `/resynthesize <id>` | Re-run synthesis on a stored session's sources |
| `/clear` | Clear the screen |
| `/help [command]` | Show the command list, or explain one command |
| `/quit` (`/exit`, `/q`) | Exit the REPL |

One-shot CLI flags still work: `deep-research "topic"`, `deep-research --depth deep "topic"`, `deep-research history`, `deep-research inspect 3`, etc. — see the flag reference above.

## Multi-Model Routing

Each pipeline stage (`decompose`, `analyze`, `gap_analysis`, `synthesize`) can use a different model. The default on Z.AI is **`glm-5.3`**, with the cheaper `glm-4.5-air` on the high-volume stages of the `balanced` and `budget` profiles. There are three ways to change it, from quickest to most permanent:

**1. In the REPL (this session only).** Type `/model` and press Enter: pick a stage (or `all`), then a model from the list — `default` returns a stage to its profile's model. Or type it: `/model analyze glm-5.1`, `/model all glm-5.3`, `/model synthesize default`. `/route` switches the whole preset (`balanced`, `budget`, `quality`), and `/status` shows what each stage resolves to.

**2. A route profile** in `.env` — a named preset that fills all four stages:

```env
LLM_ROUTE=balanced   # balanced | budget | quality
```

**3. Per-stage overrides** in `.env` (these win over the profile; leave a stage empty to use the profile's model):

```env
MODEL_DECOMPOSE=glm-4.5-air      # fast/cheap for decomposition
MODEL_ANALYZE=glm-5.3            # thinking model for analysis
MODEL_SYNTHESIZE=glm-5.3         # best model for synthesis
MODEL_GAP_ANALYSIS=glm-5.3       # thinking model for gap detection
```

`GLM_MODEL` (default `glm-5.3`) is the provider's default model — what `status` reports and what any call without a stage model uses. Registered Z.AI models are `glm-5.3`, `glm-5.1`, `glm-5` and `glm-4.5-air`; a name that isn't registered is accepted with a warning (its thinking and endpoint support are then unverified), and a model registered only for another provider is refused.

## Rate Limits & Concurrency

Deep/agent runs issue a lot of LLM calls in parallel (one analyze + summary chain
per source, plus agent decision/gap calls). On rate-limited providers — notably
Z.AI/GLM `429 Too Many Requests` — that burstiness is what trips the limiter, not
the content. The pipeline controls it in three ways, all tunable via env:

**Concurrency caps** — how many calls run at once:

| Env var | Default | Effect |
|---|:---:|---|
| `MAX_LLM_CONCURRENCY` | `2` | Concurrent analyze/summary/extract LLM calls — default matches the Z.AI GLM coding-plan concurrency cap; raise to `3`–`5` only on providers with headroom |
| `AGENT_MAX_LLM_CONCURRENCY` | `3` | Gentler ceiling auto-applied in `--agent` mode (capped by the above) |
| `MAX_SEARCH_CONCURRENCY` | `5` | Concurrent search + content-extraction HTTP requests |

**Backoff** — what happens on a `429`/timeout. Calls retry with exponential
backoff plus jitter; rate-limit errors wait ~3× longer than generic transient
errors, and each retry logs the provider, endpoint family, model, stage, and
attempt count:

| Env var | Default | Effect |
|---|:---:|---|
| `LLM_MAX_RETRIES` | `5` | Attempts before giving up (then a fallback swap or `ProviderLimitExceeded`) |
| `LLM_RETRY_BASE_SECONDS` | `2.0` | Base backoff unit |
| `LLM_RETRY_MAX_SECONDS` | `60.0` | Per-attempt wait cap |

**Run-shape budgets** — how much work a run does at all. Unset = use the
`--depth` preset:

| Env var | Effect |
|---|---|
| `MAX_SUB_QUESTIONS` / `MAX_SEARCH_RESULTS` / `MAX_ITERATIONS` / `MAX_SOURCES` | Override the depth preset |
| `AGENT_BUDGET_QUICK` / `_STANDARD` / `_DEEP` | Override agent iteration budgets (capped at 8) |

Set `LLM_FALLBACK_PROVIDER` (e.g. `zai` when primary is `opencode-go`, or vice
versa) so a run that exhausts retries on one provider swaps to the other instead
of aborting.

Failures degrade instead of discarding work. A source whose analysis fails is
skipped while the rest continue; an iteration that fails after sources were
already gathered stops the loop and synthesizes from what exists — the session
is saved and `--continue-session <id>` resumes it. A failed synthesis still
saves the collected sources — re-run with `--resynthesize <id>` to retry only
the synthesis step without repeating search/extraction. Only a failure that
leaves *zero* sources (or every analysis in a batch failing) aborts the run.

### Recommended settings: research vs coding workloads

Research is bursty and long-lived; coding/agent IDE use is interactive and
latency-sensitive. They want different knobs:

| | Research (this tool) | Coding (IDE/agent) |
|---|---|---|
| Concurrency | `MAX_LLM_CONCURRENCY=2–5`, `AGENT_MAX_LLM_CONCURRENCY=2–3` | N/A (one call at a time) |
| Retries | `LLM_MAX_RETRIES=5`, longer `LLM_RETRY_MAX_SECONDS` (60s) — OK to wait | Fewer retries, short waits — fail fast for the human |
| Depth/agent | `--depth deep`/`--agent` are heavy; pair with a fallback provider | — |
| Thinking | On for analysis/synthesis quality | Off for speed unless needed |

If you're hitting GLM 429s during ad-hoc deep/agent research, the default
(`MAX_LLM_CONCURRENCY=2`) is already the safe setting — make sure you haven't
raised it, and set `LLM_FALLBACK_PROVIDER`. If you run research alongside heavy
coding traffic on the same key, the research run is almost certainly the burst
source — cap it there rather than throttling your editor.

## Project Structure

```
src/deep_research/
  main.py                  CLI entrypoint + command dispatch
  cli_help.py              Command registry, rich --help, styled usage errors
  options.py               Option values + what they mean (shared by help and REPL)
  ui.py                    Shared theme, console, glyphs, panels
  views.py                 Shared screens: banner, history, inspect, status, saved card
  repl.py                  Interactive REPL
  config.py                Settings from .env
  credentials.py           OS-keyring credential access
  setup_wizard.py          setup / login / logout / keys flows
  logging_config.py        Logging setup
  llm.py                   LLM client (sync + async, multi-provider)
  capabilities.py          Per-model capability registry + recommended routes
  usage.py                 Per-run call tracker + fallback metadata
  models.py                Pydantic data models
  urls.py                  Shared URL normalization for dedup
  search/                  Multi-provider search
    base.py                  SearchProvider ABC
    web_search.py            Tavily + DuckDuckGo
    arxiv_search.py          arXiv API
    semantic_scholar.py      Semantic Scholar API
    aggregator.py            Parallel search + dedup
  extraction/              Content extraction
    fetcher.py               Async HTTP fetcher
    parser.py                HTML/PDF parser
    cleaner.py               Text cleaning
  pipeline/                Research orchestration
    orchestrator.py          Main pipeline
    decomposer.py            Query decomposition
    analyzer.py              Source analysis + gap detection
    synthesizer.py           Report synthesis
    agent.py                 Autonomous agent loop
    templates.py             Research templates
  output/                  Report formatting
    report.py                Markdown + terminal output
    citations.py             Citation formatting + validation
  storage/
    db.py                    SQLite persistence
tests/                     Offline + e2e test suite

evals/
  benchmarks.py            Benchmark query set, tagged by subset
  checks.py                Mechanical checks (citations, tone, structure)
  run.py                   Benchmark runner
  compare.py               Diff two runs, flag regressions
```

## Running Tests

Test dependencies (`pytest`, `pytest-asyncio`) are declared as a `dev` extra. If you installed with `pip install -e .` (no extras), add them with:

```bash
pip install -e ".[dev]"
```

Then:

```bash
python -m pytest tests/test_e2e.py -v
```

Tests hit live APIs (Z.AI, Tavily, arXiv, Semantic Scholar). Tests requiring API keys are skipped automatically when keys are not set. Non-API tests (models, citations, DB, templates) always run.

## Evaluation

A lightweight benchmark harness lives in `evals/`. It runs a fixed set of research prompts, saves reports and mechanical checks per run, and diffs runs to catch regressions when prompts or retrieval change.

```bash
# Before a change: baseline
python -m evals.run --tag smoke --label baseline

# After a change: candidate
python -m evals.run --tag smoke --label post-tone-fix

# Diff them — exits non-zero if regressions are detected
python -m evals.compare evals/runs/<baseline-dir> evals/runs/<candidate-dir>
```

Runs are written to `evals/runs/<timestamp>-<git-sha>[-label]/` with one directory per benchmark (report, raw JSON, checks). Benchmarks are defined in `evals/benchmarks.py` and tagged (`smoke` for fast iteration, `full` for the complete set, `contested` for topics with actively disputed answers).

Mechanical checks include:

- Citation density (per 100 words) and coverage (% paragraphs with `[N]`)
- Invalid citations (refs beyond `source_count`) and orphan sources (never cited)
- Snippet-only source ratio
- Banned superlative/marketing phrases (regression signal for tone rules)
- Required section presence for the report style

All checks are deterministic and run locally — no extra LLM calls.

## Contributing

1. Fork the repo
2. Create a feature branch
3. Make your changes
4. Run `python -m pytest tests/test_e2e.py -v` to verify
5. Open a PR

## License

MIT
