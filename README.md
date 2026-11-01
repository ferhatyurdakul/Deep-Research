# Synapse Deep Research

AI research agent that iteratively searches, analyzes, and synthesizes comprehensive reports with evidence-backed citations from web sources and research papers.

Built on [Z.AI](https://z.ai) GLM and [OpenCode Go](https://opencode.ai/go) models. Fully async. CLI tool.

## What It Does

Give it a question. It breaks it into sub-questions, searches the web and paper databases (arXiv, Semantic Scholar) in parallel, reads and analyzes each source, identifies what's missing, searches again, and produces a structured report with inline citations traced back to specific sources.

```
synapse-deep-research "What are the latest advances in LLM reasoning?"
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

**Recommended: pipx** — installs `synapse-deep-research` on your global PATH so you can run it from any directory without activating a venv.

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

**Coming from `deep-research`?** The tool was renamed. Install the new one and remove the old:

```bash
pipx uninstall deep-research
pipx install git+https://github.com/ferhatyurdakul/Deep-Research.git
```

The first launch copies your settings, session history and stored API keys from the old install, and leaves the old copies where they were (macOS may ask once to let the new install read the old Keychain entries). Scripts need the new names: the `deep-research` command is now `synapse-deep-research`, and `DEEP_RESEARCH_*` variables are now `SYNAPSE_DEEP_RESEARCH_*`.

### Configure

The recommended way is the setup screen. It opens by itself the first time you launch `synapse-deep-research` in a terminal with no API key, and any time with `synapse-deep-research setup` (or `/setup` inside the app):

```bash
synapse-deep-research setup
```

It walks you through five steps — each one a small picker or prompt that disappears as you go:

1. **Provider** — Z.AI or OpenCode Go.
2. **API key** — pasted hidden and stored in your OS keychain, never in a file.
3. **Tavily search key** — optional; better web search than DuckDuckGo alone.
4. **Models** — accept the recommended models, or pick one for each of the four stages (see **Multi-Model Routing** below). If your provider lists the models your account can use, that list is shown too.
5. **Connection check** — optional; sends one tiny request per chosen model so a wrong key or a model your plan can't use shows up now.

Esc skips setup at any step (the app then starts normally and tells you how to come back). It never opens outside a terminal, so scripts and pipes are never blocked. Keys are stored in:

- **macOS:** Keychain
- **Linux:** Secret Service (GNOME Keyring / KWallet)
- **Windows:** Credential Manager

Other credential management commands:

```bash
synapse-deep-research keys                  # show which keys are stored (never prints values)
synapse-deep-research login                 # set just the active provider's API key
synapse-deep-research login TAVILY_API_KEY  # set a specific key
synapse-deep-research logout                # clear the active provider's key
synapse-deep-research logout all            # clear every stored credential
```

Inside the REPL, the same commands are available as slash commands: `/setup`, `/login`, `/logout`, `/keys`.

**File-based config still works** — handy for CI, scripted overrides, or air-gapped boxes. Either set environment variables directly (`export ZAI_API_KEY=…`), or put them in `./.env` (per-project, wins when present) or `~/.config/synapse-deep-research/.env` (global fallback). The resolution order is: **env var > project `.env` > global `.env` > OS keyring**.

**Diagnostics.** The CLI shows warnings and errors only (retries, failed fetches, rate limits). Set `SYNAPSE_DEEP_RESEARCH_LOG_LEVEL=INFO` — or `DEBUG` — to also see the per-call route details and result counts. Colors follow `NO_COLOR` and are dropped automatically when output is piped.

### Where files live

| Path | Override env var | Purpose |
|------|------------------|---------|
| `~/.local/share/synapse-deep-research/research.db` | `SYNAPSE_DEEP_RESEARCH_DATA_DIR` | SQLite session store (XDG_DATA_HOME respected) |
| `<cwd>/outputs/` | `SYNAPSE_DEEP_RESEARCH_OUTPUTS_DIR` | Markdown reports — created lazily where you run from |
| `~/.config/synapse-deep-research/config.json` | `SYNAPSE_DEEP_RESEARCH_CONFIG_FILE` | Your provider and the model for each stage (no secrets) |
| `~/.config/synapse-deep-research/.env` | `XDG_CONFIG_HOME` respected | Global credentials fallback |
| `~/.config/synapse-deep-research/repl_history` | — | REPL command history |

### Run

```bash
# Interactive REPL (slash commands + persistent settings)
synapse-deep-research

# One-shot query
synapse-deep-research "your research question"
```

The bare `synapse-deep-research` command launches an interactive prompt where you can type questions directly or use slash commands like `/depth deep`, `/template science`, `/agent`, `/history`, `/continue 3`. See **Interactive Mode** below for the full command list.

Everything is discoverable from the terminal: `synapse-deep-research --help` lists every option (with what each value means) and every command, `synapse-deep-research COMMAND --help` details one command, and `synapse-deep-research status` shows the provider, the model each stage uses, credentials and file locations.

## CLI Reference

```bash
# Interactive prompt (asks for the query)
synapse-deep-research

# Direct query
synapse-deep-research "your research question"
```

### Flags

```bash
# Depth presets
synapse-deep-research --depth quick "topic"             # 1 pass, fast
synapse-deep-research --depth standard "topic"          # 2 passes, balanced (default)
synapse-deep-research --depth deep "topic"              # 3 passes, thorough, research papers

# Report style
synapse-deep-research --style brief "topic"             # ~500-1000 words, summary + key findings
synapse-deep-research --style standard "topic"          # ~2000-4000 words, themed sections (default)
synapse-deep-research --style academic "topic"          # ~3000-5000 words, numbered sections + abstract + methodology

# Pipeline features
synapse-deep-research --agent "topic"                   # autonomous agent mode (decides when to stop)
synapse-deep-research --papers "topic"                  # always include arXiv + Semantic Scholar
synapse-deep-research --depth deep --no-papers "topic"  # web only, even on deep
synapse-deep-research --template science "topic"        # domain-specific strategy
synapse-deep-research --fresh "topic"                   # ignore URLs already analyzed in past sessions

# Combine flags freely
synapse-deep-research --agent --template science --style academic --depth deep "topic"

# Session management (subcommands)
synapse-deep-research history                           # list past sessions (-n N for more)
synapse-deep-research continue 3                        # deepen a previous session
synapse-deep-research continue 3 --depth deep           # deepen with a deeper shape
synapse-deep-research fork 3                            # copy session 3 into a new session (original untouched)
synapse-deep-research fork 3 "narrower angle on X"      # fork with a different query
synapse-deep-research resynthesize 3                    # re-run only synthesis on session 3's sources, save as fork
synapse-deep-research resynthesize 3 --style academic   # same, but with a different report style
synapse-deep-research inspect 3                         # show metadata, sub-questions, sources, iterations, summary
synapse-deep-research inspect 3 --full                  # also print the full detailed findings
synapse-deep-research inspect 3 --source 5              # deep dive on source #5 (findings + evidence quotes)
synapse-deep-research delete 5                          # remove a session (confirms on a terminal)
```

Credential subcommands: `synapse-deep-research setup`, `login [KEY]`, `logout [KEY|all]`, `keys`.
System subcommands: `synapse-deep-research status` (provider, the model for each stage and where it comes from, credentials, file paths), `migrate [PATH]`, `help [COMMAND]`.
`continue` and `resynthesize` accept the same shape flags as one-shot runs (`--depth`, `--papers`, `--template`, `--style`, `--fresh`).

### Getting help

```bash
synapse-deep-research --help               # all options (with value meanings), commands, examples
synapse-deep-research continue --help      # one command: usage, arguments, options, examples
synapse-deep-research help continue        # same, as a command
synapse-deep-research --version
```

Help is generated from the same command registry and parsers the CLI runs on, so it can't drift from what is actually accepted. Usage mistakes print the error, the command's usage line and a pointer to `--help` (exit code 2).

### Flag reference

| Flag                     | Values                                        | Default    | Notes                                                    |
|--------------------------|-----------------------------------------------|------------|----------------------------------------------------------|
| `--depth`                | `quick` \| `standard` \| `deep`               | `standard` | Controls sub-questions, results, and iteration count     |
| `--style`                | `brief` \| `standard` \| `academic`           | `standard` | Shape and length of the final report                     |
| `--template`             | `technology` \| `science` \| `market` \| `literature` | —    | Domain-specific prompts and defaults                     |
| `--agent`                | —                                             | off        | Autonomous decision loop instead of fixed iterations     |
| `--papers`               | —                                             | auto       | Always adds research papers (arXiv + Semantic Scholar)   |
| `--no-papers`            | —                                             | auto       | Web only, even for `deep` or the science / literature templates |
| `--fresh`                | —                                             | off        | Disables cross-session URL deduplication                 |

## Search Sources

Up to four providers searched in parallel, deduplicated automatically. Web providers always run; the paper providers (arXiv, Semantic Scholar) follow the **papers** setting. It is `auto` by default — on for `--depth deep` and the science / literature templates, off otherwise — and `--papers` / `--no-papers` (`/papers on|off|auto` in the REPL) force it either way. (Not to be confused with `--style academic`, which is only about how the report reads.):

| Provider         | Type     | Key Required | Notes                           |
|-----------------|----------|:------------:|---------------------------------|
| Tavily          | Web      | Optional     | High quality, needs API key     |
| DuckDuckGo      | Web      | No           | Free fallback, always available |
| arXiv           | Papers   | No           | Full metadata, always available |
| Semantic Scholar | Papers   | No           | API key optional for rate limits|

When no Tavily key is set, DuckDuckGo is the primary web search. The paper providers join when the papers setting is on, or on `auto` with `--depth deep` or the science / literature template.

## Depth Presets

| Depth      | Sub-questions | Results/query | Pages scraped | Iterations | Papers   | Extraction |
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
| `science`    | Biology, physics, chemistry, medicine  | papers + extraction   |
| `market`     | Industries, competitors, market sizing | data extraction       |
| `literature` | Systematic literature review           | paper search          |

Each template provides domain-tuned prompts for decomposition, analysis, and synthesis.

## Agent Mode

With `--agent`, the fixed iteration loop is replaced by an autonomous decision loop. After each pass, the agent evaluates research quality and decides:

- **search_more** -- findings are insufficient
- **go_deeper** -- broad but shallow on a key area
- **broaden** -- missing related perspectives
- **sufficient** -- done

The agent's iteration budget is independent of the fixed-loop count and scales with `--depth` — `quick` 3, `standard` 5, `deep` 8 — capped at a hard safety limit of **8 iterations**. Every depth runs at least one decision cycle, and the agent stops early as soon as it judges coverage sufficient.

## Interactive Mode

Run `synapse-deep-research` with no arguments to drop into the REPL:

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

Anything not starting with `/` runs as a research query using the current settings. The input is a Claude-Code-style rounded box whose top border carries the run state — depth, template, non-default style, active toggles (`papers`, `agent`, `fresh`), provider — colored and refreshed on every prompt; an empty box shows a hint. Type `/` and a completion menu opens *below* the box with each command's usage and description (Tab/arrows to pick and scroll, keep typing to filter fuzzily); after `/depth ` the menu offers that command's values with what each means. Plain queries never trigger it. History persists in `~/.config/synapse-deep-research/repl_history` (↑/↓ to navigate, Ctrl-R to search, first Ctrl-C clears the line).

Seeing your options:

- `/help` lists every command grouped by purpose (Research, Model, Sessions, Credentials, General), with the allowed values under each setting; `/help depth` explains one command.
- A setting command with options is a picker: type `/depth` and press Enter, and an arrow-key list opens (`quick`, `standard`, `deep`, each with what it does, the current one marked). ↑/↓ (or j/k) move, Enter selects, 1-9 jump straight to an item, Esc cancels. The same works for `/style`, `/template`, `/papers` and `/provider`; `/model` opens a stage picker, then a model picker. Typing the value yourself (`/depth deep`) skips the picker, and setting a value echoes its meaning.
- **Nothing is left behind.** Choosing or changing a setting prints no confirmation — the status line above the input box already shows the new state, and the picker disappears when you choose. Screens you read (`/help`, `/status`, `/history`, `/inspect`, `/keys`, `/fork`) open in a viewer that vanishes when you press Enter, Esc or `q` (↑/↓ scroll when it's taller than the screen); if a setting command hits a problem (a bad value, an invalid model) that message appears in the same kind of viewer. `/delete <id>` asks with a Yes/No picker. Commands whose output is the point — runs (`/continue`, `/resynthesize`, your questions) and the credential wizards — print normally.
- The status line and banner show `papers` whenever a run will search research papers — set on with `/papers on`, or implied on `auto` by the `science` / `literature` templates or `deep` depth — and `/status` says which (`auto — on (science template)`). `/papers off` keeps a run web-only whatever the template or depth. Only a submitted research question leaves its input box in the scrollback; settings changes don't stack up boxes.
- A mistyped command suggests the closest match (`/histroy` → `/history`).

| Command | Description |
|---|---|
| `/depth [quick\|standard\|deep]` | Show or set research depth |
| `/style [brief\|standard\|academic]` | Show or set report style |
| `/template [technology\|science\|market\|literature\|none]` | Set domain template |
| `/papers [auto\|on\|off]` | Research papers (arXiv + Semantic Scholar): `auto` follows the template and depth, `on` always, `off` never |
| `/agent` | Toggle autonomous agent loop |
| `/fresh` | Toggle fresh mode (revisit URLs from past sessions) |
| `/provider [zai\|opencode-go]` | Show or switch LLM provider |
| `/model [stage] [model]` | Choose and save the model for each pipeline stage (`all`, `decompose`, `analyze`, `gap_analysis`, `synthesize`; `default` returns to the recommendation) |
| `/setup` | Open the setup screen (provider, API key, models) |
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

One-shot CLI flags still work: `synapse-deep-research "topic"`, `synapse-deep-research --depth deep "topic"`, `synapse-deep-research history`, `synapse-deep-research inspect 3`, etc. — see the flag reference above.

## Multi-Model Routing

A run has four stages that call the model, and each can use a different one:

| Stage | What it does | How often |
|---|---|---|
| `decompose` | splits your question into sub-questions | once |
| `analyze` | reads one source and extracts findings and quotes | once per source — most of the calls |
| `gap_analysis` | decides what is still missing, or when to stop | once per pass |
| `synthesize` | writes the final cited report | once |

A cheap, fast model on the easy or high-volume stages and a strong one where quality is decided gives nearly the same report faster and cheaper, and is gentler on rate limits.

**You choose the model for each stage**, in the setup screen or any time with `/model` in the app (the choice is saved, so it sticks):

- `/model` + Enter — pick a stage (or `all`), then a model. The first list says what each stage does and which model it uses now; the second lists the provider's models with what is known about each, the recommended one tagged. `other` lets you type any model ID, and `default` returns a stage to the recommendation.
- Or type it: `/model analyze glm-5.3-flash`, `/model all glm-5.3`, `/model synthesize default`.
- `/status` (or `synapse-deep-research status`) shows what each stage uses and whether that is **saved**, **recommended** (you never chose), or **from your environment**.

**Recommended models**, used until you choose (and pre-selected in setup):

| Provider | decompose | analyze | gap_analysis | synthesize |
|---|---|---|---|---|
| Z.AI | `glm-5.3-flash` | `glm-5.3-flash` | `glm-5.3` | `glm-5.3` |
| OpenCode Go | `glm-5.1` | `glm-5.1` | `glm-5.1` | `glm-5.1` |

**Overriding from the environment** (CI, scripts): `MODEL_DECOMPOSE`, `MODEL_ANALYZE`, `MODEL_GAP_ANALYSIS`, `MODEL_SYNTHESIZE` win over everything saved. `/model` and setup tell you when one of them is shadowing your choice. `LLM_ROUTE` (the old preset system) is gone; the app warns if it is still set.

**Z.AI models** (checked against [Z.AI's docs](https://docs.z.ai/guides/overview/pricing)): `glm-5.3` (flagship, 1M context), `glm-5.3-flash` (low-cost tier; on the Coding Plan with 3× the quota of `glm-5.3`), `glm-5.3-flashx` (speed tier; it appears in the Coding Plan's model list but calls are rejected with error 1311, so it needs a plan that includes it), and the previous generations `glm-5.2` and `glm-5.1`. When Z.AI's API can list the models your account has, the pickers also show those we haven't verified; a model your account lacks is tagged. `glm-5.3` models always think — thinking can't be turned off.

Any ID your provider serves works: a name that isn't in the registry is accepted with a warning (its thinking and endpoint support are then unverified), and a model registered only for another provider is refused. `GLM_MODEL` / `OPENCODE_MODEL` still exist as the provider's fallback model for calls that belong to no stage.

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
src/synapse_deep_research/
  main.py                  CLI entrypoint + command dispatch
  cli_help.py              Command registry, rich --help, styled usage errors
  options.py               Option values + what they mean (shared by help and REPL)
  ui.py                    Shared theme, console, glyphs, panels
  views.py                 Shared screens: banner, history, inspect, status, saved card
  repl.py                  Interactive REPL
  interactive.py           Transient pickers, text prompt and viewer
  onboarding.py            The setup screen (provider, keys, models)
  userconfig.py            Saved choices (provider, model per stage)
  config.py                Settings from .env
  credentials.py           OS-keyring credential access
  setup_wizard.py          setup / login / logout / keys flows
  logging_config.py        Logging setup
  llm.py                   LLM client (sync + async, multi-provider)
  capabilities.py          Per-model capability registry + recommended stage models
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
