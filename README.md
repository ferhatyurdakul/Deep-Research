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
deep-research --thinking "topic"                # enable chain-of-thought reasoning
deep-research --template science "topic"        # domain-specific strategy
deep-research --fresh "topic"                   # ignore URLs already analyzed in past sessions

# Combine flags freely
deep-research --agent --template science --style academic --depth deep "topic"

# Session management
deep-research --history                         # list past sessions
deep-research --continue-session 3              # deepen a previous session
deep-research --fork-session 3                  # copy session 3 into a new session (original untouched)
deep-research --fork-session 3 --fork-query "narrower angle on X"  # fork with a different query
deep-research --resynthesize 3                  # re-run only synthesis on session 3's sources, save as fork
deep-research --resynthesize 3 --style academic # same, but with a different report style
deep-research --inspect 3                       # show metadata, sub-questions, sources, iterations, summary
deep-research --inspect 3 --full                # also print the full detailed findings
deep-research --inspect 3 --source 5            # deep dive on source #5 (findings + evidence quotes)
deep-research --delete 5                        # remove a session
```

### Flag reference

| Flag                     | Values                                        | Default    | Notes                                                    |
|--------------------------|-----------------------------------------------|------------|----------------------------------------------------------|
| `--depth`                | `quick` \| `standard` \| `deep`               | `standard` | Controls sub-questions, results, and iteration count     |
| `--style`                | `brief` \| `standard` \| `academic`           | `standard` | Shape and length of the final report                     |
| `--template`             | `technology` \| `science` \| `market` \| `literature` | —    | Domain-specific prompts and defaults                     |
| `--agent`                | —                                             | off        | Autonomous decision loop instead of fixed iterations     |
| `--academic`             | —                                             | off        | Adds arXiv + Semantic Scholar                            |
| `--thinking`             | —                                             | off        | Enables chain-of-thought reasoning in the GLM client     |
| `--fresh`                | —                                             | off        | Disables cross-session URL deduplication                 |
| `--history`              | —                                             | —          | Lists stored sessions and exits                          |
| `--continue-session ID`  | integer                                       | —          | Resumes and deepens a prior session (gap-analysis loop; `--agent` is not applied) |
| `--fork-session ID`      | integer                                       | —          | Copies a session into a new branch; original untouched   |
| `--fork-query "..."`     | string                                        | —          | Used with `--fork-session` to set a different query      |
| `--resynthesize ID`      | integer                                       | —          | Re-runs synthesis on stored sources; saved as a child fork |
| `--inspect ID`           | integer                                       | —          | Prints session metadata, sources, iterations, summary    |
| `--full`                 | —                                             | off        | With `--inspect`: also prints the full detailed findings |
| `--source N`             | integer                                       | —          | With `--inspect`: deep dive on source #N (1-based)       |
| `--delete ID`            | integer                                       | —          | Deletes a stored session                                 |

## Search Sources

Four providers searched in parallel, deduplicated automatically:

| Provider         | Type     | Key Required | Notes                           |
|-----------------|----------|:------------:|---------------------------------|
| Tavily          | Web      | Optional     | High quality, needs API key     |
| DuckDuckGo      | Web      | No           | Free fallback, always available |
| arXiv           | Academic | No           | Full metadata, always available |
| Semantic Scholar | Academic | No           | API key optional for rate limits|

When no Tavily key is set, DuckDuckGo is the primary web search. Academic sources are always enabled.

## Depth Presets

| Depth      | Sub-questions | Results/query | Iterations | Thinking | Academic | Extraction |
|------------|:------------:|:-------------:|:----------:|:--------:|:--------:|:----------:|
| `quick`    | 3            | 3             | 1          |          |          |            |
| `standard` | 5            | 5             | 2          |          |          |            |
| `deep`     | 7            | 7             | 3          | yes      | yes      | yes        |

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

Anything not starting with `/` runs as a research query using the current settings. Slash commands change settings or browse history without leaving the session. Up/down arrows navigate command history; history persists in `~/.config/deep-research/repl_history`.

| Command | Description |
|---|---|
| `/depth [quick\|standard\|deep]` | Show or set research depth |
| `/style [brief\|standard\|academic]` | Show or set report style |
| `/template [technology\|science\|market\|literature\|none]` | Set domain template |
| `/thinking` | Toggle thinking mode |
| `/academic` | Toggle academic sources (arXiv + Semantic Scholar) |
| `/agent` | Toggle autonomous agent loop |
| `/fresh` | Toggle fresh mode (skip URLs from past sessions) |
| `/provider [zai\|opencode-go]` | Show or switch LLM provider |
| `/route [balanced\|budget\|quality]` | Show or set route profile |
| `/setup` | Run the credential setup wizard |
| `/login [KEY]` | Set a single API key in the OS keyring |
| `/logout [KEY\|all]` | Clear stored credentials |
| `/keys` | Show which credentials are stored (never shows values) |
| `/status` | Show current settings and resolved per-stage models |
| `/history` | List recent research sessions |
| `/inspect <id>` | Show a stored session's metadata + summary |
| `/continue <id>` | Deepen a stored session with current settings |
| `/delete <id>` | Delete a stored session (asks for confirmation) |
| `/clear` | Clear the screen |
| `/help` | Show command list |
| `/quit` (`/exit`, `/q`) | Exit the REPL |

One-shot CLI flags still work: `deep-research "topic"`, `deep-research --depth deep --thinking "topic"`, `deep-research --history`, `deep-research --inspect 3`, etc. — see the flag reference above.

## Multi-Model Routing

Route pipeline stages to different models for cost/quality tradeoffs:

```env
MODEL_DECOMPOSE=glm-4.5-air      # fast/cheap for decomposition
MODEL_ANALYZE=glm-5.1             # thinking model for analysis
MODEL_SYNTHESIZE=glm-5.1          # best model for synthesis
MODEL_GAP_ANALYSIS=glm-5.1        # thinking model for gap detection
```

Leave empty to use `GLM_MODEL` for everything.

## Project Structure

```
src/deep_research/
  main.py                  CLI entrypoint
  config.py                Settings from .env
  llm.py                   LLM client (sync + async, multi-provider)
  capabilities.py          Per-model capability registry + recommended routes
  usage.py                 Per-run call tracker + fallback metadata
  models.py                Pydantic data models
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

Runs are written to `evals/runs/<timestamp>-<git-sha>[-label]/` with one directory per benchmark (report, raw JSON, checks). Benchmarks are defined in `evals/benchmarks.py` and tagged (`smoke` for fast iteration, `full` for the complete set).

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
