# Deep Research

AI research agent that iteratively searches, analyzes, and synthesizes comprehensive reports with evidence-backed citations from web and academic sources.

Built on [Z.AI](https://z.ai) GLM models. Fully async. CLI + Web UI.

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

```bash
git clone https://github.com/yourusername/deep-research.git
cd deep-research
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Configure

```bash
cp .env.example .env
```

Open `.env` and set your `ZAI_API_KEY`. Everything else is optional.

### Run

```bash
# Interactive
deep-research

# Direct query
deep-research "your research question"

# Web UI
deep-research-server
```

## CLI Reference

```bash
# Depth presets
deep-research --depth quick "topic"             # 1 pass, fast
deep-research --depth standard "topic"          # 2 passes, balanced (default)
deep-research --depth deep "topic"              # 3 passes, thorough, academic sources

# Features
deep-research --agent "topic"                   # autonomous agent mode
deep-research --academic "topic"                # include arXiv + Semantic Scholar
deep-research --thinking "topic"                # enable chain-of-thought reasoning
deep-research --template science "topic"        # domain-specific strategy

# Combine
deep-research --agent --template science --depth deep "topic"

# Session management
deep-research --history                         # list past sessions
deep-research --continue-session 3              # deepen a previous session
deep-research --delete 5                        # remove a session
```

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

Safety limit: 8 iterations max.

## Web UI

```bash
deep-research-server    # http://localhost:8000
```

- Real-time progress via WebSocket
- Depth, template, thinking, and agent mode toggles
- Session history browser
- PDF and Markdown export

### API

| Endpoint                      | Method   | Description                      |
|-------------------------------|----------|----------------------------------|
| `/api/config`                 | GET      | Model info, depths, templates    |
| `/api/history`                | GET      | List past sessions               |
| `/api/session/{id}`           | GET      | Full session with rendered HTML  |
| `/api/session/{id}/markdown`  | GET      | Download as Markdown             |
| `/api/session/{id}/pdf`       | GET      | Download as PDF                  |
| `/api/session/{id}`           | DELETE   | Delete a session                 |
| `/ws/research`                | WS       | Real-time research WebSocket     |

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
  server.py                FastAPI + WebSocket server
  config.py                Settings from .env
  llm.py                   GLM client (sync + async)
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
```

## Running Tests

```bash
python -m pytest tests/test_e2e.py -v
```

Tests hit live APIs (Z.AI, Tavily, arXiv, Semantic Scholar). Tests requiring API keys are skipped automatically when keys are not set. Non-API tests (models, citations, DB, templates) always run.

## Contributing

1. Fork the repo
2. Create a feature branch
3. Make your changes
4. Run `python -m pytest tests/test_e2e.py -v` to verify
5. Open a PR

## License

MIT
