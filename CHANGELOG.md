# Changelog

## v0.2.0 — 2026-05-10

### Added

- **Interactive REPL**: bare `deep-research` (no arguments) now opens a persistent prompt with slash commands, similar to the `python` REPL or Claude Code's CLI. Type a research question to run it; use `/depth`, `/style`, `/template`, `/thinking`, `/academic`, `/agent`, `/fresh`, `/provider`, `/route` to change settings, and `/history`, `/inspect`, `/continue`, `/delete` to browse and refine stored sessions. Command history persists in `~/.config/deep-research/repl_history` (stdlib `readline`).
- **OpenCode Go provider** as a first-class alternative to Z.AI. Set `LLM_PROVIDER=opencode-go` + `OPENCODE_API_KEY=…` in `.env`. The capability map auto-routes each model to the right endpoint (OpenAI-compat for GLM/Kimi/MiMo/Qwen/DeepSeek, Anthropic-compat for MiniMax) and emits the correct thinking-payload dialect per upstream so we never ship Z.AI-shaped fields to non-GLM models.
- **Named stage routes**: `LLM_ROUTE=balanced|budget|quality` picks per-stage models from a per-provider catalog (decompose / analyze / gap_analysis / synthesize). Per-stage env overrides (`MODEL_DECOMPOSE` etc.) still win.
- **Per-run usage tracking**: `UsageTracker` counts LLM calls by `(provider, model, stage)` and surfaces a one-line summary at the end of every run. Snapshot also lands on `ResearchReport.usage` for stored sessions.
- **Rate-limit fallback**: `LLM_FALLBACK_PROVIDER=zai` (or any other configured provider) makes Deep Research swap providers mid-run when the primary trips a 429, instead of aborting. Without a fallback, you get a clean `ProviderLimitExceeded` error naming the route.
- **`UpstreamRequestError`** wraps OpenAI/Anthropic `BadRequestError` with `provider/model/family/dialect` so debugging is one log line instead of three.
- New CLI flags / env vars: `LLM_PROVIDER`, `LLM_ENDPOINT_FAMILY`, `LLM_ROUTE`, `LLM_FALLBACK_PROVIDER`, `OPENCODE_API_KEY`, `OPENCODE_BASE_URL`, `OPENCODE_MODEL`. See `.env.example`.

### Changed

- `chat()` / `achat()` now dispatch by endpoint family (OpenAI Chat Completions vs Anthropic Messages), with a single `_apply_thinking()` sink for provider extras. Generic kwargs builders carry only model/messages/system/temperature/max_tokens — no provider-specific fields leak.
- CLI banner now shows the active provider, route profile, and resolved model. The verbose route dump is suppressed when launching the REPL (use `/status` instead).

### Removed

- **Web UI / FastAPI server**. `server.py`, `static/index.html`, and the `deep-research-server` script entry are gone. `fastapi`, `uvicorn[standard]`, `websockets`, `weasyprint`, and the `markdown` package are no longer dependencies. PDF export went away with WeasyPrint; markdown export is unaffected.

### Tests

- 85 new unit tests covering capabilities, route profiles, request-shape dispatch, usage tracking, fallback behavior, and REPL command grammar. Total: 144 collected.

## v0.1.0

Initial release.
