# Changelog

## Unreleased

### Added

- **Rate-limit resilience.** Provider-aware exponential backoff with jitter for `429`/timeout errors (rate limits wait ~3× longer than generic transients), and per-retry logging of provider, endpoint family, model, stage, and attempt count. Tenacity now re-raises the underlying provider error after exhausting retries so the fallback-provider swap and `ProviderLimitExceeded` paths actually fire.
- **Concurrency knobs**: `MAX_LLM_CONCURRENCY` (analyze fan-out), `AGENT_MAX_LLM_CONCURRENCY` (gentler default auto-applied in `--agent` mode), and `MAX_SEARCH_CONCURRENCY` (shared cap on search + extraction HTTP). Search providers now honor a shared semaphore instead of opening unbounded concurrent requests.
- **Run-shape / budget knobs**: `MAX_SUB_QUESTIONS`, `MAX_SEARCH_RESULTS`, `MAX_ITERATIONS`, `MAX_SOURCES`, `AGENT_BUDGET_QUICK/_STANDARD/_DEEP`, and `LLM_MAX_RETRIES` / `LLM_RETRY_BASE_SECONDS` / `LLM_RETRY_MAX_SECONDS`.
- **Graceful partial results**: if synthesis fails after sources are gathered (e.g. a mid-run rate-limit), the run preserves the source analyses and saves the session instead of crashing — re-run with `--resynthesize <id>` to retry only synthesis.
- README "Rate Limits & Concurrency" section, including recommended Z.AI/GLM settings for research vs coding workloads. New tests for backoff/jitter, retry/fallback, bounded concurrency, the partial-result path, and an offline end-to-end pipeline run.

### Fixed

- Synthesizer no longer leaks the report title/metadata into `executive_summary` (which duplicated headers in saved reports and showed the wrong content in `display_report`/`/inspect`).
- `aidentify_gaps` tolerates malformed gap JSON instead of aborting the run; dedup no longer raises `ValueError` on a stale index pointer.
- `extracted_data` now round-trips through the session DB. arXiv / Semantic Scholar requests send a User-Agent. Eval runner uses the active provider/model; eval citation checks count grouped `[1, 2]` refs. `/history` shows 25 rows; `/route` rolls back on error.

## v0.3.0 — 2026-05-10

### Added

- **OS-native secure credential storage.** API keys are stored in the OS keyring (macOS Keychain, Linux Secret Service / GNOME Keyring / KWallet, Windows Credential Manager) instead of `.env` plaintext. Existing `.env` files keep working — keyring is checked *after* env / `.env` so CI and scripted overrides take precedence.
- **`deep-research setup`** subcommand: interactive wizard that prompts for API keys (masked input, never echoed) and stores them in the keyring. Detects existing keys and offers to keep / overwrite / clear them.
- **`deep-research login [KEY]`** / **`deep-research logout [KEY|all]`** / **`deep-research keys`** subcommands for single-key management.
- **REPL slash commands** for the same: `/setup`, `/login`, `/logout`, `/keys`. The running session reloads credentials immediately after a change — no restart needed.
- **First-run UX**: when `deep-research` is launched and no API key is found anywhere (env, `.env`, keyring), it asks "Run the credential setup wizard now?" before exiting. Press Enter and you're walked through it inline.
- **`deep-research migrate [PATH]`** subcommand: moves a legacy `<repo>/data/research.db` to `~/.local/share/deep-research/research.db`. Backs up the destination if one exists before overwriting.
- 13 new tests covering credential CRUD, settings keyring fallback, env-source preservation on reload, and setup-wizard error paths. Total: 158 tests.

### Changed

- `Settings.model_post_init` now fills empty managed-credential fields from the keyring at construction. `Settings.reload_credentials()` re-reads from the keyring without overwriting env-sourced values, so REPL `/login` / `/logout` take effect mid-session.
- README install section documents the OS keychain story and the new subcommands as the recommended way to manage credentials.

### Note

This release adds `keyring` as a dependency. On Linux the default backend (`SecretService`) needs `dbus-daemon` and a desktop keyring like `gnome-keyring` or `kwallet`; for headless servers, set `KEYRING_BACKEND=keyring.backends.fail.Keyring` and supply credentials via env or `.env` instead.

## v0.2.1 — 2026-05-10

### Changed

- **Filesystem paths now follow XDG conventions.** The session DB moves from `<repo>/data/research.db` to `~/.local/share/deep-research/research.db` (`$XDG_DATA_HOME` respected). Saved markdown reports default to `<cwd>/outputs/` so they land next to wherever you ran the command. Both are overridable via `DEEP_RESEARCH_DATA_DIR` and `DEEP_RESEARCH_OUTPUTS_DIR`. The output dir is created lazily on first save, not at import — so `deep-research --history` from a random directory no longer leaves an empty `outputs/` behind.
- **`.env` discovery falls back to `~/.config/deep-research/.env`.** Project-local `.env` (cwd or above) still wins; the XDG-config fallback lets pipx-installed users set credentials in one place.
- **Migration notice**: if the legacy `<repo>/data/research.db` exists and the new XDG path doesn't, a one-line warning fires at startup pointing at the move (or you can set `DEEP_RESEARCH_DATA_DIR` to the old location).

### Added

- `pipx install git+https://github.com/ferhatyurdakul/Deep-Research.git` is now the recommended install path. README documents both pipx and dev/editable workflows.
- 7 new tests covering path resolution, env-var overrides, and lazy outputs-dir creation.

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
