# Changelog

## v0.4.0 — 2026-10-24

### Added

- **Rate-limit resilience.** Provider-aware exponential backoff with jitter for `429`/timeout errors (rate limits wait ~3× longer than generic transients), and per-retry logging of provider, endpoint family, model, stage, and attempt count. Tenacity now re-raises the underlying provider error after exhausting retries so the fallback-provider swap and `ProviderLimitExceeded` paths actually fire.
- **Concurrency knobs**: `MAX_LLM_CONCURRENCY` (analyze fan-out), `AGENT_MAX_LLM_CONCURRENCY` (gentler default auto-applied in `--agent` mode), and `MAX_SEARCH_CONCURRENCY` (shared cap on search + extraction HTTP). Search providers now honor a shared semaphore instead of opening unbounded concurrent requests.
- **Run-shape / budget knobs**: `MAX_SUB_QUESTIONS`, `MAX_SEARCH_RESULTS`, `MAX_ITERATIONS`, `MAX_SOURCES`, `AGENT_BUDGET_QUICK/_STANDARD/_DEEP`, and `LLM_MAX_RETRIES` / `LLM_RETRY_BASE_SECONDS` / `LLM_RETRY_MAX_SECONDS`.
- **Graceful partial results.** Three layers: a source whose analysis fails (e.g. a 429 that outlives the retry budget) is skipped while its siblings are kept; an iteration that fails after sources were already collected stops the loop and synthesizes from what exists (the session is saved — resume with `--continue-session <id>`); and a synthesis failure preserves the gathered sources (retry with `--resynthesize <id>`). If *every* analysis fails, the run still aborts loudly rather than saving a hollow, source-less report.
- README "Rate Limits & Concurrency" section, including recommended Z.AI/GLM settings for research vs coding workloads. New tests for backoff/jitter, retry/fallback, bounded concurrency, the partial-result path, and an offline end-to-end pipeline run.
- **`zai` route profiles**: `budget` (glm-4.5-air for decompose+analyze, glm-5.1 for gap/synthesis) and `quality` (glm-5.1 everywhere) join `balanced`, so `LLM_ROUTE` works uniformly across both providers.
- REPL `/fork <id> [query]` and `/resynthesize <id>` mirror the CLI flags; REPL history now respects `XDG_CONFIG_HOME`.


### Changed

- `MAX_LLM_CONCURRENCY` default lowered `5` → `2`. The default provider (Z.AI GLM coding plan) enforces a low per-plan concurrency cap that returns `429` (code `1302`), so the old fan-out of 5 mostly generated retry storms on Lite/Pro tiers. Raise it back to `3`–`5` on providers with real concurrency headroom.
- **URL dedup is normalized end-to-end.** Cross-session URL filtering and the `analyzed_urls` registry compare the canonical form (scheme/`www.`/trailing-slash stripped), matching the within-search dedup, so variants of a known page are no longer re-fetched and re-analyzed. `report.searched_urls` now stores only URLs first seen in that run instead of a copy of the global history (session rows no longer grow unboundedly, and forks no longer duplicate it).
- **`max_scrape_pages` is enforced.** The depth presets' page budgets (quick 2 / standard 3 / deep 5) now actually cap fetching; results beyond the budget become snippet sources.
- Bare `deep-research logout` (and `/logout`) now clears the *active provider's* key — matching the README and `login`'s behavior — instead of every stored credential (`logout all` still clears everything).
- CLI `--history` shows 25 rows, matching the REPL.

### Fixed

- Deleting a parent session now clears `parent_session_id` on its children instead of leaving a dangling lineage id in `--history`; `--fork-query` without `--fork-session` is a hard error instead of being silently ignored; `fork_session` returns `None` instead of fabricating a `standard`-depth fork if the source session disappears mid-copy.
- Synthesizer no longer leaks the report title/metadata into `executive_summary` (which duplicated headers in saved reports and showed the wrong content in `display_report`/`/inspect`).
- `aidentify_gaps` tolerates malformed gap JSON instead of aborting the run; dedup no longer raises `ValueError` on a stale index pointer.
- `extracted_data` now round-trips through the session DB. arXiv / Semantic Scholar requests send a User-Agent. Eval runner uses the active provider/model; eval citation checks count grouped `[1, 2]` refs. `/history` shows 25 rows; `/route` rolls back on error.
- **Startup gating**: a failed capability check now aborts the run instead of proceeding with silently mismatched models, a typo'd `LLM_PROVIDER`/`LLM_ENDPOINT_FAMILY` prints a clean error (from the banner and `check_config`) instead of a traceback, an invalid `RESEARCH_DEPTH` env value falls back to `standard` with a warning, and `LLM_FALLBACK_PROVIDER` is validated at startup rather than exploding mid-429-recovery.
- **Fallback-provider swap is route-safe**: after swapping, per-stage models registered only under the old provider fall back to the new provider's default (instead of being sent to an endpoint that 400s them), and an incompatible pinned `LLM_ENDPOINT_FAMILY` is cleared.
- `--continue-session` now honors `--template` (guidance + system prompt + academic/extraction toggles), matching run/agent/resynthesize; follow-up questions after an `--agent` run stay in agent mode.
- Hardening: negative `MAX_*`/retry env overrides are clamped instead of producing hollow zero-iteration runs or retry storms; malformed agent-decision JSON (non-dict, string `queries`) degrades instead of crashing; `aanalyze_source`/`aextract_data` tolerate top-level JSON arrays; `adecompose_query` truncates to `max_q`; synthesis with zero sources skips the LLM call with an actionable message instead of emitting "cite sources 1 to 0"; CLI `--delete` asks for confirmation on a tty; dead `Settings.max_sub_questions/max_search_results/max_scrape_pages` fields removed.

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
