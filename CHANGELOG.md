# Changelog


## Unreleased

### Breaking — models are chosen per stage; route profiles are gone

- **`LLM_ROUTE` and `/route` are removed**, along with the `balanced` / `budget` / `quality` presets. You now choose the model for each pipeline stage (decompose, analyze, gap_analysis, synthesize) yourself — in the new setup screen, or any time with `/model` — and the choice is saved to `~/.config/deep-research/config.json` (`DEEP_RESEARCH_CONFIG_FILE` overrides the path). Per stage the model comes from, in order: a `MODEL_<STAGE>` environment variable, what you saved, then the provider's recommendation (Z.AI: `glm-5.3-flash` for decompose/analyze, `glm-5.3` for gap_analysis/synthesize; OpenCode Go: `glm-5.1` everywhere). A leftover `LLM_ROUTE` is ignored with a one-time warning. `deep-research status` and `/status` show where each stage's model comes from. The Python API changes accordingly: `RECOMMENDED_ROUTES`, `known_profiles`, `resolve_stage_route` and `Settings.llm_route` are replaced by `recommended_stage_models()` and `Settings.stage_sources()`.
- **`/model` now saves** instead of lasting only for the session (that was the only way to keep a choice before; now it is the way to make one). It also gains an `other` entry to type any model ID, and `default` means "back to the recommendation".
- **`/provider` saves the provider** too, so the app opens on the one you last chose (`LLM_PROVIDER` in the environment still wins).

### Breaking — research papers are `auto`, `on` or `off` (was the `academic` switch)

- **`/academic` is now `/papers`, and no longer an on/off toggle.** Searching research papers (arXiv + Semantic Scholar) was switched on by any of three things — the switch, the science / literature templates, and `deep` depth — and the switch could only add to that, so "off" could still mean "on". It is now a three-way setting: **`auto`** (the default) follows the template and depth, **`on`** always searches papers, **`off`** never does, even on `deep` or `science`. Bare `/papers` opens a picker whose `auto` row says what it amounts to right now (`now on (science template)`); `/papers on|off|auto` sets it directly; `/status` shows `auto — on (science template)`; the status line and banner tag is `papers`.
- **Why the rename:** the switch was called `academic`, and so is a report style (`--style academic`, `/style academic`) — `--no-academic --style academic` read as a contradiction when the two have nothing to do with each other (one is where sources come from, the other how the report reads). The style keeps its name, because it is stored with saved sessions.
- **`--papers` / `--no-papers`** replace `--academic`, and add the missing way to turn it off. Giving both is a usage error. Both work on `continue` and `resynthesize` too. **The old names `--academic`, `--no-academic` and `/academic` are removed, not aliased** — an alias would keep the clashing word alive. Scripts that pass `--academic` need `--papers` (or just drop it on `--depth deep`, which includes papers on `auto`); with no flag a run behaves as it did.

### Added — setup screen

- **A real setup screen** replaces the key-only wizard: provider → API key (hidden, OS keyring) → optional Tavily key → a model for each stage (or accept the recommended set) → optional connection check, one tiny request per model, so a wrong key or a model your plan can't use shows up immediately. Every step is a transient picker or prompt; only a summary stays on screen.
- **It opens automatically on first launch** in a terminal when the active provider has no API key, and via `deep-research setup` / `/setup`. Esc skips it at any step. It never appears outside a terminal (pipes, scripts, `deep-research "question"` with no key still print the error and exit 1). One-shot runs and `continue` / `resynthesize` with no key open it too.
- **Live model list.** When the provider's API can list the models the account has (Z.AI's coding endpoint can), the pickers tag registry models the account lacks and offer the others as "on your account · not verified here". Any failure falls back to the built-in list silently.

### Changed — Z.AI model list and default `glm-5.3`

- The registry now matches Z.AI's current lineup (checked 2026-10-25): `glm-5.3`, `glm-5.3-flash`, `glm-5.3-flashx`, `glm-5.2`, `glm-5.1` are offered; `glm-5` and `glm-4.5-air` stay registered (existing configs keep validating) but are no longer listed. `glm-5.3`, `glm-5.3-flash`, `glm-5.2` and `glm-5.1` are production tier (each answered a live call, with and without thinking); `glm-5.3-flashx` stays experimental because a Coding Plan account is refused (error 1311) despite it being in the account's model list. Models carry a `listed` flag for this.
- `GLM_MODEL` now defaults to `glm-5.3`. A `GLM_MODEL` or `MODEL_*` you have set yourself keeps winning. OpenCode Go's list and defaults are unchanged (to be refreshed separately).


### Changed — CLI redesign

- **One look everywhere.** A shared theme and console (`ui.py`) replace the five separate `Console()` instances and ad-hoc colors: the same ✓ / ✗ / ! lines, titled panels, section rules and key-value grids across one-shot runs, the REPL, the setup wizard and the pipeline's own output. Glyphs fall back to ASCII on non-UTF-8 terminals; colors follow `NO_COLOR` and disappear when piped.
- **Discoverable options.** `deep-research --help` is now a grouped reference — usage, every research option with what each value means (`quick` = 3 sub-questions / 1 pass, …), all commands by group, and examples — instead of the bare argparse listing that omitted every subcommand. `deep-research COMMAND --help` details one command; `deep-research help [COMMAND]` and `--version` are new. Usage errors print the problem, the command's usage line, and a pointer to `--help`.
- **Shared screens.** `history`, `inspect`, banner and run summary now render through `views.py`, so the CLI and the REPL show identical screens (the REPL's `/inspect` used to be a cut-down copy and now supports `--full` and `--source N`). History shows fork lineage and truncates to the terminal width.
- **Run summary.** A finished run ends with a "Saved" card (session ID, report path, source/iteration count, elapsed time, next-step commands) instead of two loose lines. Follow-up questions are numbered so the "enter a number" prompt matches what is on screen. The one-shot banner carries the resolved model, search backend, run shape and the query, replacing the raw route-dict dump.
- **REPL.** `/help` is grouped and shows each setting's allowed values; `/help <command>` explains one. A setting command with options is an interactive picker: `/depth⏎` (also `/style`, `/template`, `/provider`) opens an arrow-key list with each value's meaning and the current one marked — ↑/↓ or j/k to move, Enter to select, 1-9 to jump, Esc to cancel. Outside a terminal (pipes, scripts) the same command prints the list instead. Tab-completion now completes values after the command (`/depth ` → quick, standard, deep), the menu scrolls with the selection so all commands are reachable, aliases (`/q`, `/exit`, `/?`) stay out of the bare-`/` menu, and an empty input box shows a hint. Mistyped commands suggest the closest match. `/fresh`'s description is corrected (it revisits past URLs). Only a submitted research question leaves its input box in the scrollback (it records what was asked and under which settings); slash commands, empty Enters and quitting erase theirs, so changing settings no longer stacks up boxes.
- **Quieter, cleaner logs.** The CLI logs warnings and errors only by default (the INFO stream was per-call route dicts and result counts interleaved with the UI); `DEEP_RESEARCH_LOG_LEVEL=INFO|DEBUG` restores them. Log lines now print through the shared console, so they appear above a live progress spinner instead of tearing it, and message text is never interpreted as markup. The eval runner keeps its existing stderr format.
- **REPL credential check.** A query with no API key configured now stops with `✗ ZAI_API_KEY is not set` and a pointer to `/setup` and `/status`, instead of starting a run that dies with an SDK "missing credentials" error naming the wrong provider.
- **Faster startup for non-research commands.** The LLM and search client stack is imported only when a run starts, so `--help`, `history`, `inspect`, `status` and the REPL banner no longer pay ~1.5s of import time (`deep_research.pipeline` re-exports are now lazy).

### Fixed (REPL display)

- **Paper search is shown when implied.** The `science` and `literature` templates and `deep` depth switch academic search on inside the pipeline, but the status line, banner and `/status` only reflected the explicit switch, so it looked like paper search was off. They now show `papers` whenever a run will use it (and `/status` names the reason).

### Added

- **`/model`** — change the model per pipeline stage from the REPL without editing `.env`: `/model` opens a stage picker then a model picker (the provider's current models with the recommended one tagged, `other` to type an ID, `default` to return to the recommendation); `/model analyze glm-5.1` and `/model all glm-5.3` work typed, with Tab-completion for both arguments. Saved; unregistered names are accepted with a warning, models registered only for another provider are refused, and an invalid combination rolls back.
- **Slash commands leave nothing behind.** Setting commands are silent on success (the status line shows the new state); read-only screens (`/help`, `/status`, `/history`, `/inspect`, `/keys`, `/fork`) and any error open in a closable, scrollable viewer that disappears on Enter, Esc or `q`; `/delete <id>` confirms with a Yes/No picker instead of a typed prompt (`/delete <id> --yes` skips it). Runs and credential wizards still print normally. Outside a terminal (pipes, scripts) every command prints as before.
- The banner's model line shows what each stage actually uses (`glm-5.3-flash · glm-5.3 for gap+synthesize`) instead of only `GLM_MODEL`, which could disagree with the stage models.
- `deep-research status` — resolved provider and per-stage models (with where each comes from), which credentials are set, and where the session DB and reports live.
- `deep-research history -n N` to choose how many sessions to list.

### Fixed

- **REPL: finished research crashed before saving.** `repl.py` called an unimported `db_save`, so every completed query or `/continue` in the REPL raised `NameError` after the report was displayed and before the session was stored.
- **REPL: `/style` had no effect.** `ReplState.to_config()` stopped copying the style into the research config; reports always came out in the default style.
- **Markup injection in messages.** Queries, session text, exception messages and LLM-written agent reasoning were interpolated into rich markup strings; text such as `[/foo]` could raise `MarkupError` mid-run. All user- and model-supplied text is now escaped or rendered as plain text.
- The credentials flow in non-interactive shells no longer blocks on a confirmation prompt when a key is missing; it prints how to configure one and exits.

### Breaking

- **Session management moved from flags to subcommands**: `deep-research history`, `delete <ID>`, `inspect <ID> [--full] [--source N]`, `fork <ID> [query...]`, `continue <ID> [shape flags]`, `resynthesize <ID> [shape flags]`. The old `--history`, `--delete`, `--inspect`, `--full`, `--source`, `--fork-session`, `--fork-query`, `--continue-session`, and `--resynthesize` flags are rejected by argparse. `continue` and `resynthesize` accept the same shape flags as one-shot runs and run the same config gate (key check / route validation) as a research run. In-run hint messages updated to the new syntax.
- **Thinking mode is always on** and the `--thinking` flag is removed. Every registered model either supports the thinking payload (GLM via `zai-extra-body`), reasons natively with the payload omitted (Kimi, DeepSeek, MiMo, Qwen — dialect `none`), or is auto-gated off (`glm-4.5-air`). (The REPL `/thinking` command was removed along with the flag.)

### Added

- **Interactive REPL UI** (prompt_toolkit): a Claude-Code-style rounded input box whose top border carries a colored, minimal run state (depth, template, non-default style, active toggles, provider/route — defaults are silent); typing `/` opens a fuzzy completion menu *below* the box with each command's usage and description, and plain queries never trigger it; persistent history with ↑/↓ and Ctrl-R; first Ctrl-C clears the input line. Non-tty stdin (pipes, scripts, tests) automatically falls back to the plain prompt. The `/thinking` command is gone along with the flag — thinking is always on.

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
