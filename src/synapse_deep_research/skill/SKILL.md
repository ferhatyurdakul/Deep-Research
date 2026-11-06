---
name: synapse-deep-research
description: Run a multi-source, cited deep-research report from the terminal with the Synapse Deep Research CLI (web search plus arXiv and Semantic Scholar). Use it when the user asks for a thorough, sourced investigation, a literature review, a market or technology survey, or says "deep research". Do not use it for quick factual lookups or anything one web page can answer. A run takes minutes and spends the user's own model quota, so start one only when that is wanted.
---

# Synapse Deep Research

A command-line research agent. It splits a question into sub-questions, searches the web and paper databases, reads and analyzes each source, goes around again for what is missing, and writes a cited Markdown report. It runs on the user's own model account, so it uses none of your context or quota.

## Before you run it

1. `synapse-deep-research --version`. If the command is not found, tell the user how to install it (`pipx install git+https://github.com/ferhatyurdakul/synapse-deep-research.git`) and stop; do not install it yourself unasked.
2. It needs an API key that only the user can set up, with `synapse-deep-research setup` in their own terminal. If a run fails because the key is missing, tell them to run that. Never ask for a key in the chat or put one on a command line.
3. Ask before `--agent` and before running the same question twice; both use a large share of the quota. Depth, template and style are your call (see below), so do not ask about those.

## Run it

```bash
synapse-deep-research --json --depth <depth> --style <style> [--template <template>] "<a self-contained question>" 2>/tmp/synapse-progress.log
```

- **Always `--json`.** Stdout is then exactly one line of JSON. Progress and messages go to stderr; send them to a file, they are noisy.
- **Pass `--depth` and `--style` yourself.** Left out, they fall back to the user's saved settings, which may not suit this question.
- **It takes minutes.** That is longer than many shell tools wait by default, so use a long timeout or run it in the background and wait for it to exit. One run at a time: the model provider caps concurrency.
- **Make the question self-contained.** The tool sees none of this conversation. Put in the names, the timeframe and the scope.

Success:

```json
{"ok": true, "session_id": 7, "parent_session_id": null, "query": "...", "depth": "quick",
 "report_path": "/abs/path/outputs/20261101_120000_question.md",
 "executive_summary": "...", "follow_up_questions": ["..."],
 "sources": 12, "iterations": 1, "seconds": 84.3}
```

Failure: `{"ok": false, "error": "..."}` and a non-zero exit status. Show the user the error. Do not retry blindly: a missing key or a rejected model will fail the same way again.

## Choose the options yourself

The user asked a question, not for a particular setup, so pick the options that fit it and do not ask. If the user named one (a depth, "a brief report", "a literature review", "no papers"), use exactly that and decide only the rest. Say your choices in one line when you start, for example "Running standard depth, market template, brief style", so they can steer; do not wait for an answer.

**`--depth`**: how much is searched and read. quick: 3 sub-questions, 1 pass. standard: 5, 2 passes. deep: 7, 3 passes, and research papers are added.

- `quick`: one narrow question, a definition or explainer, or a first look to see whether more is worth it.
- `standard`: the usual choice. A question with several angles, a comparison of options, "what is the state of X". Use it when unsure.
- `deep`: broad or high-stakes questions. A decision with real consequences, a survey of a field, claims that have to hold up, or the user asking for something thorough.

When torn between two, take the lower one. If the report comes back thin, `continue` (below) adds to that session instead of starting over.

**`--template`**: only when the topic clearly fits one; leave it out otherwise.

- `technology`: software, hardware, AI, infrastructure.
- `market`: industries, competitors, trends, sizing.
- `science`: biology, physics, chemistry, medicine. Also searches research papers.
- `literature`: a systematic survey of an academic field. Also searches research papers.

**`--style`**: the shape and length of the report.

- `brief`: about 500-1000 words. The user wants an answer to act on or pass along.
- `standard`: about 2000-4000 words in themed sections. The usual choice.
- `academic`: about 3000-5000 words with abstract, methodology and discussion. The user will cite or publish it, or asked for a paper-style write-up.

**`--papers` / `--no-papers`**: leave it on auto, where the template and depth decide. Use `--papers` when the user wants scholarly sources on a topic that is not science, and `--no-papers` to keep a practical or product question on web sources even at `deep`.

**`--fresh`**: pages analyzed in earlier sessions are skipped by default; use it if a topic researched before comes back thin.

## Use the result

- Start from `executive_summary`. Read `report_path` for the detail, only the parts you need, and give the user the path rather than pasting the whole report.
- Claims in the report carry numbered citations. When one matters, check it: `synapse-deep-research inspect <session_id> --source <N>` prints that source's findings and its verbatim evidence quotes.
- `follow_up_questions` are the tool's own suggestions for the next question.
- Say what the research found and where it is thin. A report built on only a few sources should be described that way.

## Follow up on a session

Both accept `--json` and save a new session linked to the original:

- `synapse-deep-research continue <session_id> --json` runs another pass over what is missing (`--depth deep` to go further).
- `synapse-deep-research resynthesize <session_id> --json --style brief` rewrites the report from the stored sources with no new searching, which is cheap.

`synapse-deep-research history` lists sessions (plain text). Sessions are kept in the user's data directory and are shared by every project; reports are written to `./outputs/` of the directory the command ran in.
