"""Research pipeline: orchestration, decomposition, analysis, synthesis, agent."""
# Re-exports are resolved lazily (PEP 562) so importing a light submodule such
# as `pipeline.templates` — which `--help` and the REPL's option listings do —
# doesn't drag in the LLM/search client stack.
_LAZY_EXPORTS = {
    "arun_research": "orchestrator",
    "acontinue_research": "orchestrator",
    "run_research": "orchestrator",
    "arun_agent_research": "agent",
}


def __getattr__(name: str):
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    return getattr(import_module(f".{module}", __name__), name)


__all__ = list(_LAZY_EXPORTS)
