"""LLMAgentOpt - canonical entry point for the multi-agent hyperparameter tuner.

This module re-exports everything from `lean_tuner.py` under the cleaner
`llm_agent_opt` name. New code should import from here:

    from llm_agent_opt import tune, SEARCH_SPACE
    from llm_agent_opt import diagnostic_agent, strategy_agent, tuning_agent

CLI usage is identical to the legacy script:

    python llm_agent_opt.py --study-name noiseless --n-trials 30 \\
        --mode llm_agent_opt --device cuda

The legacy `python lean_tuner.py --mode lean3 ...` keeps working — both
files share the same code via this shim. The literal mode string "lean3"
is accepted as an alias for "llm_agent_opt" so existing run scripts /
saved JSON results stay valid.
"""
from lean_tuner import *                          # noqa: F401, F403
from lean_tuner import (                          # explicit re-exports for IDEs
    SEARCH_SPACE,
    diagnostic_agent, strategy_agent, tuning_agent,
    DIAGNOSTIC_SYSTEM, DIAGNOSTIC_USER_TEMPLATE, STRATEGY_SYSTEM,
    tune,
    _filter_to_search_space, sample_from_space,
)


if __name__ == "__main__":
    # Defer to the lean_tuner CLI - same flags, same behavior
    import runpy
    runpy.run_module("lean_tuner", run_name="__main__")
