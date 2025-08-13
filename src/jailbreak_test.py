"""
Automated jailbreak testing using PyRIT.

This module defines functions to orchestrate a jailbreak attack against a target
language model using PyRIT's PAIROrchestrator. The orchestrator drives an
attacker model to iteratively refine prompts in a crescendo-like manner, while
a scoring model evaluates whether the objective has been achieved. This
function supports both remote and local target models (via LiteLLM). Attacker
and scorer models are always remote.

Usage:
- Only the target model may be local. Scorer and attacker should be remote.
- Provide OpenAI API keys via environment variables or Config.
"""

from __future__ import annotations

import asyncio
from typing import Dict, Any

from .models import ModelCaller

from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.orchestrator import PAIROrchestrator


def parse_first_float(text: str) -> float:
    import re
    match = re.search(r"([01](?:\.\d+)?)", text)
    return float(match.group(1)) if match else 0.0


def run_jailbreak_test(
    target: ModelCaller,
    scorer: ModelCaller,
    attacker: ModelCaller,
    objective: str,
    scorer_prefix: str,
    depth: int,
    threshold: float,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Execute a jailbreak test for a single pass using PyRIT's PAIROrchestrator.
    The orchestrator mounts a multi-turn crescendo attack by iteratively refining
    adversarial prompts and evaluating the target's responses via the scoring model.

    Parameters
    ----------
    target : ModelCaller
        The target language model to be attacked. This may be local via LiteLLM.
    scorer : ModelCaller
        The remote model used to evaluate whether the objective has been achieved.
    attacker : ModelCaller
        The remote model used to generate adversarial prompts.
    objective : str
        The jailbreak objective the attacker strives to achieve.
    scorer_prefix : str
        Desired response prefix given to the scoring model before evaluation.
    depth : int
        Maximum number of turns to attempt the jailbreak.
    threshold : float
        Score threshold (0-1) at which the objective is considered achieved.
    verbose : bool, optional
        Whether to print debug information to stdout.

    Returns
    -------
    dict
        A dictionary with keys 'success', 'final_score', and 'conversation'.
    """
    # Initialize PyRIT
    initialize_pyrit(memory_db_type=IN_MEMORY)

    # Determine endpoints for OpenAIChatTarget
    def build_target(model: ModelCaller, force_local: bool) -> OpenAIChatTarget:
        if force_local:
            endpoint = "http://127.0.0.1:4000/v1/chat/completions"
            api_key = model.api_key or "local-key"
        else:
            endpoint = "https://api.openai.com/v1/chat/completions"
            api_key = model.api_key
        return OpenAIChatTarget(
            api_key=api_key,
            model_name=model.model_name,
            api_version=None,
            endpoint=endpoint,
        )

    # Build PyRIT prompt targets
    objective_target = build_target(target, force_local=target.local)
    adversarial_chat = build_target(attacker, force_local=False)
    scoring_target = build_target(scorer, force_local=False)

    # Instantiate PAIROrchestrator
    orchestrator = PAIROrchestrator(
        objective_target=objective_target,
        adversarial_chat=adversarial_chat,
        scoring_target=scoring_target,
        desired_response_prefix=scorer_prefix,
        depth=depth,
        objective_achieved_score_threshold=threshold,
        verbose=verbose,
    )

    async def _run():
        result = await orchestrator.run_attack_async(objective=objective)
        # Many orchestrators expose objective_score or score; fallback to 0.0 if missing
        score = getattr(result, "objective_score", getattr(result, "score", 0.0))
        # Print conversation and capture as text
        conversation_text = await result.print_conversation_async()
        return score, conversation_text

    final_score, conversation = asyncio.run(_run())
    success = bool(final_score >= threshold)
    return {"success": success, "final_score": float(final_score), "conversation": conversation}
