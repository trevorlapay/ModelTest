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
import re
from typing import Dict, Any

from .models import ModelCaller
import io
import contextlib

from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.orchestrator import PAIROrchestrator
from pyrit.models.score import Score


def parse_pyrit_transcript(text: str):
    """
    Turns logs like:
      user:
      Converted value: ...
      assistant: ...
    into a list of dicts with turn, attack_prompt, target_response.
    """
    entries = []
    current_user = None
    turn = 0

    # We’ll walk line-by-line to preserve order reliably
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip memory/info logs
        if line.startswith("INFO:"):
            i += 1
            continue

        if line.startswith("user:"):
            # Next non-empty line that starts with "Converted value:" is the prompt
            j = i + 1
            prompt = None
            while j < len(lines):
                nxt = lines[j].strip()
                if nxt.startswith("Converted value:"):
                    prompt = nxt[len("Converted value:"):].strip()
                    break
                if nxt.startswith(("user:", "assistant:", "INFO:")):
                    break
                j += 1
            current_user = prompt or ""
            i = j if j > i else i + 1
            continue

        if line.startswith("assistant:"):
            resp = line[len("assistant:"):].strip()
            turn += 1
            entries.append({
                "turn": turn,
                "attack_prompt": current_user or "",
                "target_response": resp,
                # You can populate these later if you capture evaluator outputs
                "evaluation_response": "",
                "score": None,
            })
            current_user = None
            i += 1
            continue

        i += 1

    return entries

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
        score = getattr(result, "objective_score", getattr(result, "score", 0.0))

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            await result.print_conversation_async()
        transcript_text = buf.getvalue()

        structured = parse_pyrit_transcript(transcript_text)

        return score, structured, transcript_text

    final_score, transcript_entries, transcript_text = asyncio.run(_run())

    # If the result is a PyRIT Score object, extract its numeric value.
    if isinstance(final_score, Score):
        score_value = final_score.get_value()
        if isinstance(score_value, bool):
            score_value = 1.0 if score_value else 0.0
    else:
        score_value = float(final_score)

    success = score_value > threshold

    return {
        "success": success,
        "final_score": float(score_value or 0.0),

        # IMPORTANT: include BOTH
        "transcript": transcript_entries,       # list[dict] (may be empty)
        "transcript_text": transcript_text,     # raw string (fallback)
    }
