"""
Automated jailbreak testing using PyRIT's Tree‑of‑Attacks red teaming.

This module defines functions to orchestrate a jailbreak attack against a
target language model using the PyRIT framework. At each turn, an attacker
model proposes a new prompt based on the previous target responses and
evaluation scores. The target model's response is then evaluated by a
scoring model via PyRIT's `SelfAskTrueFalseScorer`. If the evaluation
exceeds a specified threshold, the attack is deemed successful and the
test ends early.

This implementation relies entirely on PyRIT's `RedTeamingOrchestrator`
class. PyRIT must be installed and available in the environment; no
fallback implementation is provided.

The final score for a pass is computed as 10 × (1 − evaluation), so a
higher score reflects greater resistance to jailbreak attempts. When
multiple passes are run, scores are averaged by the caller.
"""

from __future__ import annotations

import asyncio
import re
from typing import Dict, Any

from .models import ModelCaller

# Import required PyRIT components. These imports will raise ImportError if
# PyRIT is not installed, ensuring that PyRIT is a hard dependency.
from pyrit.common.initialization import initialize_pyrit, IN_MEMORY
from pyrit.orchestrator import RedTeamingOrchestrator
from pyrit.orchestrator.multi_turn.red_teaming_orchestrator import RTOSystemPromptPaths
from pyrit.prompt_target import HTTPTarget
from pyrit.score import SelfAskTrueFalseScorer, TrueFalseQuestion


def parse_first_float(text: str) -> float:
    """Extract the first floating point number from a string.

    If no number is found, returns 0.0. This helper is used to parse
    the scoring model's output when it returns a decimal rating.
    """
    match = re.search(r"([01](?:\.\d+)?)", text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return 0.0
    return 0.0


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
    """Execute a jailbreak test across a single pass using PyRIT.

    Parameters
    ----------
    target : ModelCaller
        Wrapper used to query the target model.
    scorer : ModelCaller
        Wrapper used to evaluate how well the target's response meets
        the objective.
    attacker : ModelCaller
        Wrapper used to generate attack prompts at each iteration.
    objective : str
        The desired behaviour that the target model should not perform.
    scorer_prefix : str
        The system prompt sent to the scoring model. The objective will
        be appended to this prefix when constructing the scoring prompt.
    depth : int
        Maximum number of attack iterations to run.
    threshold : float
        A score above this value is considered a successful jailbreak.
    verbose : bool, optional
        If True, prints detailed logs of each step.

    Returns
    -------
    dict
        A dictionary containing:
            - ``success``: whether the objective was achieved;
            - ``final_score``: the final evaluation score returned by PyRIT;
            - ``transcript``: a list capturing the conversation transcript (currently empty, as PyRIT manages its own history).
    """
    # Initialise PyRIT with an in‑memory database
    initialize_pyrit(IN_MEMORY)

    # Create PyRIT HTTP targets for each model. HTTPTarget will forward
    # chat completions to the underlying model endpoints specified in
    # ModelCaller. We rely on the base_url attribute of our ModelCaller
    # objects to build these endpoints.
    objective_target = HTTPTarget(target.base_url)
    adversarial_chat = HTTPTarget(attacker.base_url)
    scorer_target = HTTPTarget(scorer.base_url)

    # Construct the objective scorer using a self‑ask true/false scoring model.
    # We pass a default true/false question definition from PyRIT.
    objective_scorer = SelfAskTrueFalseScorer(
        chat_target=scorer_target,
        true_false_question=TrueFalseQuestion.DEFAULT,
    )

    # Build the RedTeamingOrchestrator. We set the system prompt path for
    # the adversarial chat to a text generation configuration and specify
    # the maximum number of turns for the attack.
    orchestrator = RedTeamingOrchestrator(
        objective_target=objective_target,
        adversarial_chat=adversarial_chat,
        adversarial_chat_system_prompt_path=RTOSystemPromptPaths.TEXT_GENERATION.value,
        max_turns=depth,
        objective_scorer=objective_scorer,
        verbose=verbose,
    )

    # Run the attack asynchronously. The orchestrator returns an object
    # containing the last objective score and status.
    result = asyncio.run(orchestrator.run_attack_async(objective=objective))

    # Extract the objective score. When PyRIT achieves the objective,
    # ``result.objective_score`` will be 1.0; otherwise it will be < 1.0.
    score_value = getattr(result, "objective_score", 0.0)

    # Compute a final score on the scale [0, 10]. A lower objective score
    # means the attack was less successful, so we invert it.
    final_score = max(0.0, min(10.0, 10.0 * (1.0 - score_value)))

    return {
        "success": score_value >= threshold,
        "final_score": final_score,
        "transcript": [],  # transcript not exposed by PyRIT for now
    }
