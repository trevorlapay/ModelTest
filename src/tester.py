"""
Top‑level test harness for evaluating language models.

This module orchestrates the different individual tests (jailbreak,
bias, injection) and aggregates their results into a single final
score. It handles configuration loading, instantiates the required
model callers and manages multiple passes for each test.
"""

from __future__ import annotations

import statistics
from typing import Dict, Any, List

from .config import Config
from .models import ModelCaller
from .jailbreak_test import run_jailbreak_test
from .bias_test import run_bias_test
from .injection_test import run_injection_test


def run_all_tests(config: Config) -> Dict[str, Any]:
    """Run all defined tests and return aggregated results.

    Parameters
    ----------
    config : Config
        Configuration dataclass containing all test parameters and
        environment variables.

    Returns
    -------
    dict
        A dictionary summarising the outcomes of each test and the final
        aggregated score for the model under evaluation.
    """

    # Instantiate model callers according to the configuration
    target = ModelCaller(config.target_model, config.target_api_key, local=config.local_flag)
    scorer = ModelCaller(config.scorer_model, config.scorer_api_key, local=False)
    attacker = ModelCaller(config.attacker_model, config.attacker_api_key, local=False)

    # Collect results across passes
    jailbreak_scores: List[float] = []
    jailbreak_details: List[Any] = []

    for i in range(config.number_of_passes):
        jb_result = run_jailbreak_test(
            target=target,
            scorer=scorer,
            attacker=attacker,
            objective=config.objective,
            scorer_prefix=config.scorer_prefix,
            depth=config.node_depth,
            threshold=config.objective_threshold,
            verbose=config.verbose,
        )
        # Convert the 0–1 evaluation into a 1–10 score, invert to reflect desirability
        eval_score = jb_result["final_score"]
        score_out_of_ten = max(1.0, 10 * (1 - eval_score))
        jailbreak_scores.append(score_out_of_ten)
        jailbreak_details.append(jb_result)

    # Average jailbreak score across passes
    avg_jb_score = statistics.mean(jailbreak_scores) if jailbreak_scores else 1.0

    # Bias test (single pass)
    bias_result = run_bias_test(target, scorer, verbose=config.verbose)
    bias_score = bias_result["score"]

    # Prompt injection test (single pass)
    injection_result = run_injection_test(target, scorer, verbose=config.verbose)
    injection_score = injection_result["score"]

    # Final aggregated score (simple average of the three test scores)
    final_score = statistics.mean([avg_jb_score, bias_score, injection_score])

    return {
        "final_score": final_score,
        "jailbreak_scores": jailbreak_scores,
        "avg_jailbreak_score": avg_jb_score,
        "jailbreak_details": jailbreak_details,
        "bias_result": bias_result,
        "injection_result": injection_result,
    }
