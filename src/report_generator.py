"""
Report generation for ModelTest.

This module contains a function that takes the raw results of the
testing harness and turns them into a visually appealing HTML
report. The report includes a summary table of scores, bar charts
illustrating each test’s performance and transcripts of the
jailbreak sessions. Charts are created with matplotlib and embedded
into the HTML as base64‑encoded images. The resulting HTML can be
viewed locally in a browser or uploaded elsewhere.
"""

from __future__ import annotations

import base64
import io
import json
import datetime
from typing import Dict, Any, List

import matplotlib.pyplot as plt


def _encode_fig_to_base64(fig) -> str:
    """Convert a matplotlib figure to a base64 encoded PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return encoded


def generate_report(results: Dict[str, Any], output_path: str) -> None:
    """Generate an HTML report summarising the test results.

    Parameters
    ----------
    results : dict
        The dictionary returned by `tester.run_all_tests()`.
    output_path : str
        Path on disk where the generated HTML file should be saved.
    """

    # Extract scores
    jailbreak_scores = results.get("jailbreak_scores", [])
    bias_score = results.get("bias_result", {}).get("score", 0)
    injection_score = results.get("injection_result", {}).get("score", 0)
    final_score = results.get("final_score", 0)

    # Create bar chart for test scores
    test_names = ["Jailbreak", "Bias", "Injection"]
    test_values = [results.get("avg_jailbreak_score", 0), bias_score, injection_score]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(test_names, test_values, color=["#4CAF50", "#2196F3", "#FF9800"])
    ax.set_ylim(0, 10)
    ax.set_ylabel("Score (1–10)")
    ax.set_title("Model Evaluation Scores")
    for bar, value in zip(bars, test_values):
        ax.annotate(f"{value:.2f}", xy=(bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, color='black')
    chart_img = _encode_fig_to_base64(fig)

    # Build HTML
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_parts: List[str] = []
    html_parts.append("<html><head><meta charset='UTF-8'><title>ModelTest Report</title>"
                      "<style>\nbody{font-family:Arial, sans-serif;margin:40px;background-color:#f9f9f9;}"
                      ".card{background-color:white;border-radius:8px;padding:20px;margin-bottom:20px;box-shadow:0 2px 4px rgba(0,0,0,0.1);}"
                      ".header{font-size:24px;font-weight:bold;margin-bottom:10px;}"
                      ".score-table{width:100%;border-collapse:collapse;margin-top:10px;}"
                      ".score-table th, .score-table td{border:1px solid #ddd;padding:8px;text-align:left;}"
                      ".score-table th{background-color:#f2f2f2;}"
                      ".transcript{font-family:monospace;white-space:pre-wrap;background:#fafafa;border:1px solid #eee;padding:10px;}"
                      "</style></head><body>")
    html_parts.append(f"<div class='card'><div class='header'>Model Evaluation Report</div>"
                      f"<p><strong>Generated:</strong> {timestamp}</p>"
                      f"<p><strong>Final Score:</strong> {final_score:.2f} / 10</p>"
                      f"<p>This report summarises the results of running jailbreak, bias and injection tests against the target model. "
                      "Scores are higher when the model is more robust to the corresponding vulnerability.</p></div>")
    # Chart section
    html_parts.append(f"<div class='card'><div class='header'>Score Summary</div>"
                      f"<img src='data:image/png;base64,{chart_img}' alt='Score chart' style='width:100%;max-width:600px;'></div>")
    # Score table
    html_parts.append("<div class='card'><div class='header'>Detailed Scores</div>")
    html_parts.append("<table class='score-table'><tr><th>Test</th><th>Score</th><th>Mean evaluation (0-1)</th></tr>")
    html_parts.append(f"<tr><td>Jailbreak (average of {len(jailbreak_scores)} runs)</td>"
                      f"<td>{results.get('avg_jailbreak_score',0):.2f}</td>"
                      f"<td>{results.get('jailbreak_details')[-1]['final_score'] if results.get('jailbreak_details') else 0:.3f}</td></tr>")
    html_parts.append(f"<tr><td>Bias</td><td>{bias_score:.2f}</td><td>{results.get('bias_result').get('mean_evaluation',0):.3f}</td></tr>")
    html_parts.append(f"<tr><td>Injection</td><td>{injection_score:.2f}</td><td>{results.get('injection_result').get('mean_evaluation',0):.3f}</td></tr>")
    html_parts.append("</table></div>")
    # Jailbreak transcripts
    if results.get("jailbreak_details"):
        html_parts.append("<div class='card'><div class='header'>Jailbreak Transcripts</div>")
        for pass_idx, jb in enumerate(results["jailbreak_details"], start=1):
            html_parts.append(f"<h4>Pass {pass_idx} (Success: {jb['success']}, Final score: {jb['final_score']:.3f})</h4>")
            for entry in jb.get("transcript", []):
                html_parts.append("<div class='transcript'>")
                html_parts.append(f"<strong>Turn {entry['turn']} - Attack prompt:</strong> {entry['attack_prompt']}\n")
                html_parts.append(f"<strong>Target response:</strong> {entry['target_response']}\n")
                html_parts.append(f"<strong>Evaluation response:</strong> {entry['evaluation_response']} (score={entry['score']:.3f})\n")
                html_parts.append("</div><br>")
        html_parts.append("</div>")
    # Bias details
    if results.get("bias_result"):
        html_parts.append("<div class='card'><div class='header'>Bias Test Details</div>")
        for detail in results["bias_result"]["details"]:
            html_parts.append("<div class='transcript'>")
            html_parts.append(f"<strong>Prompt:</strong> {detail['prompt']}\n")
            html_parts.append(f"<strong>Target response:</strong> {detail['target_response']}\n")
            html_parts.append(f"<strong>Evaluation response:</strong> {detail['evaluation_response']} (score={detail['score']:.3f})\n")
            html_parts.append("</div><br>")
        html_parts.append("</div>")
    # Injection details
    if results.get("injection_result"):
        html_parts.append("<div class='card'><div class='header'>Injection Test Details</div>")
        for detail in results["injection_result"]["details"]:
            html_parts.append("<div class='transcript'>")
            html_parts.append(f"<strong>Context:</strong> {detail['context']}\n")
            html_parts.append(f"<strong>Attack instruction:</strong> {detail['attack_instruction']}\n")
            html_parts.append(f"<strong>Target response:</strong> {detail['target_response']}\n")
            html_parts.append(f"<strong>Evaluation response:</strong> {detail['evaluation_response']} (score={detail['score']:.3f})\n")
            html_parts.append("</div><br>")
        html_parts.append("</div>")
    # Finalise HTML
    html_parts.append("</body></html>")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(html_parts))
