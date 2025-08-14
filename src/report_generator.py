from __future__ import annotations

import base64
import io
import json
import datetime
import html
from typing import Dict, Any, List, Optional

import matplotlib.pyplot as plt


def _encode_fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return encoded


def _coerce_float(x: Any, default: float = 0.0) -> float:
    try:
        # Handle Score-like objects with get_value()
        if hasattr(x, "get_value"):
            x = x.get_value()
        return float(x)
    except Exception:
        return default


def _render_structured_entry(e: Dict[str, Any]) -> str:
    turn = e.get("turn")
    ap = html.escape(str(e.get("attack_prompt", "")))
    tr = html.escape(str(e.get("target_response", "")))
    er = html.escape(str(e.get("evaluation_response", "")))
    sc = e.get("score")
    sc_txt = f"{_coerce_float(sc):.3f}" if sc is not None else ""
    title = f"<strong>Turn {turn} - Attack prompt:</strong> " if turn is not None else "<strong>Attack prompt:</strong> "
    return (
        "<div class='transcript'>"
        f"{title}{ap}\n"
        f"<strong>Target response:</strong> {tr}\n"
        f"<strong>Evaluation response:</strong> {er}"
        f"{f' (score={sc_txt})' if sc_txt else ''}\n"
        "</div><br>"
    )


def _render_jailbreak_pass(jb: Dict[str, Any]) -> str:
    parts: List[str] = []
    # Header
    success = bool(jb.get("success", False))
    fs = _coerce_float(jb.get("final_score", 0.0))
    parts.append(f"<h4>Pass (Success: {success}, Final score: {fs:.3f})</h4>")

    # Prefer structured transcript if present & non-empty
    structured = jb.get("transcript")
    if isinstance(structured, list) and structured:
        for e in structured:
            if isinstance(e, dict):
                parts.append(_render_structured_entry(e))
        return "".join(parts)

    # Fallback: any raw text field
    raw = jb.get("transcript_text") or jb.get("conversation") or jb.get("conversation_text")
    if isinstance(raw, str) and raw.strip():
        parts.append(f"<div class='transcript'>{html.escape(raw)}</div><br>")
        return "".join(parts)

    # Nothing to show
    parts.append("<div class='transcript'><em>No transcript available.</em></div><br>")
    return "".join(parts)

def _render_bias_pass(b: Dict[str, Any]) -> str:
    parts: List[str] = []
    # Header
    success = bool(b.get("success", False))
    fs = _coerce_float(b.get("final_score", 0.0))
    parts.append(f"<h4>Pass (Success: {success}, Final score: {fs:.3f})</h4>")

    # Prefer structured transcript if present & non-empty
    structured = b.get("transcript")
    if isinstance(structured, list) and structured:
        for e in structured:
            if isinstance(e, dict):
                parts.append(_render_structured_entry(e))
        return "".join(parts)

    # Fallback: any raw text field
    raw = b.get("transcript_text") or b.get("conversation") or b.get("conversation_text")
    if isinstance(raw, str) and raw.strip():
        parts.append(f"<div class='transcript'>{html.escape(raw)}</div><br>")
        return "".join(parts)

    # Nothing to show
    parts.append("<div class='transcript'><em>No transcript available.</em></div><br>")
    return "".join(parts)


def generate_report(results: Dict[str, Any], output_path: str) -> None:
    jailbreak_scores = results.get("jailbreak_scores", [])
    bias_scores = results.get("bias_scores", [])
    injection_score = _coerce_float(results.get("injection_result", {}).get("score", 0))
    final_score = _coerce_float(results.get("final_score", 0))

    # Chart values (coerced and bounded to [0,10] visually)
    test_names = ["Jailbreak", "Bias", "Injection"]
    test_values = [
        _coerce_float(results.get("avg_jailbreak_score", 0)),
        _coerce_float(results.get("avg_bias_score", 0)),
        injection_score,
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(test_names, test_values)
    ax.set_ylim(0, 10)
    ax.set_ylabel("Score (1–10)")
    ax.set_title("")
    for bar, value in zip(bars, test_values):
        ax.annotate(f"{value:.2f}", xy=(bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, color='black')
    chart_img = _encode_fig_to_base64(fig)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_parts: List[str] = []
    html_parts.append(
        "<html><head><meta charset='UTF-8'><title>ModelTest Report</title>"
        "<style>"
        "body{font-family:Arial, sans-serif;margin:40px;background-color:#f9f9f9;}"
        ".card{background-color:white;border-radius:8px;padding:20px;margin-bottom:20px;box-shadow:0 2px 4px rgba(0,0,0,0.1);}"
        ".header{font-size:24px;font-weight:bold;margin-bottom:10px;}"
        ".score-table{width:100%;border-collapse:collapse;margin-top:10px;}"
        ".score-table th, .score-table td{border:1px solid #ddd;padding:8px;text-align:left;}"
        ".score-table th{background-color:#f2f2f2;}"
        ".transcript{font-family:monospace;white-space:pre-wrap;background:#fafafa;border:1px solid #eee;padding:10px;}"
        "</style></head><body>"
    )

    # Header card
    html_parts.append(
        f"<div class='card'><div class='header'>Model Evaluation Report</div>"
        f"<p><strong>Generated:</strong> {timestamp}</p>"
        f"<p><strong>Final Score:</strong> {final_score:.2f} / 10</p>"
        "<p>This report summarises the results of running jailbreak, bias and injection tests against the target model. "
        "Scores are higher when the model is more robust to the corresponding vulnerability.</p></div>"
    )

    # Chart card
    html_parts.append(
        f"<div class='card'><div class='header'>Score Summary</div>"
        f"<img src='data:image/png;base64,{chart_img}' alt='Score chart' style='width:100%;max-width:600px;'></div>"
    )

    # Detailed scores card (robust to missing fields)
    mean_eval_jb = 0.0
    mean_eval_b = 0.0
    if results.get("jailbreak_details"):
        # try to pull a 0-1 eval if stored; otherwise fallback to 1 - (score/10)
        last = results["jailbreak_details"][-1]
        raw_eval = last.get("mean_evaluation") or last.get("final_evaluation")
        if raw_eval is None:
            avg10 = _coerce_float(results.get("avg_jailbreak_score", 0))
            mean_eval_jb = max(0.0, min(1.0, 1.0 - (avg10 / 10.0)))
        else:
            mean_eval_jb = _coerce_float(raw_eval, 0.0)

    inj_mean = _coerce_float(results.get("injection_result", {}).get("mean_evaluation", 0))

    html_parts.append("<div class='card'><div class='header'>Detailed Scores</div>")
    html_parts.append("<table class='score-table'><tr><th>Test</th><th>Score</th><th>Mean evaluation (0-1)</th></tr>")
    html_parts.append(
        f"<tr><td>Jailbreak (average of {len(jailbreak_scores)} runs)</td>"
        f"<td>{_coerce_float(results.get('avg_jailbreak_score',0)):.2f}</td>"
        f"<td>{mean_eval_jb:.3f}</td></tr>"
    )
    html_parts.append(
        f"<tr><td>Jailbreak (average of {len(bias_scores)} runs)</td>"
        f"<td>{_coerce_float(results.get('avg_bias_score',0)):.2f}</td>"
        f"<td>{mean_eval_b:.3f}</td></tr>"
    )
    html_parts.append(
        f"<tr><td>Injection</td><td>{injection_score:.2f}</td><td>{inj_mean:.3f}</td></tr>"
    )
    html_parts.append("</table></div>")

    # Jailbreak transcripts card – supports structured or raw text per pass
    if results.get("jailbreak_details"):
        html_parts.append("<div class='card'><div class='header'>Jailbreak Transcripts</div>")
        for jb in results["jailbreak_details"]:
            html_parts.append(_render_jailbreak_pass(jb))
        html_parts.append("</div>")
        
    if results.get("bias_details"):
        html_parts.append("<div class='card'><div class='header'>Bias Transcripts</div>")
        for b in results["bias_details"]:
            html_parts.append(_render_bias_pass(b))
        html_parts.append("</div>")


    # Injection details
    if results.get("injection_result"):
        html_parts.append("<div class='card'><div class='header'>Injection Test Details</div>")
        for detail in results["injection_result"].get("details", []):
            html_parts.append("<div class='transcript'>")
            html_parts.append(f"<strong>Context:</strong> {html.escape(str(detail.get('context','')))}\n")
            html_parts.append(f"<strong>Attack instruction:</strong> {html.escape(str(detail.get('attack_instruction','')))}\n")
            html_parts.append(f"<strong>Target response:</strong> {html.escape(str(detail.get('target_response','')))}\n")
            evr = html.escape(str(detail.get('evaluation_response','')))
            sc = _coerce_float(detail.get('score'), None)
            html_parts.append(f"<strong>Evaluation response:</strong> {evr}{f' (score={sc:.3f})' if sc is not None else ''}\n")
            html_parts.append("</div><br>")
        html_parts.append("</div>")

    html_parts.append("</body></html>")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(html_parts))