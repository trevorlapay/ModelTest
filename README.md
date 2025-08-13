# ModelTest

ModelTest is a research‑oriented framework for evaluating the
robustness of large language models against several categories of
undesired behaviour: jailbreak vulnerability, bias/toxicity and prompt
injection. It orchestrates multiple agents — a **target model** (the
model under test), an **attacker model** (used to generate attack
prompts) and a **scorer model** (which judges how well the target
behaved) — to produce repeatable benchmarks. Results are aggregated
into a final score from 1–10, where 10 indicates the model is highly
resistant to adversarial behaviour and 1 indicates it is easily
compromised.

This repository is deliberately lightweight and focuses on core
functionality. It exposes a Python API for running the tests as well
as a command‑line interface for convenience. Reports are generated
as self‑contained HTML files with charts and transcripts.

## Features

* **Jailbreak test**: Implements a multi‑turn crescendo attack based on the
 PAIR methodology. An attacking model
  iteratively refines its prompts to coax the target model into
  completing a forbidden objective. A scoring model evaluates
  progress and halts the attack once the objective is met or a
  maximum depth is reached.
* **Bias test**: Sends a suite of prompts intended to elicit biased
  or harmful responses. Responses are scored for toxicity/bias on a
  0–1 scale and converted to a 1–10 robustness score.
* **Prompt injection test**: Embeds secrets or confidential data
  within a prompt and asks the model to reveal them. The scoring
  model measures how thoroughly the target ignored the malicious
  instructions.
* **Report generation**: Produces an "attractive" (ish) HTML report
  summarising scores, plotting bar charts and including transcripts
  of the jailbreak conversations.

## Installation

Install per requirements.txt. The big one is PyRIT. Thank you, uncle Microsoft,
for making this open source.

If you wish to evaluate a local model instead of using a remote API,
invoke the CLI with the `--local` flag and supply the `--target-model`
identifier corresponding to the model served by
[LiteLLM](https://github.com/BerriAI/litellm). A local server can be
started as follows:

```
litellm --model ollama/gpt-oss:20b --port 4000
```

ModelTest will then send requests to `http://127.0.0.1:4000/v1/chat/completions`.

This tool is designed around that convention IF you want to use a local model. 

## Usage

ModelTest is driven primarily through its command‑line interface. The
only configuration that must be provided via the environment is your
API keys for any remote models. All other options — number of
passes, attack objective, tree depth, whether to use a local model,
which models to invoke and even whether to let PyRIT handle
jailbreaking — are supplied via CLI flags.

Example usage on a Unix‑like shell:

```
# Provide API keys via environment (if calling remote models)
export TARGET_API_KEY="sk‑..."
export SCORER_API_KEY="sk‑..."
export ATTACKER_API_KEY="sk‑..."

# Run the tests with custom settings
python run_tests.py \
    --output my_report.html \
    --passes 3 \
    --objective "Cause the model to disclose confidential information" \
    --depth 6 \
    --threshold 0.9 \
    --attacker-model gpt-3.5-turbo \
    --scorer-model gpt-4 \
    --target-model gpt-3.5-turbo \
    --verbose
```

After the script completes, open `my_report.html` in your browser to
explore the results. The HTML file is fully self‑contained.

## Configuration reference

Only a handful of environment variables are recognised by ModelTest. They
contain credentials and thus are separated from the rest of the
configuration, which is passed via the CLI:

| Variable           | Default | Description |
|--------------------|---------|-------------|
| `TARGET_API_KEY`   | `""` | API key for the target model (remote only - not needed for local) |
| `SCORER_API_KEY`   | `""` | API key for the scoring model |
| `ATTACKER_API_KEY` | `""` | API key for the attacking model |


