"""
Model abstraction for interacting with both remote and local language models.

This module defines the `ModelCaller` class that wraps the underlying
HTTP or SDK calls needed to communicate with a language model. It
supports both remote (e.g. OpenAI) and local (LiteLLM) backends.

If the required SDK (such as `openai`) is not installed or the API call
fails for any reason, the call will gracefully fall back to a stub
implementation that simply echoes the last user prompt. This allows
the rest of the testing harness to be exercised even in environments
where external dependencies are unavailable. When running the tests
for real, ensure that the appropriate SDKs are installed and that
valid API keys are provided via environment variables.
"""

from __future__ import annotations

import json
import re
from typing import List, Dict, Any
import requests


class ModelCaller:
    """Wrapper for calling either a remote or local chat model.

    Parameters
    ----------
    model_name : str
        Identifier of the model to call. For OpenAI this might be
        'gpt-3.5-turbo'. For local models this should correspond to
        whichever model LiteLLM has been configured to serve.
    api_key : str
        API key used to authorise calls to remote providers. Ignored
        when `local` is True.
    local : bool, optional
        If True, calls a locally hosted model via HTTP rather than a
        remote provider. Defaults to False.
    """

    def __init__(self, model_name: str, api_key: str, local: bool = False) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.local = local

    def call(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Send a chat completion request to the configured model.

        Parameters
        ----------
        messages : list of dict
            Conversation history, formatted as a list of {"role": ..., "content": ...}
            dictionaries. At least one message with role 'user' should be
            provided.
        max_tokens : int, optional
            Maximum number of tokens to generate. This is passed through
            to the underlying provider. Defaults to 512.
        temperature : float, optional
            Sampling temperature for the model. Lower values make the
            output more deterministic. Defaults to 0.7.

        Returns
        -------
        str
            The content of the model's response.
        """

        # Sanity check: ensure there is at least one user message
        if not any(m.get("role") == "user" for m in messages):
            raise ValueError("At least one user message must be provided to call the model.")

        # Local model invocation via LiteLLM REST API
        if self.local:
            url = "http://127.0.0.1:4000/v1/chat/completions"
            payload: Dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            try:
                response = requests.post(url, json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except Exception:
                # On failure return a basic stubbed response
                last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
                return f"[local stub] {last_user}"

        # Remote provider (e.g. OpenAI). Try to use openai SDK if available
        try:
            import openai  # type: ignore[import]

            # Configure API key if provided
            if self.api_key:
                # openai>=1 uses a client, openai<1 uses global api_key attribute.
                try:
                    # Newer versions expect a client instance
                    client = openai.OpenAI(api_key=self.api_key)
                    completion = client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    return completion.choices[0].message.content
                except AttributeError:
                    # Fallback to legacy API
                    openai.api_key = self.api_key
                    completion = openai.ChatCompletion.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    # For openai>=1, the return object uses attribute access
                    try:
                        return completion.choices[0].message.content  # type: ignore[attr-defined]
                    except Exception:
                        return completion["choices"][0]["message"]["content"]
            else:
                # If no API key provided, still attempt call for public endpoints
                openai.api_key = None
                completion = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                try:
                    return completion.choices[0].message.content
                except Exception:
                    return completion["choices"][0]["message"]["content"]
        except Exception:
            # Either openai package is missing or call failed; return stub
            last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            return f"[remote stub] {last_user}"
