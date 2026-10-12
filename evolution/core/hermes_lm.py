"""A DSPy language model answered by the user's own Hermes.

Passing ``hermes`` (or ``hermes/<model>``) wherever the pipeline asks for a
model name routes those calls through ``hermes chat`` with no tools: the same
model, provider and credentials as everyday Hermes use, so an evolution run
needs no separate API key. Sampling options such as temperature are not
forwarded; Hermes applies its own configuration.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Optional

import dspy

from evolution.core.hermes_runner import HermesRunner

HERMES_MODEL = "hermes"


def is_hermes_model(model: Optional[str]) -> bool:
    return bool(model) and (model == HERMES_MODEL or model.startswith(HERMES_MODEL + "/"))


def render_messages(messages: list[dict]) -> str:
    """Flatten chat messages into one query: system text first, then the turns."""
    if len(messages) == 1:
        return str(messages[0].get("content") or "")
    parts = []
    for message in messages:
        content = str(message.get("content") or "")
        role = message.get("role", "user")
        parts.append(content if role == "system" else f"[{role}]\n{content}")
    return "\n\n".join(parts)


class HermesLM(dspy.BaseLM):
    """``dspy.BaseLM`` whose completions come from ``HermesRunner.ask``."""

    def __init__(self, runner: HermesRunner):
        super().__init__(model=f"{HERMES_MODEL}/{runner.model or 'default'}", model_type="chat", cache=False)
        self.runner = runner

    def forward(self, prompt=None, messages=None, **kwargs):
        text = render_messages(messages) if messages else str(prompt or "")
        run = self.runner.ask(text)
        if run.error or not run.final_text.strip():
            raise RuntimeError(f"Hermes did not answer: {run.error or 'empty answer'}")
        message = SimpleNamespace(content=run.final_text, tool_calls=None, reasoning_content=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop", logprobs=None)],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            model=self.model,
        )


def make_lm(model: str, *, runner: Optional[HermesRunner] = None, **kwargs) -> dspy.BaseLM:
    """``dspy.LM(model, **kwargs)``, or a HermesLM for ``hermes`` / ``hermes/<model>``."""
    if not is_hermes_model(model):
        return dspy.LM(model, **kwargs)
    runner = runner or HermesRunner()
    _, _, override = model.partition("/")
    if override:
        runner = dataclasses.replace(runner, model=override)
    return HermesLM(runner)
