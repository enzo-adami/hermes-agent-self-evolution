"""Wraps a SKILL.md file as a DSPy module for optimization.

The key abstraction: a skill file becomes a parameterized DSPy module
where the skill text is the optimizable parameter. GEPA can then
mutate the skill text and evaluate the results.
"""

import re
from pathlib import Path
from typing import Optional

import dspy

from evolution.core.hermes_runner import HermesRunner


def load_skill(skill_path: Path) -> dict:
    """Load a skill file and parse its frontmatter + body.

    Returns:
        {
            "path": Path,
            "raw": str (full file content),
            "frontmatter": str (YAML between --- markers),
            "body": str (markdown after frontmatter),
            "name": str,
            "description": str,
        }
    """
    raw = skill_path.read_text()

    # Parse YAML frontmatter
    frontmatter = ""
    body = raw
    if raw.strip().startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            frontmatter = parts[1].strip()
            body = parts[2].strip()

    # Extract name and description from frontmatter
    name = ""
    description = ""
    for line in frontmatter.split("\n"):
        if line.strip().startswith("name:"):
            name = line.split(":", 1)[1].strip().strip("'\"")
        elif line.strip().startswith("description:"):
            description = line.split(":", 1)[1].strip().strip("'\"")

    return {
        "path": skill_path,
        "raw": raw,
        "frontmatter": frontmatter,
        "body": body,
        "name": name,
        "description": description,
    }


def find_skill(skill_name: str, hermes_agent_path: Path) -> Optional[Path]:
    """Find a skill by name in the hermes-agent skills directory.

    Searches recursively for a SKILL.md in a directory matching the skill name.
    """
    skills_dir = hermes_agent_path / "skills"
    if not skills_dir.exists():
        return None

    # Direct match: skills/<category>/<skill_name>/SKILL.md
    for skill_md in skills_dir.rglob("SKILL.md"):
        if skill_md.parent.name == skill_name:
            return skill_md

    # Fuzzy match: check the name field in frontmatter
    for skill_md in skills_dir.rglob("SKILL.md"):
        try:
            content = skill_md.read_text()[:500]
            if f"name: {skill_name}" in content or f'name: "{skill_name}"' in content:
                return skill_md
        except Exception:
            continue

    return None


class SkillModule(dspy.Module):
    """A DSPy module that wraps a skill file for optimization.

    The skill text (body) is the parameter that GEPA optimizes. GEPA's
    candidate space is exactly the `signature.instructions` of the module's
    named predictors (see dspy.teleprompt.gepa), so the skill text must live
    in the signature instructions — not in an input field, which GEPA never
    mutates. On each forward pass, the module:
    1. Uses the skill text as the predictor's signature instructions
    2. Processes the task input
    3. Returns the agent's response

    `skill_text` is a read-through property over the predictor's current
    instructions, so after `optimizer.compile()` it reflects the evolved
    text rather than the original.
    """

    class TaskWithSkill(dspy.Signature):
        """Complete a task following the provided skill instructions."""

        task_input: str = dspy.InputField(desc="The task to complete")
        output: str = dspy.OutputField(desc="Your response following the skill instructions")

    def __init__(self, skill_text: str):
        super().__init__()
        signature = self.TaskWithSkill.with_instructions(skill_text)
        self.predictor = dspy.ChainOfThought(signature)

    @property
    def skill_text(self) -> str:
        """The current skill text — evolved instructions after optimization."""
        for _, predictor in self.named_predictors():
            return predictor.signature.instructions
        raise AttributeError("SkillModule has no predictors")

    def forward(self, task_input: str) -> dspy.Prediction:
        result = self.predictor(task_input=task_input)
        # Carry the candidate's own size so a metric can price growth during
        # optimization instead of discovering it at the final constraint gate.
        return dspy.Prediction(output=result.output, skill_chars=len(self.skill_text))


class HermesSkillModule(dspy.Module):
    """Evaluates the candidate skill by running the real Hermes agent on each task.

    The predictor is never called: it only holds the skill text as signature
    instructions, which is the component GEPA mutates. ``forward`` installs
    the current text as the skill, runs the task through Hermes with its real
    tools, and records a trace entry for the predictor so GEPA's reflection
    sees what Hermes actually did (tool calls and their results).
    """

    def __init__(
        self,
        skill_text: str,
        *,
        skill_name: str,
        frontmatter: str,
        runner: HermesRunner,
        skill_dir: Optional[Path] = None,
    ):
        super().__init__()
        self.predictor = dspy.Predict(SkillModule.TaskWithSkill.with_instructions(skill_text))
        self.skill_name = skill_name
        self.frontmatter = frontmatter
        self.runner = runner
        self.skill_dir = skill_dir

    @property
    def skill_text(self) -> str:
        """The current skill text — evolved instructions after optimization."""
        return self.predictor.signature.instructions

    def forward(self, task_input: str) -> dspy.Prediction:
        run = self.runner.run(
            skill_name=self.skill_name,
            skill_file=reassemble_skill(self.frontmatter, self.skill_text),
            task=task_input,
            skill_dir=self.skill_dir,
        )
        transcript = run.transcript()
        _record_trace(self.predictor, {"task_input": task_input}, dspy.Prediction(output=transcript))
        return dspy.Prediction(
            output=run.final_text,
            transcript=transcript,
            skill_chars=len(self.skill_text),
        )


def _record_trace(predictor: dspy.Predict, inputs: dict, prediction: dspy.Prediction) -> None:
    """Append a trace entry the way ``dspy.Predict`` does after its own LM call."""
    trace = dspy.settings.trace
    if trace is None or dspy.settings.max_trace_size <= 0:
        return
    if len(trace) >= dspy.settings.max_trace_size:
        trace.pop(0)
    trace.append((predictor, inputs, prediction))


def reassemble_skill(frontmatter: str, evolved_body: str) -> str:
    """Reassemble a skill file from frontmatter and evolved body.

    Preserves the original YAML frontmatter (name, description, metadata)
    and replaces only the body with the evolved version.
    """
    return f"---\n{frontmatter}\n---\n\n{evolved_body}\n"
