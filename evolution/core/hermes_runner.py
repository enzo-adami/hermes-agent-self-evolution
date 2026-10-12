"""Run one task through the real Hermes agent with a candidate skill installed.

The evaluation must exercise what users actually get: Hermes's own agent
loop, system prompt and tools (terminal, files, web, browser, ...). A bare
LLM call that only reads the skill text measures something else entirely.

Each run uses ``hermes chat -q ... --oneshot --format stream-json -s <skill>``
in a throwaway HERMES_HOME. The user's model, provider credentials and persona
are copied in from their real profile, so the run uses the same model and
auth as normal Hermes use, while the evaluation's sessions and memories never
touch that profile. Bundled-skill seeding is disabled so the only copy of the
skill under evaluation is the candidate.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Files carried over from the source profile: model/provider config, API keys,
# OAuth tokens, and the persona that shapes every answer.
_PROFILE_FILES = ("config.yaml", ".env", "auth.json", "SOUL.md")
_TRANSCRIPT_OUTPUT_CAP = 1500


def default_hermes_home() -> Path:
    """The profile Hermes itself would use: HERMES_HOME, else ~/.hermes."""
    return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes").expanduser()


@dataclass
class ToolCall:
    name: str
    input: Optional[dict] = None
    output: str = ""
    is_error: bool = False


@dataclass
class HermesRun:
    final_text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    exit_code: int = 0
    error: str = ""

    def transcript(self) -> str:
        """What the agent did, in order, for the reflection LM and the judge."""
        lines = []
        for i, call in enumerate(self.tool_calls, 1):
            args = json.dumps(call.input, ensure_ascii=False) if call.input is not None else ""
            output = call.output
            if len(output) > _TRANSCRIPT_OUTPUT_CAP:
                output = output[:_TRANSCRIPT_OUTPUT_CAP] + "..."
            status = " (error)" if call.is_error else ""
            lines.append(f"Tool call {i}: {call.name}({args}){status}\n  -> {output}")
        if not self.tool_calls:
            lines.append("No tool calls.")
        if self.error:
            lines.append(f"Run error (exit code {self.exit_code}): {self.error}")
        lines.append(f"Final answer:\n{self.final_text}")
        return "\n".join(lines)


def parse_stream_json(stdout: str) -> HermesRun:
    """Parse ``--format stream-json`` output; non-JSON lines (warnings) are skipped."""
    run = HermesRun()
    pending: dict[str, ToolCall] = {}
    saw_result = False
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        if kind == "tool_use":
            call = ToolCall(name=event.get("name", "unknown"), input=event.get("input"))
            run.tool_calls.append(call)
            pending[event.get("tool_call_id") or call.name] = call
        elif kind == "tool_result":
            key = event.get("tool_call_id") or event.get("name", "unknown")
            call = pending.pop(key, None)
            if call is None:
                call = ToolCall(name=event.get("name", "unknown"))
                run.tool_calls.append(call)
            call.output = str(event.get("output", ""))
            call.is_error = bool(event.get("is_error", False))
        elif kind == "result":
            saw_result = True
            run.final_text = str(event.get("text") or "")
            run.exit_code = int(event.get("exit_code") or 0)
            run.error = str(event.get("error") or "")
    if not saw_result:
        run.exit_code = run.exit_code or 1
        run.error = run.error or "Hermes produced no result event"
    return run


@dataclass
class HermesRunner:
    """Runs tasks through a Hermes installation.

    ``command`` is how to invoke Hermes, e.g. ``("hermes",)`` or
    ``("/path/to/venv/bin/python", "-m", "hermes_cli.main")``.
    """

    command: tuple[str, ...] = ("hermes",)
    source_home: Optional[Path] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    toolsets: Optional[str] = None
    max_turns: int = 20
    timeout: float = 600.0

    def _model_flags(self) -> list[str]:
        flags = []
        if self.model:
            flags += ["-m", self.model]
        if self.provider:
            flags += ["--provider", self.provider]
        return flags

    def build_command(self, skill_name: str, task: str) -> list[str]:
        cmd = [
            *self.command, "chat", "-q", task, "--oneshot", "--format", "stream-json",
            "-s", skill_name, "--max-turns", str(self.max_turns), *self._model_flags(),
        ]
        if self.toolsets:
            cmd += ["-t", self.toolsets]
        return cmd

    def _copy_profile(self, home: Path) -> None:
        source = self.source_home or default_hermes_home()
        home.mkdir(parents=True, exist_ok=True)
        for name in _PROFILE_FILES:
            if (source / name).is_file():
                shutil.copy2(source / name, home / name)
        (home / ".no-bundled-skills").touch()

    def prepare_home(self, home: Path, skill_name: str, skill_file: str, skill_dir: Optional[Path]) -> None:
        """Populate a throwaway HERMES_HOME holding the candidate skill."""
        self._copy_profile(home)
        dest = home / "skills" / skill_name
        if skill_dir is not None and skill_dir.is_dir():
            # Keep the skill's scripts/ and references/ next to the candidate text.
            shutil.copytree(skill_dir, dest)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "SKILL.md").write_text(skill_file)

    def run(self, skill_name: str, skill_file: str, task: str, skill_dir: Optional[Path] = None) -> HermesRun:
        """Run ``task`` through the agent with the candidate skill and the profile's tools."""
        with tempfile.TemporaryDirectory(prefix="hermes-eval-") as tmp:
            home = Path(tmp) / "home"
            self.prepare_home(home, skill_name, skill_file, skill_dir)
            return self._execute(self.build_command(skill_name, task), home, Path(tmp))

    def ask(self, prompt: str) -> HermesRun:
        """One answer from the profile's model, with no tools and no skill.

        Used where the pipeline needs a plain LLM (GEPA reflection, the judge,
        dataset generation) so it runs on the user's Hermes credentials. The
        prompt goes through a file: reflection prompts outgrow argv limits.
        """
        with tempfile.TemporaryDirectory(prefix="hermes-ask-") as tmp:
            home = Path(tmp) / "home"
            self._copy_profile(home)
            prompt_file = Path(tmp) / "prompt.txt"
            prompt_file.write_text(prompt)
            cmd = [
                *self.command, "chat", "--query-file", str(prompt_file), "--oneshot",
                "--format", "stream-json", "-t", "none", "--max-turns", "1", *self._model_flags(),
            ]
            return self._execute(cmd, home, Path(tmp))

    def _execute(self, cmd: list[str], home: Path, tmp: Path) -> HermesRun:
        workspace = tmp / "workspace"
        workspace.mkdir()
        try:
            proc = subprocess.run(
                cmd,
                cwd=workspace,
                env={**os.environ, "HERMES_HOME": str(home)},
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return HermesRun(exit_code=-1, error=f"timed out after {self.timeout:.0f}s")
        except OSError as exc:
            return HermesRun(exit_code=-1, error=f"could not start Hermes: {exc}")
        run = parse_stream_json(proc.stdout)
        if proc.returncode and not run.error:
            run.exit_code = proc.returncode
            run.error = proc.stderr.strip()[-500:] or f"exit code {proc.returncode}"
        return run
