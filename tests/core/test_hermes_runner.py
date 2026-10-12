"""HermesRunner / HermesSkillModule contracts, against a stand-in `hermes` command.

The stand-in reads the throwaway HERMES_HOME exactly as Hermes would and
answers in the ``--format stream-json`` protocol, so these tests pin what the
runner hands Hermes (skill text, profile files, flags) without a real install.
"""

import copy
import json
import sys
import textwrap
from pathlib import Path

import dspy
import pytest

from evolution.core.fitness import FitnessScore, SemanticSkillFitnessMetric
from evolution.core.hermes_runner import HermesRunner, parse_stream_json
from evolution.skills.skill_module import HermesSkillModule

FAKE_HERMES = textwrap.dedent(
    """
    import json, os, sys, time
    from pathlib import Path

    args = sys.argv[1:]
    home = Path(os.environ["HERMES_HOME"])
    skill = args[args.index("-s") + 1]
    task = args[args.index("-q") + 1]
    if task == "sleep":
        time.sleep(30)
    skill_text = (home / "skills" / skill / "SKILL.md").read_text()
    print("  warning line that is not JSON")
    print(json.dumps({"type": "system", "subtype": "init"}))
    print(json.dumps({"type": "tool_use", "name": "terminal", "tool_call_id": "c1",
                      "input": {"command": "echo hi"}}))
    print(json.dumps({"type": "tool_result", "name": "terminal", "tool_call_id": "c1",
                      "output": "hi", "is_error": False}))
    report = {
        "skill_text": skill_text,
        "profile_files": sorted(p.name for p in home.iterdir() if p.is_file()),
        "skill_files": sorted(p.name for p in (home / "skills" / skill).iterdir()),
        "cwd_is_home": Path.cwd() == home,
        "args": args,
    }
    print(json.dumps({"type": "result", "exit_code": 0, "text": json.dumps(report)}))
    """
)


@pytest.fixture
def fake_hermes(tmp_path):
    script = tmp_path / "fake_hermes.py"
    script.write_text(FAKE_HERMES)
    return (sys.executable, str(script))


@pytest.fixture
def profile(tmp_path):
    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text("model:\n  default: m\n")
    (home / ".env").write_text("KEY=1\n")
    (home / "state.db").write_text("must not be copied")
    return home


def _report(run):
    assert run.error == "", run.error
    return json.loads(run.final_text)


class TestParseStreamJson:
    def test_pairs_tool_calls_with_results_and_skips_noise(self):
        stdout = "\n".join([
            "not json at all",
            json.dumps({"type": "tool_use", "name": "terminal", "tool_call_id": "a", "input": {"command": "ls"}}),
            json.dumps({"type": "tool_result", "name": "terminal", "tool_call_id": "a",
                        "output": "file.txt", "is_error": True}),
            json.dumps({"type": "result", "exit_code": 0, "text": "done"}),
        ])
        run = parse_stream_json(stdout)
        assert run.final_text == "done"
        assert [(c.name, c.input, c.output, c.is_error) for c in run.tool_calls] == [
            ("terminal", {"command": "ls"}, "file.txt", True)
        ]
        assert "Tool call 1: terminal" in run.transcript()
        assert run.transcript().endswith("Final answer:\ndone")

    def test_missing_result_event_is_an_error(self):
        run = parse_stream_json(json.dumps({"type": "system", "subtype": "init"}))
        assert run.exit_code != 0
        assert "no result event" in run.error


class TestHermesRunner:
    def test_runs_the_candidate_skill_in_a_throwaway_profile(self, tmp_path, fake_hermes, profile):
        skill_dir = tmp_path / "original-skill"
        (skill_dir / "scripts").mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("original text")
        (skill_dir / "scripts" / "helper.py").write_text("print(1)")
        runner = HermesRunner(command=fake_hermes, source_home=profile, max_turns=7, toolsets="terminal")

        run = runner.run("demo", "---\nname: demo\n---\n\ncandidate text\n", "Do the task", skill_dir=skill_dir)
        report = _report(run)

        assert "candidate text" in report["skill_text"]
        assert "original text" not in report["skill_text"]
        assert report["skill_files"] == ["SKILL.md", "scripts"]
        assert report["profile_files"] == [".env", ".no-bundled-skills", "config.yaml"]
        assert report["cwd_is_home"] is False
        args = report["args"]
        assert args[:2] == ["chat", "-q"]
        assert {"--oneshot", "--format", "stream-json"} <= set(args)
        assert args[args.index("--max-turns") + 1] == "7"
        assert args[args.index("-t") + 1] == "terminal"
        assert [c.name for c in run.tool_calls] == ["terminal"]

    def test_leaves_the_source_profile_untouched_and_cleans_up(self, tmp_path, fake_hermes, profile, monkeypatch):
        before = sorted(p.name for p in profile.iterdir())
        made = []
        real_mkdtemp = __import__("tempfile").mkdtemp

        def tracking_mkdtemp(*args, **kwargs):
            made.append(Path(real_mkdtemp(*args, **kwargs)))
            return str(made[-1])

        monkeypatch.setattr("evolution.core.hermes_runner.tempfile.mkdtemp", tracking_mkdtemp)
        HermesRunner(command=fake_hermes, source_home=profile).run("demo", "x", "task")

        assert sorted(p.name for p in profile.iterdir()) == before
        assert made and not made[0].exists()

    def test_timeout_is_reported_not_raised(self, fake_hermes, profile):
        run = HermesRunner(command=fake_hermes, source_home=profile, timeout=1).run("demo", "x", "sleep")
        assert run.exit_code == -1
        assert "timed out" in run.error

    def test_missing_command_is_reported_not_raised(self, profile):
        run = HermesRunner(command=("/nonexistent/hermes",), source_home=profile).run("demo", "x", "task")
        assert run.exit_code == -1
        assert "could not start Hermes" in run.error


class TestHermesSkillModule:
    def _module(self, fake_hermes, profile):
        return HermesSkillModule(
            "baseline body",
            skill_name="demo",
            frontmatter="name: demo\ndescription: d",
            runner=HermesRunner(command=fake_hermes, source_home=profile),
        )

    def test_runs_the_current_instructions_and_traces_the_predictor(self, fake_hermes, profile):
        module = self._module(fake_hermes, profile)
        # GEPA proposes a candidate by rewriting the predictor's instructions on a copy.
        candidate = copy.deepcopy(module)
        for _, predictor in candidate.named_predictors():
            predictor.signature = predictor.signature.with_instructions("evolved body")

        with dspy.context(trace=[]):
            prediction = candidate(task_input="Do the task")
            trace = dspy.settings.trace

        assert "evolved body" in json.loads(prediction.output)["skill_text"]
        assert candidate.skill_text == "evolved body"
        assert module.skill_text == "baseline body"
        assert "Tool call 1: terminal" in prediction.transcript
        assert len(trace) == 1
        predictor, inputs, traced = trace[0]
        assert predictor is candidate.predictor
        assert inputs == {"task_input": "Do the task"}
        assert traced.output == prediction.transcript


class TestSemanticMetricSeesTranscript:
    def test_judge_receives_the_transcript_when_present(self):
        seen = {}

        class Judge:
            def score(self, **kwargs):
                seen.update(kwargs)
                return FitnessScore(correctness=1, procedure_following=1, epistemic_calibration=1,
                                    metric_integrity=1, conciseness=1, feedback="ok")

        metric = SemanticSkillFitnessMetric(Judge())
        example = dspy.Example(task_input="Check the build", expected_behavior="Run the tests")
        metric(example, dspy.Prediction(output="All green.", transcript="Tool call 1: terminal(...)\nFinal answer:\nAll green."))
        assert seen["agent_output"].startswith("Tool call 1: terminal")

        metric(example, dspy.Prediction(output="All green."))
        assert seen["agent_output"] == "All green."
