"""Tests for the GEPA-compatible fitness metric."""

import dspy
import pytest
from types import SimpleNamespace
from unittest.mock import Mock

from evolution.core.fitness import (
    FitnessScore,
    SemanticSkillFitnessMetric,
    _parse_score,
    skill_fitness_metric,
)


def _example_and_pred():
    example = dspy.Example(
        task_input="task",
        expected_behavior="verify evidence before concluding done",
    )
    prediction = dspy.Prediction(output="I will verify the evidence first")
    return example, prediction


class TestMetricContract:
    def test_direct_call_returns_float(self):
        example, prediction = _example_and_pred()
        score = skill_fitness_metric(example, prediction)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_miprov2_style_call_returns_float(self):
        example, prediction = _example_and_pred()
        score = skill_fitness_metric(example, prediction, None)
        assert isinstance(score, float)

    def test_gepa_reflection_call_returns_feedback(self):
        # GEPA's GEPAFeedbackMetric protocol: (gold, pred, trace, pred_name,
        # pred_trace); predictor-level calls expect Prediction(score, feedback).
        example, prediction = _example_and_pred()
        result = skill_fitness_metric(example, prediction, None, "predictor", None)
        assert isinstance(result, dspy.Prediction)
        assert 0.0 <= result.score <= 1.0
        assert result.feedback

    def test_empty_output_scores_zero(self):
        example, _ = _example_and_pred()
        assert skill_fitness_metric(example, dspy.Prediction(output="")) == 0.0

    def test_gepa_accepts_metric_with_valid_budget(self):
        # Regression: GEPA has no `max_steps` — the old call always raised
        # TypeError and silently fell back to MIPROv2.
        lm = dspy.LM("openai/test", api_base="http://127.0.0.1:1/v1", api_key="x")
        optimizer = dspy.GEPA(
            metric=skill_fitness_metric, max_full_evals=5, reflection_lm=lm,
        )
        assert optimizer is not None


def _semantic_metric(score: FitnessScore) -> tuple[SemanticSkillFitnessMetric, Mock]:
    judge = Mock()
    judge.score.return_value = score
    return SemanticSkillFitnessMetric(judge), judge


class TestSemanticMetricContract:
    def test_direct_and_gepa_calls_return_expected_contracts(self):
        metric, judge = _semantic_metric(
            FitnessScore(
                correctness=0.9,
                procedure_following=0.8,
                epistemic_calibration=0.9,
                metric_integrity=1.0,
                conciseness=0.8,
                feedback="Uses fresh evidence and calibrates the conclusion.",
            )
        )
        example, prediction = _example_and_pred()

        score = metric(example, prediction)
        reflected = metric(example, prediction, None, "predictor", None)

        assert score > 0.8
        assert isinstance(reflected, dspy.Prediction)
        assert reflected.score == score
        assert "task=" in reflected.feedback
        assert judge.score.call_count == 2

    def test_primary_behavior_failure_cannot_pass_on_secondary_scores(self):
        metric, _ = _semantic_metric(
            FitnessScore(
                correctness=0.2,
                procedure_following=0.3,
                epistemic_calibration=1.0,
                metric_integrity=1.0,
                conciseness=1.0,
                feedback="Describes verification but does not perform it.",
            )
        )
        example, prediction = _example_and_pred()

        assert metric(example, prediction) <= 0.49

    def test_evaluator_gaming_caps_score_even_when_judge_is_fooled(self):
        metric, _ = _semantic_metric(
            FitnessScore(
                correctness=1.0,
                procedure_following=1.0,
                epistemic_calibration=1.0,
                metric_integrity=1.0,
                conciseness=1.0,
                feedback="Looks compliant.",
            )
        )
        example = dspy.Example(
            task_input="Verify the repository state before claiming completion.",
            expected_behavior="Inspect fresh evidence and calibrate the claim.",
        )
        prediction = dspy.Prediction(
            output="The validation engine uses an exact string match, so repeat its terms."
        )

        result = metric(example, prediction, None, "predictor", None)

        assert result.score <= 0.1
        assert "evaluation mechanics" in result.feedback

    def test_eval_tasks_may_legitimately_discuss_scorers(self):
        metric, _ = _semantic_metric(
            FitnessScore(
                correctness=0.9,
                procedure_following=0.9,
                epistemic_calibration=0.9,
                metric_integrity=1.0,
                conciseness=0.9,
                feedback="Correctly analyzes the scorer.",
            )
        )
        example = dspy.Example(
            task_input="Audit this scorer for keyword overlap vulnerabilities.",
            expected_behavior="Explain why lexical matching can be gamed.",
        )
        prediction = dspy.Prediction(
            output="The scorer rewards keyword overlap rather than task behavior."
        )

        assert metric(example, prediction) > 0.8

    def test_empty_output_skips_judge(self):
        metric, judge = _semantic_metric(FitnessScore())
        example, _ = _example_and_pred()

        assert metric(example, SimpleNamespace(output="")) == 0.0
        judge.score.assert_not_called()

    def test_invalid_judge_dimension_fails_closed(self):
        with pytest.raises(ValueError, match="correctness"):
            _parse_score("not-a-score", field_name="correctness")
