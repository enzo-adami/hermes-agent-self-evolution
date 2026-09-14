"""The export gate must precede access to every selected importer."""
import json
from unittest.mock import Mock

import pytest
from click.testing import CliRunner
from evolution.core import external_importers as imports


@pytest.fixture
def readers(monkeypatch):
    spies = []
    for importer in [imports.ClaudeCodeImporter, imports.CopilotImporter, imports.HermesSessionImporter]:
        spy = Mock(return_value=[])
        monkeypatch.setattr(importer, "extract_messages", spy)
        spies.append(spy)
    monkeypatch.setattr(imports, "RelevanceFilter", Mock(side_effect=AssertionError("No scoring expected")))
    return spies


@pytest.mark.parametrize("sources", [
    ["claude-code", "copilot", "hermes"],
    ["copilot", "hermes", "claude-code"],
    ["hermes"],
])
def test_builder_missing_policy_reads_no_source(tmp_path, readers, sources):
    with pytest.raises(imports.SessionExportPolicyError, match="explicit allowlist"):
        imports.build_dataset_from_external(
            skill_name="demo", skill_text="Demo", sources=sources,
            output_path=tmp_path / "out", model="unused",
        )
    for reader in readers:
        reader.assert_not_called()
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.parametrize("source", ["all", "hermes"])
def test_cli_missing_policy_reads_no_source(monkeypatch, readers, dry_run, source):
    monkeypatch.setattr(imports, "_load_skill_text", lambda _: ("demo", "Demo"))
    args = ["--skill", "demo", "--source", source]
    if dry_run:
        args.append("--dry-run")
    result = CliRunner().invoke(imports.main, args)
    assert result.exit_code == 1
    assert "explicit allowlist" in result.output
    for reader in readers:
        reader.assert_not_called()


@pytest.mark.parametrize("dry_run", [True, False])
def test_cli_valid_policy_reaches_all_importers(tmp_path, monkeypatch, readers, dry_run):
    monkeypatch.setattr(imports, "_load_skill_text", lambda _: ("demo", "Demo"))
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({
        "allowed_session_ids": ["session-1"],
        "allowed_session_sources": ["cli"],
        "allowed_project_paths": [str(tmp_path)],
    }))
    args = ["--skill", "demo", "--source", "all", "--hermes-export-policy", str(policy)]
    if dry_run:
        args.append("--dry-run")
    result = CliRunner().invoke(imports.main, args)
    assert result.exit_code == 0, result.output
    readers[0].assert_called_once_with()
    readers[1].assert_called_once_with()
    readers[2].assert_called_once()
    parsed = readers[2].call_args.kwargs["export_policy"]
    assert parsed.allowed_session_ids == frozenset({"session-1"})


def test_cli_malformed_policy_reads_no_source(tmp_path, monkeypatch, readers):
    monkeypatch.setattr(imports, "_load_skill_text", lambda _: ("demo", "Demo"))
    policy = tmp_path / "policy.json"
    policy.write_text("{}")
    result = CliRunner().invoke(imports.main, [
        "--skill", "demo", "--source", "all", "--dry-run",
        "--hermes-export-policy", str(policy),
    ])
    assert result.exit_code == 1
    for reader in readers:
        reader.assert_not_called()


def test_non_hermes_source_needs_no_policy(monkeypatch, readers):
    monkeypatch.setattr(imports, "_load_skill_text", lambda _: ("demo", "Demo"))
    result = CliRunner().invoke(imports.main, [
        "--skill", "demo", "--source", "claude-code", "--dry-run",
    ])
    assert result.exit_code == 0
    readers[0].assert_called_once_with()
    readers[1].assert_not_called()
    readers[2].assert_not_called()
