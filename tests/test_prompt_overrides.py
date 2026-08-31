"""Tests for per-run prompt overrides (26Q4-EVO-01).

The override hook is the injection point the promptbreeder loop uses to put a
bred prompt in front of agents for one run. Two properties are load-bearing:
an override must actually reach `get_prompt`, and it must leave a provenance
trail (versions + source) that lands in run metadata — an unrecorded bred
prompt would make its run's fitness unattributable.
"""

import json
from pathlib import Path

import pytest

from lms.prompts import (
    CURRENT_PROMPTS,
    active_overrides,
    apply_prompt_overrides,
    clear_prompt_overrides,
    get_all_versions,
    get_prompt,
    load_prompt_overrides,
)


@pytest.fixture(autouse=True)
def restore_prompts():
    """Every test starts and ends with the base prompts."""
    clear_prompt_overrides()
    yield
    clear_prompt_overrides()


class TestApplyPromptOverrides:
    def test_override_reaches_get_prompt(self):
        apply_prompt_overrides({"agent_system": "You are a bred agent."})
        assert get_prompt("agent_system").content == "You are a bred agent."

    def test_version_names_base_and_content_hash(self):
        base_version = get_prompt("agent_system").version
        apply_prompt_overrides({"agent_system": "bred content"})

        version = get_prompt("agent_system").version
        assert version.startswith(f"{base_version}+override.")
        # Same content, same hash — the version string is deterministic.
        clear_prompt_overrides()
        apply_prompt_overrides({"agent_system": "bred content"})
        assert get_prompt("agent_system").version == version

    def test_version_flows_into_get_all_versions(self):
        """Run metadata records prompt_versions; overrides must show there."""
        apply_prompt_overrides({"agent_system_goal": "bred goal prompt"})
        assert "+override." in get_all_versions()["agent_system_goal"]

    def test_untouched_prompts_stay_untouched(self):
        before = get_prompt("review_system")
        apply_prompt_overrides({"agent_system": "bred"})
        assert get_prompt("review_system") is before

    def test_unknown_name_fails_loudly(self):
        with pytest.raises(ValueError, match="agent_systm"):
            apply_prompt_overrides({"agent_systm": "typo'd name"})
        # And nothing was half-applied.
        assert active_overrides() is None

    def test_empty_content_fails_loudly(self):
        with pytest.raises(ValueError, match="non-empty"):
            apply_prompt_overrides({"agent_system": "   "})

    def test_provenance_records_source_and_versions(self):
        apply_prompt_overrides({"agent_system": "bred"}, source="genomes/g07.json")
        overrides = active_overrides()
        assert overrides is not None
        assert overrides["source"] == "genomes/g07.json"
        assert list(overrides["versions"]) == ["agent_system"]

    def test_no_overrides_means_none(self):
        assert active_overrides() is None


class TestClearPromptOverrides:
    def test_clear_restores_base_registry(self):
        base = dict(CURRENT_PROMPTS)
        apply_prompt_overrides({"agent_system": "bred", "agent_user": "bred too"})
        clear_prompt_overrides()
        assert CURRENT_PROMPTS == base
        assert active_overrides() is None


class TestLoadPromptOverrides:
    def test_round_trip(self, tmp_path: Path):
        path = tmp_path / "genome.json"
        path.write_text(json.dumps({"agent_system": "bred from file"}))
        assert load_prompt_overrides(path) == {"agent_system": "bred from file"}

    def test_invalid_json_fails_loudly(self, tmp_path: Path):
        path = tmp_path / "genome.json"
        path.write_text("{not json")
        with pytest.raises(ValueError, match="not valid JSON"):
            load_prompt_overrides(path)

    def test_non_object_fails_loudly(self, tmp_path: Path):
        path = tmp_path / "genome.json"
        path.write_text(json.dumps(["a", "list"]))
        with pytest.raises(ValueError, match="JSON object"):
            load_prompt_overrides(path)

    def test_non_string_values_fail_loudly(self, tmp_path: Path):
        path = tmp_path / "genome.json"
        path.write_text(json.dumps({"agent_system": ["not", "a", "string"]}))
        with pytest.raises(ValueError, match="content strings"):
            load_prompt_overrides(path)


class TestCLIWiring:
    def test_parser_accepts_prompt_file(self):
        from lms.run import build_parser

        args = build_parser().parse_args(["--prompt-file", "genomes/g07.json"])
        assert args.prompt_file == "genomes/g07.json"

    def test_prompt_file_defaults_to_none(self):
        from lms.run import build_parser

        assert build_parser().parse_args([]).prompt_file is None
