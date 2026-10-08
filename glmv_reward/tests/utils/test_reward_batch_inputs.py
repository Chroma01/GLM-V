from unittest.mock import patch

import pytest

from glmv_reward import RewardSystem


@pytest.fixture
def system(tmp_path):
    config = tmp_path / "reward.yaml"
    config.write_text(
        "datasource_reward_config_mapping:\n  default: counting\n"
        "reward_configs:\n  counting:\n    verifier_type: counting\n"
        "    enable_llm_judge_fallback: false\nenable_mix_verifier: false\n",
        encoding="utf-8",
    )
    return RewardSystem(config)


@pytest.mark.parametrize("field", ["answers", "gt_answers", "image_files",
                                   "uuids", "answer_lengths", "datasources"])
def test_mismatched_batch_is_rejected_before_processing(system, field):
    inputs = {"prompts": ["p1", "p2"], "answers": ["a1", "a2"],
              "gt_answers": ["g1", "g2"]}
    inputs[field] = [1] if field == "answer_lengths" else ["default"]
    with patch.object(system, "_process_single_item", return_value=(0.0, None, None)) as process:
        with pytest.raises(ValueError, match="length"):
            system.get_reward(**inputs)
        process.assert_not_called()


def test_empty_scalar_prompt_is_still_one_item(system):
    with patch.object(system, "_process_single_item", return_value=(1.0, "a", "g")):
        assert system.get_reward("", "a", "g") == [1.0]


def test_valid_batch_keeps_output_order(system):
    with patch.object(system, "_process_single_item", side_effect=[(0.0, "a1", "g1"),
                                                                  (1.0, "a2", "g2")]):
        assert system.get_reward(["p1", "p2"], ["a1", "a2"], ["g1", "g2"]) == [0.0, 1.0]


@pytest.mark.parametrize("extracted", [False, True])
def test_empty_batch_has_empty_outputs(system, extracted):
    result = system.get_reward([], [], [], return_extracted_answers=extracted)
    assert result == (([], [], []) if extracted else [])
