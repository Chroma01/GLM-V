import pytest

from glmv_reward.utils.text import find_boxed_content, find_boxed_content_with_boxed
from glmv_reward.verifiers.counting_verifier import CountingVerifier


@pytest.mark.parametrize("text", [r"\boxed{42", r"\boxed{\frac{1}{2}"])
def test_incomplete_latex_boxes_are_not_answers(text):
    assert find_boxed_content_with_boxed(text) == []
    assert find_boxed_content(text) == []


def test_complete_boxes_survive_a_truncated_trailing_box():
    assert find_boxed_content(r"\boxed{42} then \boxed{43") == ["42"]


@pytest.mark.parametrize("text, expected", [
    (r"\boxed{\frac{1}{2}}", [r"\frac{1}{2}"]),
    (r"\boxed{\boxed{42}}", [r"\boxed{42}"]),
    (r"\boxed{42} and \boxed{43}", ["42", "43"]),
    ("<|begin_of_box|>42<|end_of_box|>", ["42"]),
])
def test_complete_boxes_keep_existing_semantics(text, expected):
    assert find_boxed_content(text) == expected


def test_truncated_counting_response_is_not_rewarded():
    verifier = CountingVerifier(enable_llm_judge_fallback=False)
    answer = verifier.extract_answer(r"<think>Counted objects.</think>\boxed{42")
    assert answer is None
    assert verifier.judge(answer, "42") == 0.0
