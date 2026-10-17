from glmv_reward.configs.verifiers.counting import CountingVerifierConfig
from glmv_reward.verifiers import get_verifier_from_config


def test_distinct_configs_do_not_share_verifier_state():
    first_config = CountingVerifierConfig(strict_boxed_extraction=True,
                                          llm_temperature=0.1)
    second_config = CountingVerifierConfig(strict_boxed_extraction=False,
                                           llm_temperature=0.8)
    first = get_verifier_from_config(first_config, "cache-config-regression")
    second = get_verifier_from_config(second_config, "cache-config-regression")
    assert first is not second
    assert first.strict_boxed is True
    assert second.strict_boxed is False
    assert second.llm_temperature == 0.8


def test_equivalent_config_still_reuses_cached_verifier():
    first = get_verifier_from_config(CountingVerifierConfig(), "cache-reuse-regression")
    second = get_verifier_from_config(CountingVerifierConfig(), "cache-reuse-regression")
    assert first is second


def test_distinct_datasources_keep_separate_instances():
    config = CountingVerifierConfig()
    first = get_verifier_from_config(config, "cache-datasource-a")
    second = get_verifier_from_config(config, "cache-datasource-b")
    assert first is not second
