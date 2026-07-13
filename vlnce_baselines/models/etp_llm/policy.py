"""Navigation policies backed by LLM-derived cognitive maps."""

from habitat_baselines.common.baseline_registry import baseline_registry

from vlnce_baselines.models.etp_prior_gt.policy import PriorGTPolicy


@baseline_registry.register_policy
class LLMBoxesCurrentPolicy(PriorGTPolicy):
    cognitive_map_source = "llm_boxes"


@baseline_registry.register_policy
class LLMGridTry5Policy(PriorGTPolicy):
    navigation_architecture = "try5"
    cognitive_map_source = "llm_grid"
