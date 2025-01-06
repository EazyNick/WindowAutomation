from .templetmatching import TemplateMatcher, ExactMatchStrategy
from .strategy.exactmatchstrategy import ExactMatchStrategy
from .strategy.multiscalematchingstrategy import MultiScaleMatchingStrategy

# ExactMatchStrategy를 사용하는 TemplateMatcher 인스턴스
exact_matcher = TemplateMatcher()
exact_matcher.set_strategy(ExactMatchStrategy())

# MultiScaleMatchingStrategy를 사용하는 TemplateMatcher 인스턴스
multi_scale_matcher = TemplateMatcher()
multi_scale_matcher.set_strategy(MultiScaleMatchingStrategy(scale_factors=[0.5, 1.0, 1.5, 2.0], threshold=0.9))

# __all__에 두 개의 인스턴스를 추가
__all__ = ["exact_matcher", "multi_scale_matcher"]
