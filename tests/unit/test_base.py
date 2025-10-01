"""Unit tests for reward_hub/base.py"""

import pytest
from reward_hub.base import AggregationMethod, PRMResult

pytestmark = pytest.mark.unit


class TestAggregationMethod:
    """Test AggregationMethod enum"""

    def test_enum_values(self):
        """Verify all enum values are correct"""
        assert AggregationMethod.MODEL.value == "model"
        assert AggregationMethod.PRODUCT.value == "prod"
        assert AggregationMethod.MIN.value == "min"
        assert AggregationMethod.LAST.value == "last"

    def test_enum_string_conversion(self):
        """Test creating enum from string"""
        assert AggregationMethod("model") == AggregationMethod.MODEL
        assert AggregationMethod("prod") == AggregationMethod.PRODUCT


class TestPRMResult:
    """Test PRMResult class"""

    def test_aggregation_methods_work(self):
        """All aggregation methods produce a score"""
        scores = [0.9, 0.8, 0.7]

        result_prod = PRMResult(scores, AggregationMethod.PRODUCT)
        result_min = PRMResult(scores, AggregationMethod.MIN)
        result_last = PRMResult(scores, AggregationMethod.LAST)

        # Just verify they work and produce scores
        assert isinstance(result_prod.score, float)
        assert isinstance(result_min.score, float)
        assert isinstance(result_last.score, float)
        assert result_prod.step_scores == scores

    def test_model_aggregation_single_score(self):
        """MODEL aggregation requires single score"""
        scores = [0.85]
        result = PRMResult(scores, AggregationMethod.MODEL)
        assert result.score == 0.85

    def test_model_aggregation_multiple_scores_raises(self):
        """MODEL aggregation with >1 score should raise AssertionError"""
        scores = [0.9, 0.8, 0.7]
        with pytest.raises(AssertionError, match="model aggregate method should only have one step"):
            PRMResult(scores, AggregationMethod.MODEL)

    def test_string_aggregation_method(self):
        """Accepts string aggregation method and converts to enum"""
        scores = [0.9, 0.8, 0.7]
        result = PRMResult(scores, "prod")
        assert isinstance(result.score, float)

    def test_default_aggregation_is_last(self):
        """Default aggregation method is LAST"""
        scores = [0.9, 0.8, 0.7]
        result = PRMResult(scores)
        assert result.score == 0.7

    def test_properties_accessible(self):
        """All properties are accessible"""
        scores = [0.9, 0.8, 0.7]
        result = PRMResult(scores, AggregationMethod.PRODUCT)

        assert hasattr(result, 'step_scores')
        assert hasattr(result, 'product')
        assert hasattr(result, 'min')
        assert hasattr(result, 'last')
        assert hasattr(result, 'score')
