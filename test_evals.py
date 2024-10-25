from deepeval.metrics import GEval, HallucinationMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import assert_test

def test_hallucination():
    hallucination_metric = HallucinationMetric(minimum_score=
0.5
)
    test_case = LLMTestCase(
    	input="What if these shoes don't fit?", 
      actual_output="We offer a 30-day full refund at no extra costs.", 
      context=["All customers are eligible for a 30 day full refund at no extra costs."]
     )
    assert_test(test_case, [hallucination_metric])

def test_relevancy():
    answer_relevancy_metric = AnswerRelevancyMetric(minimum_score=
0.5
)
    test_case = LLMTestCase(
    	input="What does your company do?", 
      actual_output="Our company specializes in cloud computing"
     )
    assert_test(test_case, [relevancy_metric])
    
def test_humor():
    funny_metric = GEval(
    		name="Humor",
        criteria="How funny it is",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT]
    )
    test_case = LLMTestCase(
    	input="Write me something funny related to programming", 
      actual_output="Why did the programmer quit his job? Because he didn't get arrays!"
    )
    assert_test(test_case, [funny_metric])
