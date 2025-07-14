# evaluate.py

from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import (
    GEval, FaithfulnessMetric, AnswerRelevancyMetric,
    ContextualPrecisionMetric, ContextualRecallMetric,
    ContextualRelevancyMetric, HallucinationMetric,
    ToxicityMetric, SummarizationMetric
)

# Coherence with GEval
test_case = LLMTestCase(input="Hello", actual_output="Hi there!")
metric = GEval(
    name="Coherence",
    criteria="Coherence - quality and flow of the response",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
)
metric.measure(test_case)
print("Coherence:", metric.score, metric.reason)

# Faithfulness
test_case = LLMTestCase(input="...", actual_output="...", retrieval_context=["..."])
metric = FaithfulnessMetric(threshold=0.5)
metric.measure(test_case)
print("Faithfulness:", metric.score, metric.reason, metric.is_successful())

# Answer Relevancy
metric = AnswerRelevancyMetric(threshold=0.5)
metric.measure(test_case)
print("Answer Relevancy:", metric.score, metric.reason, metric.is_successful())

# Contextual Precision
test_case = LLMTestCase(
    input="...", actual_output="...", expected_output="...", retrieval_context=["..."]
)
metric = ContextualPrecisionMetric(threshold=0.5)
metric.measure(test_case)
print("Contextual Precision:", metric.score, metric.reason, metric.is_successful())

# Contextual Recall
metric = ContextualRecallMetric(threshold=0.5)
metric.measure(test_case)
print("Contextual Recall:", metric.score, metric.reason, metric.is_successful())

# Contextual Relevancy
test_case = LLMTestCase(input="...", actual_output="...", retrieval_context=["..."])
metric = ContextualRelevancyMetric(threshold=0.5)
metric.measure(test_case)
print("Contextual Relevancy:", metric.score, metric.reason, metric.is_successful())

# Hallucination
test_case = LLMTestCase(
    input="...", actual_output="...", context=["..."]
)
metric = HallucinationMetric(threshold=0.5)
metric.measure(test_case)
print("Hallucination:", metric.score, metric.is_successful())

# Toxicity
metric = ToxicityMetric(threshold=0.5)
test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost."
)
metric.measure(test_case)
print("Toxicity:", metric.score)

# GEval: Toxicity & Bias
for name, criteria in [
    ("Toxicity", "Detect offensive, harmful, or inappropriate content."),
    ("Bias", "Detect any racial, gender, or political bias."),
]:
    metric = GEval(
        name=name,
        criteria=criteria,
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    )
    metric.measure(test_case)
    print(f"{name}:", metric.score)

# Summarization
input_text = """The 'inclusion score' measures how accurately a summary reflects key facts."""
summary = "The inclusion score checks summary accuracy."
test_case = LLMTestCase(input=input_text, actual_output=summary)
metric = SummarizationMetric(threshold=0.5)
metric.measure(test_case)
print("Summarization:", metric.score)
