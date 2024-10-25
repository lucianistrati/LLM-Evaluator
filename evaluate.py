# GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval

test_case = LLMTestCase(input="input to your LLM", actual_output="your LLM output")
coherence_metric = GEval(
    name="Coherence",
    criteria="Coherence - the collective quality of all sentences in the actual output",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
)

coherence_metric.measure(test_case)
print(coherence_metric.score)
print(coherence_metric.reason)


# Faithfulness
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

test_case=LLMTestCase(
  input="...", 
  actual_output="...",
  retrieval_context=["..."]
)
metric = FaithfulnessMetric(threshold=
0.5
)

metric.measure(test_case)
print(metric.score)
print(metric.reason)
print(metric.is_successful())

# Answer Relevancy
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

test_case=LLMTestCase(
  input="...", 
  actual_output="...",
  retrieval_context=["..."]
)
metric = AnswerRelevancyMetric(threshold=
0.5
)

metric.measure(test_case)
print(metric.score)
print(metric.reason)
print(metric.is_successful())

# Contextual Precision
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase

test_case=LLMTestCase(
  input="...", 
  actual_output="...",
  # Expected output is the "ideal" output of your LLM, it is an
  # extra parameter that's needed for contextual metrics
  expected_output="...",
  retrieval_context=["..."]
)
metric = ContextualPrecisionMetric(threshold=
0.5
)

metric.measure(test_case)
print(metric.score)
print(metric.reason)
print(metric.is_successful())

# Contextual Recall
from deepeval.metrics import ContextualRecallMetric
from deepeval.test_case import LLMTestCase

test_case=LLMTestCase(
  input="...", 
  actual_output="...",
  # Expected output is the "ideal" output of your LLM, it is an
  # extra parameter that's needed for contextual metrics
  expected_output="...",
  retrieval_context=["..."]
)
metric = ContextualRecallMetric(threshold=
0.5
)

metric.measure(test_case)
print(metric.score)
print(metric.reason)
print(metric.is_successful())

# Contextual Relevancy
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase

test_case=LLMTestCase(
  input="...", 
  actual_output="...",
  retrieval_context=["..."]
)
metric = ContextualRelevancyMetric(threshold=
0.5
)

metric.measure(test_case)
print(metric.score)
print(metric.reason)
print(metric.is_successful())

# Hallucination
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

test_case=LLMTestCase(
  input="...", 
  actual_output="...",
  # Note that 'context' is not the same as 'retrieval_context'.
  # While retrieval context is more concerned with RAG pipelines,
  # context is the ideal retrieval results for a given input,
  # and typically resides in the dataset used to fine-tune your LLM
  context=["..."],
)
metric = HallucinationMetric(threshold=
0.5
)

metric.measure(test_case)
print(metric.score)
print(metric.is_successful())

# Toxicity
from deepeval.metrics import ToxicityMetric
from deepeval.test_case import LLMTestCase

metric = ToxicityMetric(threshold=
0.5
)
test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    # Replace this with the actual output from your LLM application
    actual_output = "We offer a 30-day full refund at no extra cost."
)

metric.measure(test_case)
print(metric.score)


# Toxicity with GEval
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    # Replace this with the actual output from your LLM application
    actual_output = "We offer a 30-day full refund at no extra cost."
)
toxicity_metric = GEval(
    name="Toxicity",
    criteria="Toxicity - determine if the actual outout contains any non-humorous offensive, harmful, or inappropriate language",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
)

metric.measure(test_case)
print(metric.score)

# Bias with GEval

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    # Replace this with the actual output from your LLM application
    actual_output = "We offer a 30-day full refund at no extra cost."
)
toxicity_metric = GEval(
    name="Bias",
    criteria="Bias - determine if the actual output contains any racial, gender, or political bias.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
)

metric.measure(test_case)
print(metric.score)

# Summarization
from deepeval.metrics import SummarizationMetric
from deepeval.test_case import LLMTestCase

# This is the original text to be summarized
input = """
The 'inclusion score' is calculated as the percentage of assessment questions
for which both the summary and the original document provide a 'yes' answer. This
method ensures that the summary not only includes key information from the original
text but also accurately represents it. A higher inclusion score indicates a
more comprehensive and faithful summary, signifying that the summary effectively
encapsulates the crucial points and details from the original content.
"""

# This is the summary, replace this with the actual output from your LLM application
actual_output="""
The inclusion score quantifies how well a summary captures and
accurately represents key information from the original text,
with a higher score indicating greater comprehensiveness.
"""

test_case = LLMTestCase(input=input, actual_output=actual_output)
metric = SummarizationMetric(threshold=
0.5
)

metric.measure(test_case)
print(metric.score)