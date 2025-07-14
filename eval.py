import openai
from langsmith.wrappers import wrap_openai
from langsmith import traceable, Client
from langsmith.evaluation import evaluate

# Auto-trace LLM calls in-context
client = wrap_openai(openai.Client())

@traceable
def pipeline(user_input: str):
    result = client.chat.completions.create(
        messages=[{"role": "user", "content": user_input}],
        model="gpt-3.5-turbo"
    )
    return result.choices[0].message.content

pipeline("Hello, world!")

client = Client()

# Create a sample dataset
dataset_name = "Sample Dataset"
dataset = client.create_dataset(dataset_name, description="A sample dataset in LangSmith.")
client.create_examples(
    inputs=[
        {"postfix": "to LangSmith"},
        {"postfix": "to Evaluations in LangSmith"},
    ],
    outputs=[
        {"output": "Welcome to LangSmith"},
        {"output": "Welcome to Evaluations in LangSmith"},
    ],
    dataset_id=dataset.id,
)

# Define an exact match evaluator
def exact_match(run, example):
    return {"score": run.outputs["output"] == example.outputs["output"]}

# Evaluate using a simple function for testing
experiment_results = evaluate(
    lambda input: {"output": "Welcome " + input['postfix']},
    data=dataset_name,
    evaluators=[exact_match],
    experiment_prefix="sample-experiment",
    metadata={
        "version": "1.0.0",
        "revision_id": "beta"
    },
)
