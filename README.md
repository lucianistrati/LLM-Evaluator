# LLM-Evaluator

This repo provides a minimal setup for evaluating LLM-generated responses using both [LangSmith](https://smith.langchain.com/) and [DeepEval](https://github.com/confident-ai/deepeval).

## 📁 Contents

- `eval.py` – Evaluates a simple string-manipulation pipeline using LangSmith.
- `evaluate.py` – Applies various DeepEval metrics like Faithfulness, Hallucination, Toxicity, etc.
- `test_evals.py` – Defines unit-style tests with assertions using DeepEval.
- `requirements.txt` – *(optional)* Add dependencies like `openai`, `langsmith`, `deepeval`.

## 📦 Setup

```bash
pip install openai langsmith deepeval
````

Make sure you set the required environment variables:

* `OPENAI_API_KEY`
* LangSmith credentials (for evaluation logging)

## 🚀 Usage

Run LangSmith evaluation:

```bash
python eval.py
```

Test individual evaluation metrics:

```bash
python evaluate.py
```

Run tests (e.g. in CI or pytest):

```bash
pytest test_evals.py
```

## ✅ Supported Metrics

DeepEval metrics included:

* Faithfulness
* Hallucination
* Answer Relevancy
* Contextual Precision / Recall / Relevancy
* Toxicity
* Humor / Bias (via GEval)
* Summarization

## 🔗 Resources

* [LangSmith Docs](https://docs.smith.langchain.com/)
* [DeepEval GitHub](https://github.com/confident-ai/deepeval)
* [OpenAI API](https://platform.openai.com/docs)

---

**Author**: Lucian Istrati

License: MIT
