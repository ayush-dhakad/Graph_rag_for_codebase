"""Compare a whole-repository prompt with a GraphRAG prompt using Gemini usage metadata."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from .gemini import GeminiClient, GeminiResult
from .pipeline import CodeGraphRAG


@dataclass(frozen=True)
class BenchmarkResult:
    question: str
    expected_text: str
    baseline: GeminiResult
    graph_rag: GeminiResult
    baseline_correct: bool
    graph_rag_correct: bool
    context_tokens: int
    total_token_reduction_percent: float

    def to_dict(self) -> dict:
        return asdict(self)


def _prompt(question: str, context: str) -> str:
    return f"""You are answering a codebase question. Use only the supplied source context. Be concise and name the exact function/class/file that supports your answer. If the answer is unavailable, say so.

Question: {question}

Source context:
```text
{context}
```"""


def _contains_expected(answer: str, expected: str) -> bool:
    """Compare expected phrases without formatting differences such as Markdown backticks."""
    normalise = lambda value: " ".join(re.findall(r"[a-z0-9_]+", value.lower()))
    return normalise(expected) in normalise(answer)


def run_benchmark(
    repository: str | Path,
    question: str,
    expected_text: str,
    model: str = "gemini-3.6-flash",
    budget_tokens: int = 2500,
) -> BenchmarkResult:
    repository = Path(repository)
    with TemporaryDirectory(prefix="codegraphrag-benchmark-") as temp:
        rag = CodeGraphRAG(Path(temp) / "index.json")
        rag.index(repository)
        graph_context = rag.ask(question, budget_tokens)
    client = GeminiClient(model=model)
    baseline = client.generate(_prompt(question, rag.repository_source(repository)))
    graph = client.generate(_prompt(question, graph_context.context))
    baseline_correct = _contains_expected(baseline.text, expected_text)
    graph_correct = _contains_expected(graph.text, expected_text)
    reduction = 100 * (1 - graph.total_tokens / baseline.total_tokens) if baseline.total_tokens else 0.0
    return BenchmarkResult(
        question=question,
        expected_text=expected_text,
        baseline=baseline,
        graph_rag=graph,
        baseline_correct=baseline_correct,
        graph_rag_correct=graph_correct,
        context_tokens=graph_context.tokens,
        total_token_reduction_percent=round(reduction, 1),
    )
