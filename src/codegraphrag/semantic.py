from __future__ import annotations

from dataclasses import replace
from math import sqrt

from .models import CodeGraph, CodeNode


EMBEDDING_MODEL = "text-embedding-3-small"
SEMANTIC_LIMIT = 5
BATCH_SIZE = 64
EXCERPT_TOKENS = 1500


def _client():
    from openai import OpenAI

    return OpenAI()


def _embed(client, model: str, inputs: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=model, input=inputs)
    vectors = []
    for item in sorted(response.data, key=lambda item: item.index):
        norm = sqrt(sum(value * value for value in item.embedding))
        vectors.append([value / norm for value in item.embedding])
    return vectors


def _excerpt(node: CodeNode, encoding) -> CodeNode:
    tokens = encoding.encode_ordinary(node.source)
    if len(tokens) <= EXCERPT_TOKENS:
        return node
    source = encoding.decode(tokens[:EXCERPT_TOKENS], errors="ignore")
    return replace(node, source=source, end_line=node.start_line + source.count("\n"))


def build_semantic_index(graph: CodeGraph) -> dict:
    import tiktoken

    encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    symbols = [
        node for node in graph.nodes.values()
        if node.kind in {"function", "method", "class"} and node.source.strip()
    ]
    symbol_paths = {node.path for node in symbols}
    candidates = symbols + [
        node for node in graph.nodes.values()
        if node.kind == "file" and node.path not in symbol_paths and node.source.strip()
    ]
    vectors = {}
    if candidates:
        client = _client()
        for start in range(0, len(candidates), BATCH_SIZE):
            batch = candidates[start:start + BATCH_SIZE]
            inputs = [
                f"{node.kind} {node.qualified_name or node.name}\nPath: {node.path}\n"
                f"{_excerpt(node, encoding).source}"
                for node in batch
            ]
            embeddings = _embed(client, EMBEDDING_MODEL, inputs)
            vectors.update((node.id, vector) for node, vector in zip(batch, embeddings, strict=True))
    return {"model": EMBEDDING_MODEL, "vectors": vectors}


def semantic_search(graph: CodeGraph, index: dict, question: str) -> list[CodeNode]:
    import tiktoken

    vectors = index["vectors"]
    if not vectors:
        return []
    model = index["model"]
    encoding = tiktoken.encoding_for_model(model)
    query = encoding.encode_ordinary(question)
    if not question.strip():
        raise ValueError("Enter a question to search the index.")
    query_text = encoding.decode(query[:EXCERPT_TOKENS], errors="ignore")
    (query_vector,) = _embed(_client(), model, [query_text])
    ranked = sorted(
        vectors,
        key=lambda node_id: -sum(
            left * right for left, right in zip(query_vector, vectors[node_id], strict=True)
        ),
    )
    seeds = ranked[:SEMANTIC_LIMIT]
    by_source: dict[str, list[str]] = {}
    for edge in graph.edges:
        if edge.kind in {"CALLS", "IMPORTS", "CONTAINS"} and edge.source in seeds:
            by_source.setdefault(edge.source, []).append(edge.target)
    ordered = dict.fromkeys(seeds)
    for seed in seeds:
        ordered.update(dict.fromkeys(by_source.get(seed, ())))
    return [_excerpt(graph.nodes[node_id], encoding) for node_id in ordered]
