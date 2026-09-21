from __future__ import annotations

from pathlib import Path

from tree_sitter_language_pack import detect_language_from_path

from .context import assemble_context, count_tokens
from .models import ContextResult
from .parser import TreeSitterCodeParser
from .query import QueryTranslator
from .store import InMemoryGraphStore


class CodeGraphRAG:
    def __init__(self, index_path: str | Path = ".codegraphrag/index.json") -> None:
        self.index_path = Path(index_path)

    def index(self, repository: str | Path) -> InMemoryGraphStore:
        # Grammar binaries are a process-wide cache, independent of where an index is saved.
        graph = TreeSitterCodeParser().parse_repository(repository)
        store = InMemoryGraphStore(graph)
        store.save(self.index_path)
        return store

    def ask(self, question: str, budget_tokens: int = 2500) -> ContextResult:
        store = InMemoryGraphStore.load(self.index_path)
        query = QueryTranslator().translate(question)
        cypher, _ = query.to_cypher()
        nodes = store.execute(query)
        context, tokens = assemble_context(nodes, budget_tokens)
        return ContextResult(query, cypher, nodes, context, tokens)

    def token_report(self, repository: str | Path, question: str, budget_tokens: int = 2500) -> dict:
        full_code = "\n".join(
            p.read_text(encoding="utf-8", errors="ignore")
            for p in Path(repository).rglob("*")
            if p.is_file()
            and detect_language_from_path(str(p))
            and ".codegraphrag" not in p.parts
            and ".venv" not in p.parts
        )
        result = self.ask(question, budget_tokens)
        full_tokens = count_tokens(full_code)
        reduction = (1 - result.tokens / full_tokens) * 100 if full_tokens else 0
        return {"whole_repository_tokens": full_tokens, "graph_context_tokens": result.tokens, "saved_tokens": max(0, full_tokens - result.tokens), "reduction_percent": round(reduction, 1), "matched_nodes": len(result.nodes)}

    def repository_source(self, repository: str | Path) -> str:
        """Return only parseable source files, using the same scope as token_report."""
        return "\n".join(
            p.read_text(encoding="utf-8", errors="ignore")
            for p in Path(repository).rglob("*")
            if p.is_file()
            and detect_language_from_path(str(p))
            and ".codegraphrag" not in p.parts
            and ".venv" not in p.parts
        )
