from pathlib import Path

from codegraphrag.pipeline import CodeGraphRAG
from codegraphrag.query import QueryTranslator


def test_rule_translator_handles_callers():
    query = QueryTranslator._rule_based("What calls 'authenticate'?")
    assert query.intent == "callers"
    assert query.symbol == "authenticate"


def test_indexes_and_retrieves_callers(tmp_path: Path):
    (tmp_path / "auth.py").write_text("def authenticate(user):\n    return user\n")
    (tmp_path / "api.py").write_text("from auth import authenticate\ndef login(user):\n    return authenticate(user)\n")
    rag = CodeGraphRAG(tmp_path / ".graph" / "index.json")
    store = rag.index(tmp_path)
    assert any(edge.kind == "CALLS" for edge in store.graph.edges)
    result = rag.ask("What calls 'authenticate'?")
    assert [node.name for node in result.nodes] == ["login"]
    assert "def login" in result.context
