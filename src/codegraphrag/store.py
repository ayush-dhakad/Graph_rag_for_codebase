from __future__ import annotations

import json
from pathlib import Path

from .models import CodeGraph, CodeNode, Edge, GraphQuery


class InMemoryGraphStore:
    """Portable graph store; persisted as JSON so Neo4j is optional during development."""
    def __init__(self, graph: CodeGraph | None = None, semantic_index: dict | None = None) -> None:
        self.graph = graph or CodeGraph()
        self.semantic_index = semantic_index

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = self.graph.to_dict()
        if self.semantic_index is not None:
            data["semantic_index"] = self.semantic_index
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "InMemoryGraphStore":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(CodeGraph.from_dict(data), data.get("semantic_index"))

    def execute(self, query: GraphQuery) -> list[CodeNode]:
        nodes = self.graph.nodes
        matching = lambda n: n.name == query.symbol or n.qualified_name.endswith("." + query.symbol) or n.path == query.symbol
        selected: set[str] = set()
        if query.intent == "definitions":
            selected = {n.id for n in nodes.values() if matching(n)}
        elif query.intent == "callers":
            targets = {n.id for n in nodes.values() if matching(n)}
            selected = {e.source for e in self.graph.edges if e.kind == "CALLS" and e.target in targets}
        elif query.intent == "callees":
            sources = {n.id for n in nodes.values() if matching(n)}
            selected = {e.target for e in self.graph.edges if e.kind == "CALLS" and e.source in sources}
        elif query.intent == "imports":
            sources = {n.id for n in nodes.values() if matching(n)}
            selected = {e.target for e in self.graph.edges if e.kind == "IMPORTS" and e.source in sources}
        elif query.intent == "class_members":
            classes = {n.id for n in nodes.values() if matching(n)}
            selected = {e.target for e in self.graph.edges if e.kind == "CONTAINS" and e.source in classes}
        return sorted((nodes[item] for item in selected), key=lambda n: (n.path, n.start_line))[: query.limit]


class Neo4jGraphStore:
    """Neo4j adapter. Use `write_graph` when a Neo4j/Memgraph endpoint is available."""
    def __init__(self, uri: str, username: str, password: str) -> None:
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self) -> None:
        self.driver.close()

    def write_graph(self, graph: CodeGraph) -> None:
        with self.driver.session() as session:
            session.run("MATCH (n:Code) DETACH DELETE n")
            session.run("UNWIND $nodes AS n MERGE (x:Code {id:n.id}) SET x += n", nodes=[n.to_dict() for n in graph.nodes.values()])
            # Relationship labels are fixed by our model, never sourced from an LLM/user.
            for relation in ("CONTAINS", "IMPORTS", "CALLS", "INHERITS"):
                edges = [edge.to_dict() for edge in graph.edges if edge.kind == relation]
                if edges:
                    session.run(
                        f"UNWIND $edges AS e MATCH (a:Code {{id:e.source}}), (b:Code {{id:e.target}}) MERGE (a)-[:{relation}]->(b)",
                        edges=edges,
                    )

    def execute(self, query: GraphQuery) -> list[CodeNode]:
        cypher, params = query.to_cypher()
        if not cypher.lstrip().upper().startswith("MATCH"):
            raise ValueError("Only read-only MATCH queries are allowed")
        with self.driver.session() as session:
            records = session.run(cypher, **params)
            return [CodeNode.from_dict(dict(record[0])) for record in records]
