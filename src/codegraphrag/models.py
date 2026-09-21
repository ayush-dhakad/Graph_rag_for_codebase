from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

NodeKind = Literal["file", "module", "class", "function", "method", "import"]


@dataclass(frozen=True)
class CodeNode:
    id: str
    kind: NodeKind
    name: str
    path: str
    start_line: int
    end_line: int
    source: str = ""
    qualified_name: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "CodeNode":
        return CodeNode(**data)


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    kind: Literal["CONTAINS", "IMPORTS", "CALLS", "INHERITS"]

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "Edge":
        return Edge(**data)


@dataclass
class CodeGraph:
    nodes: dict[str, CodeNode] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    def add_node(self, node: CodeNode) -> None:
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        if edge.source != edge.target and edge not in self.edges:
            self.edges.append(edge)

    def to_dict(self) -> dict:
        return {"nodes": [n.to_dict() for n in self.nodes.values()], "edges": [e.to_dict() for e in self.edges]}

    @staticmethod
    def from_dict(data: dict) -> "CodeGraph":
        return CodeGraph(
            nodes={node["id"]: CodeNode.from_dict(node) for node in data["nodes"]},
            edges=[Edge.from_dict(edge) for edge in data["edges"]],
        )


@dataclass(frozen=True)
class GraphQuery:
    intent: Literal["callers", "callees", "imports", "definitions", "class_members"]
    symbol: str
    limit: int = 12

    def to_cypher(self) -> tuple[str, dict[str, str | int]]:
        """Return parameterized, read-only Cypher for the query intent."""
        patterns = {
            "callers": "MATCH (caller)-[:CALLS]->(target) WHERE target.name = $symbol RETURN caller",
            "callees": "MATCH (source)-[:CALLS]->(callee) WHERE source.name = $symbol RETURN callee",
            "imports": "MATCH (source)-[:IMPORTS]->(module) WHERE source.name = $symbol OR source.path = $symbol RETURN module",
            "definitions": "MATCH (node) WHERE node.name = $symbol RETURN node",
            "class_members": "MATCH (klass)-[:CONTAINS]->(member) WHERE klass.name = $symbol RETURN member",
        }
        return patterns[self.intent] + " LIMIT $limit", {"symbol": self.symbol, "limit": self.limit}


@dataclass(frozen=True)
class ContextResult:
    query: GraphQuery
    cypher: str
    nodes: list[CodeNode]
    context: str
    tokens: int


def stable_node_id(path: str | Path, kind: str, name: str, line: int) -> str:
    return f"{Path(path).as_posix()}:{kind}:{name}:{line}"
