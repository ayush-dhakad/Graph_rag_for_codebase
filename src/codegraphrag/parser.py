from __future__ import annotations

from pathlib import Path

from tree_sitter_language_pack import configure, detect_language_from_path, get_parser

from .models import CodeGraph, CodeNode, Edge, stable_node_id

IGNORED_DIRECTORIES = {".git", ".venv", "node_modules", "dist", "build", "__pycache__", ".codegraphrag"}
DEFINITION_TYPES = {"function_definition", "function_declaration", "method_definition", "method_declaration", "class_definition", "class_declaration"}


class TreeSitterCodeParser:
    """Extract files, symbols, imports and direct calls from Tree-sitter ASTs."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        # Keeps downloaded grammar libraries out of the user's home directory.
        cache_dir = cache_dir or Path(".codegraphrag/tree-sitter")
        cache_dir.mkdir(parents=True, exist_ok=True)
        configure({"cache_dir": str(cache_dir)})

    def parse_repository(self, root: str | Path) -> CodeGraph:
        root = Path(root).resolve()
        graph = CodeGraph()
        for path in root.rglob("*"):
            if not path.is_file() or any(part in IGNORED_DIRECTORIES for part in path.parts):
                continue
            if detect_language_from_path(str(path)):
                self.parse_file(path, root, graph)
        self._link_calls(graph)
        return graph

    def parse_file(self, path: Path, root: Path, graph: CodeGraph) -> None:
        language = detect_language_from_path(str(path))
        if not language:
            return
        try:
            source = path.read_text(encoding="utf-8")
            tree = get_parser(language).parse(source.encode())
        except (UnicodeDecodeError, OSError, RuntimeError) as error:
            print(f"Skipping {path}: {error}")
            return
        relative = path.relative_to(root).as_posix()
        file_id = stable_node_id(relative, "file", relative, 1)
        graph.add_node(CodeNode(file_id, "file", relative, relative, 1, source.count("\n") + 1, source))
        self._walk(tree.root_node, source, relative, file_id, graph, None, language)

    def _walk(self, node, source: str, path: str, file_id: str, graph: CodeGraph, parent_id: str | None, language: str) -> None:
        current_parent = parent_id or file_id
        if node.type in DEFINITION_TYPES:
            name_node = node.child_by_field_name("name")
            name = self._text(name_node, source) if name_node else "<anonymous>"
            is_class = "class" in node.type
            kind = "class" if is_class else ("method" if parent_id and graph.nodes[parent_id].kind == "class" else "function")
            line = node.start_point.row + 1
            node_id = stable_node_id(path, kind, name, line)
            code = source[node.start_byte:node.end_byte]
            parent_name = graph.nodes[current_parent].qualified_name or graph.nodes[current_parent].name
            qualified = f"{parent_name}.{name}" if parent_id else name
            graph.add_node(CodeNode(node_id, kind, name, path, line, node.end_point.row + 1, code, qualified))
            graph.add_edge(Edge(current_parent, node_id, "CONTAINS"))
            current_parent = node_id
        if self._is_import(node.type, language):
            module = self._import_text(node, source, language)
            if module:
                import_id = stable_node_id(path, "import", module, node.start_point.row + 1)
                graph.add_node(CodeNode(import_id, "import", module, path, node.start_point.row + 1, node.end_point.row + 1, source[node.start_byte:node.end_byte]))
                graph.add_edge(Edge(current_parent, import_id, "IMPORTS"))
        for child in node.children:
            self._walk(child, source, path, file_id, graph, current_parent, language)

    @staticmethod
    def _text(node, source: str) -> str:
        return source[node.start_byte:node.end_byte].strip().strip("'\"")

    @staticmethod
    def _is_import(node_type: str, language: str) -> bool:
        return node_type in {"import_statement", "import_from_statement", "import_declaration", "namespace_import"}

    def _import_text(self, node, source: str, language: str) -> str | None:
        text = self._text(node, source)
        if language == "python":
            if text.startswith("from "):
                return text.split()[1]
            if text.startswith("import "):
                return text[7:].split(",")[0].strip().split(" as ")[0]
        if "from" in text:
            return text.rsplit("from", 1)[1].strip().strip(";'\"")
        return text.replace("import", "").strip().split()[0].strip(";'\"") or None

    def _link_calls(self, graph: CodeGraph) -> None:
        """Resolve simple identifier calls to known local functions/methods."""
        definitions = {node.name: node.id for node in graph.nodes.values() if node.kind in {"function", "method"}}
        for node in list(graph.nodes.values()):
            if node.kind not in {"function", "method"}:
                continue
            # AST-level parsing is retained above; call identifiers are deliberately conservative.
            import re
            for call_name in re.findall(r"(?<!def\s)\b([A-Za-z_]\w*)\s*\(", node.source):
                if call_name in definitions and definitions[call_name] != node.id:
                    graph.add_edge(Edge(node.id, definitions[call_name], "CALLS"))
