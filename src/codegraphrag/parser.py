from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tree_sitter_language_pack import configure, detect_language_from_path, get_parser

from .models import CodeGraph, CodeNode, Edge, stable_node_id

IGNORED_DIRECTORIES = {".git", ".venv", "node_modules", "dist", "build", "__pycache__", ".codegraphrag"}
DEFINITION_TYPES = {"function_definition", "function_declaration", "method_definition", "method_declaration", "class_definition", "class_declaration"}
CALL_TYPES = {"call", "call_expression", "method_invocation", "function_call_expression", "invocation_expression", "new_expression", "object_creation_expression"}
FUNCTION_TYPES = {"function", "method"}


@dataclass(frozen=True)
class _Call:
    scope: str
    reference: tuple[str, ...]
    position: int


@dataclass(frozen=True)
class _Binding:
    position: int
    # None marks a parameter or assignment whose target cannot be resolved.
    reference: tuple[str, ...] | None
    constructed: bool = False


class TreeSitterCodeParser:
    """Extract syntax with Tree-sitter and resolve statically identifiable calls.

    This is a structural index, not a type checker. Unknown receivers, dynamic
    dispatch, and ambiguous imports remain unresolved instead of producing a
    guessed CALLS edge. No regular expressions are used for source extraction.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        cache_dir = cache_dir or Path(".codegraphrag/tree-sitter")
        cache_dir.mkdir(parents=True, exist_ok=True)
        configure({"cache_dir": str(cache_dir)})
        self._reset()

    def _reset(self) -> None:
        self._calls: list[_Call] = []
        self._parents: dict[str, str] = {}
        self._imports: dict[str, dict[str, tuple[str, str | None]]] = {}
        self._bindings: dict[str, dict[str, list[_Binding]]] = {}

    def parse_repository(self, root: str | Path) -> CodeGraph:
        self._reset()
        root = Path(root).resolve()
        graph = CodeGraph()
        for path in sorted(root.rglob("*")):
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
            tree = get_parser(language).parse(source.encode("utf-8"))
        except (UnicodeDecodeError, OSError, RuntimeError) as error:
            print(f"Skipping {path}: {error}")
            return
        relative = path.relative_to(root).as_posix()
        file_id = stable_node_id(relative, "file", relative, 1)
        graph.add_node(CodeNode(file_id, "file", relative, relative, 1, source.count("\n") + 1, source))
        self._walk(tree.root_node, relative, file_id, graph, language)

    def _walk(self, node, path: str, scope: str, graph: CodeGraph, language: str) -> None:
        # Anonymous callbacks need their own graph identity to own body calls.
        # Until they are indexed, do not attribute their calls to an outer scope.
        if node.type in {"lambda", "arrow_function", "function_expression", "generator_function"}:
            return
        if node.type in DEFINITION_TYPES:
            name_node = node.child_by_field_name("name")
            name = self._text(name_node) if name_node is not None else "<anonymous>"
            parent = graph.nodes[scope]
            kind = "class" if "class" in node.type else ("method" if parent.kind == "class" else "function")
            line = node.start_point.row + 1
            node_id = stable_node_id(path, kind, name, line)
            parent_name = parent.qualified_name or parent.name
            qualified = f"{parent_name}.{name}"
            graph.add_node(CodeNode(node_id, kind, name, path, line, node.end_point.row + 1, self._source(node), qualified))
            graph.add_edge(Edge(scope, node_id, "CONTAINS"))
            self._parents[node_id] = scope
            # Defaults, decorators and base expressions are evaluated in the
            # enclosing scope; only the definition body gets the new scope.
            body = node.child_by_field_name("body")
            parameters = node.child_by_field_name("parameters")
            if parameters is not None:
                for parameter in parameters.named_children:
                    identifier = parameter if parameter.type == "identifier" else parameter.child_by_field_name("name")
                    if identifier is None:
                        identifier = parameter.child_by_field_name("pattern")
                    if identifier is not None and identifier.type == "identifier":
                        self._bind(node_id, self._text(identifier), -1, None)
            for child in node.named_children:
                self._walk(child, path, node_id if child == body else scope, graph, language)
            return

        if node.type in {"import_statement", "import_from_statement", "import_declaration", "namespace_import"}:
            self._record_imports(node, path, scope, graph, language)

        if node.type in {"assignment", "variable_declarator"}:
            left = self._field(node, "left", "name")
            right = self._field(node, "right", "value")
            if left is not None and left.type == "identifier":
                value = right
                constructed = value is not None and value.type in CALL_TYPES
                if constructed:
                    value = self._field(value, "function", "constructor")
                self._bind(scope, self._text(left), node.end_byte, self._reference(value), constructed)

        if node.type in CALL_TYPES and graph.nodes[scope].kind in FUNCTION_TYPES:
            target = self._field(node, "function", "constructor")
            reference = self._reference(target)
            if target is None:
                # Java-style method_invocation exposes object and name fields.
                name = node.child_by_field_name("name")
                receiver = self._field(node, "object", "expression")
                if name is not None:
                    prefix = self._reference(receiver) if receiver is not None else ()
                    reference = (*prefix, self._text(name)) if prefix is not None else None
            if reference:
                self._calls.append(_Call(scope, reference, node.start_byte))

        for child in node.named_children:
            self._walk(child, path, scope, graph, language)

    @staticmethod
    def _source(node) -> str:
        # Tree-sitter positions are byte offsets, not Python string offsets.
        return node.text.decode("utf-8")

    @classmethod
    def _text(cls, node) -> str:
        return cls._source(node).strip().strip("'\"")

    @staticmethod
    def _field(node, *names):
        for name in names:
            child = node.child_by_field_name(name)
            if child is not None:
                return child
        return None

    def _reference(self, node) -> tuple[str, ...] | None:
        if node is None:
            return None
        if node.type in {"identifier", "property_identifier", "type_identifier", "this", "self"}:
            return (self._text(node),)
        if node.type in {"attribute", "member_expression", "field_expression"}:
            receiver = self._field(node, "object", "value", "argument")
            member = self._field(node, "attribute", "property", "field")
            prefix = self._reference(receiver)
            if prefix and member is not None and member.type in {"identifier", "property_identifier", "field_identifier"}:
                return (*prefix, self._text(member))
        return None

    def _bind(self, scope: str, name: str, position: int, reference: tuple[str, ...] | None, constructed: bool = False) -> None:
        self._bindings.setdefault(scope, {}).setdefault(name, []).append(_Binding(position, reference, constructed))

    def _record_imports(self, node, path: str, scope: str, graph: CodeGraph, language: str) -> None:
        imports: list[tuple[str, str | None, str | None]] = []
        if language == "python":
            module = node.child_by_field_name("module_name")
            for index, child in enumerate(node.children):
                if node.field_name_for_child(index) != "name":
                    continue
                original = child.child_by_field_name("name") if child.type == "aliased_import" else child
                alias = child.child_by_field_name("alias")
                name = self._text(original)
                if module is not None:
                    imports.append((self._text(module), name, self._text(alias) if alias is not None else name))
                else:
                    imported_module = name if alias is not None else name.split(".")[0]
                    imports.append((name, None, self._text(alias) if alias is not None else imported_module))
        else:
            module = node.child_by_field_name("source")
            if module is not None:
                imports.append((self._text(module), None, None))
                # Named ES imports and namespace imports are explicit AST nodes.
                pending = list(node.named_children)
                while pending:
                    child = pending.pop()
                    if child.type == "import_specifier":
                        original = child.child_by_field_name("name")
                        alias = child.child_by_field_name("alias")
                        if original is not None:
                            name = self._text(original)
                            imports.append((self._text(module), name, self._text(alias) if alias is not None else name))
                    elif child.type == "namespace_import":
                        identifiers = [c for c in child.named_children if c.type == "identifier"]
                        if identifiers:
                            imports.append((self._text(module), None, self._text(identifiers[0])))
                    pending.extend(child.named_children)
        for module, symbol, alias in imports:
            import_id = stable_node_id(path, "import", module, node.start_point.row + 1)
            graph.add_node(CodeNode(import_id, "import", module, path, node.start_point.row + 1, node.end_point.row + 1, self._source(node)))
            graph.add_edge(Edge(scope, import_id, "IMPORTS"))
            if alias:
                # `import package.module` binds package, not package.module.
                bound_module = module if symbol or alias != module.split(".")[0] else alias
                self._imports.setdefault(scope, {})[alias] = (bound_module, symbol)

    @staticmethod
    def _module_files(module: str, source_path: str, graph: CodeGraph) -> set[str]:
        source = Path(source_path)
        if module.startswith("."):
            if module.startswith("./") or module.startswith("../"):
                raw = source.parent / module
            else:
                dots = len(module) - len(module.lstrip("."))
                base = source.parent
                for _ in range(dots - 1):
                    base = base.parent
                raw = base / module[dots:].replace(".", "/")
            # Normalize relative segments without consulting the filesystem.
            parts: list[str] = []
            for part in raw.parts:
                if part == ".." and parts and parts[-1] != "..":
                    parts.pop()
                elif part != ".":
                    parts.append(part)
            expected = "/".join(parts)
            relative = True
        else:
            expected = module.replace(".", "/")
            relative = False
        if Path(expected).suffix in {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
            expected = Path(expected).with_suffix("").as_posix()
        matches = set()
        for item in graph.nodes.values():
            if item.kind != "file":
                continue
            stem = Path(item.path).with_suffix("").as_posix()
            if Path(stem).name in {"__init__", "index"}:
                stem = Path(stem).parent.as_posix()
            if stem == expected or (not relative and stem.endswith("/" + expected)):
                matches.add(item.id)
        return matches

    def _resolve(self, scope: str, reference: tuple[str, ...], position: int, graph: CodeGraph, seen: set[tuple] | None = None) -> set[str]:
        seen = set() if seen is None else seen
        key = (scope, reference, position)
        if key in seen:
            return set()
        seen = seen | {key}
        first, *rest = reference
        current = scope
        targets: set[str] = set()
        while current:
            owner = graph.nodes[current]
            if first in {"self", "cls", "this"}:
                if owner.kind == "class":
                    targets = {current}
                    break
            elif owner.kind != "class":
                bindings = self._bindings.get(current, {}).get(first, [])
                previous = [binding for binding in bindings if binding.position < position]
                if previous:
                    binding = max(previous, key=lambda item: item.position)
                    targets = self._resolve(current, binding.reference, binding.position, graph, seen) if binding.reference else set()
                    if binding.constructed:
                        targets = {target for target in targets if graph.nodes[target].kind == "class"}
                    break
                imported = self._imports.get(current, {}).get(first)
                if imported:
                    module, symbol = imported
                    targets = self._module_files(module, owner.path, graph)
                    if symbol:
                        targets = self._members(targets, symbol, graph)
                    break
                targets = self._members({current}, first, graph)
                if targets:
                    break
            current = self._parents.get(current, "")
        for member in rest:
            targets = self._members(targets, member, graph)
        return targets

    @staticmethod
    def _members(owners: set[str], name: str, graph: CodeGraph) -> set[str]:
        return {edge.target for edge in graph.edges if edge.kind == "CONTAINS" and edge.source in owners and graph.nodes[edge.target].name == name}

    def _link_calls(self, graph: CodeGraph) -> None:
        for call in self._calls:
            targets = self._resolve(call.scope, call.reference, call.position, graph)
            # Ambiguous definitions must not silently select the last one parsed.
            if len(targets) == 1:
                target = next(iter(targets))
                if graph.nodes[target].kind in FUNCTION_TYPES | {"class"}:
                    graph.add_edge(Edge(call.scope, target, "CALLS"))
