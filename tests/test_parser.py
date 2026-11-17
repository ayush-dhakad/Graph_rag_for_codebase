"""Regression coverage for AST extraction and conservative call resolution.

These tests need the relevant Tree-sitter grammar in the configured cache.
"""
from pathlib import Path

from codegraphrag.parser import TreeSitterCodeParser


def build_graph(tmp_path: Path, files: dict[str, str]):
    for name, source in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return TreeSitterCodeParser().parse_repository(tmp_path)


def calls(graph):
    return {
        (graph.nodes[edge.source].qualified_name, graph.nodes[edge.target].qualified_name)
        for edge in graph.edges
        if edge.kind == "CALLS"
    }


def test_calls_ignore_strings_comments_and_preserve_unicode(tmp_path):
    graph = build_graph(tmp_path, {"app.py": '''# café 🐍
def target():
    return "héllo"

def fake():
    # target()
    return "target()"

def real():
    return target()
'''})
    assert calls(graph) == {("app.py.real", "app.py.target")}
    target = next(node for node in graph.nodes.values() if node.name == "target")
    assert target.source == 'def target():\n    return "héllo"'


def test_self_and_constructor_receivers_disambiguate_methods(tmp_path):
    graph = build_graph(tmp_path, {"app.py": '''class First:
    def execute(self):
        pass
    def run(self):
        return self.execute()

class Second:
    def execute(self):
        pass

def main():
    store = First()
    store.execute()
'''})
    edges = calls(graph)
    assert ("app.py.First.run", "app.py.First.execute") in edges
    assert ("app.py.main", "app.py.First.execute") in edges
    assert not any(target == "app.py.Second.execute" for _, target in edges)


def test_import_alias_resolves_the_correct_module(tmp_path):
    graph = build_graph(tmp_path, {
        "first.py": "def work():\n    pass\n",
        "second.py": "def work():\n    pass\n",
        "app.py": "from first import work as run\nimport second as other\ndef main():\n    run()\n    other.work()\n",
    })
    assert calls(graph) == {
        ("app.py.main", "first.py.work"),
        ("app.py.main", "second.py.work"),
    }


def test_nested_function_owns_its_calls(tmp_path):
    graph = build_graph(tmp_path, {"app.py": '''def target():
    pass

def outer():
    def inner():
        target()
    inner()
'''})
    assert calls(graph) == {
        ("app.py.outer.inner", "app.py.target"),
        ("app.py.outer", "app.py.outer.inner"),
    }


def test_unknown_receivers_and_shadowed_names_do_not_guess(tmp_path):
    graph = build_graph(tmp_path, {"app.py": '''def target():
    pass

class Store:
    def execute(self):
        pass

def run(target, unknown):
    target()
    unknown.execute()
    store = Store()
    store = unknown
    store.execute()
'''})
    assert calls(graph) == {("app.py.run", "app.py.Store")}


def test_relative_imported_class_and_multiple_imports(tmp_path):
    graph = build_graph(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/store.py": "class Store:\n    def execute(self):\n        pass\n",
        "pkg/app.py": "from .store import Store as LocalStore\nimport os, sys\ndef run():\n    store = LocalStore()\n    store.execute()\n",
    })
    assert ("pkg/app.py.run", "pkg/store.py.Store.execute") in calls(graph)
    imported = {node.name for node in graph.nodes.values() if node.kind == "import"}
    assert {".store", "os", "sys"} <= imported


def test_javascript_calls_use_syntax_and_this_receiver(tmp_path):
    graph = build_graph(tmp_path, {"app.js": '''function target() {}
class Store {
    execute() { target(); }
    run() { this.execute(); }
}
function fake() { return "target()"; }
'''})
    assert calls(graph) == {
        ("app.js.Store.execute", "app.js.target"),
        ("app.js.Store.run", "app.js.Store.execute"),
    }


def test_same_named_functions_in_other_files_are_not_guessed(tmp_path):
    graph = build_graph(tmp_path, {
        "first.py": "def target():\n    pass\n",
        "second.py": "def target():\n    pass\n",
        "app.py": "def run():\n    target()\n",
    })
    assert calls(graph) == set()


def test_lambda_body_is_not_attributed_to_enclosing_function(tmp_path):
    graph = build_graph(tmp_path, {"app.py": '''def target():
    pass

def outer():
    callback = lambda: target()
    return callback
'''})
    assert calls(graph) == set()


def test_named_es_import_with_explicit_extension(tmp_path):
    graph = build_graph(tmp_path, {
        "util.js": "export function work() {}\n",
        "app.js": 'import {work as run} from "./util.js";\nfunction main() { run(); }\n',
    })
    assert calls(graph) == {("app.js.main", "util.js.work")}
