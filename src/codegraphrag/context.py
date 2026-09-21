from __future__ import annotations

from .models import CodeNode


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    try:
        import tiktoken
        return len(tiktoken.encoding_for_model(model).encode(text))
    except Exception:
        return max(1, len(text) // 4)


def assemble_context(nodes: list[CodeNode], budget_tokens: int = 2500) -> tuple[str, int]:
    sections: list[str] = []
    used = 0
    for node in nodes:
        content = node.source or f"{node.kind} {node.name} ({node.path}:{node.start_line})"
        section = f"# {node.kind} {node.qualified_name or node.name} — {node.path}:{node.start_line}-{node.end_line}\n{content}".strip()
        tokens = count_tokens(section)
        if used + tokens > budget_tokens:
            continue
        sections.append(section)
        used += tokens
    return "\n\n".join(sections), used
