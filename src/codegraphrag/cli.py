from __future__ import annotations

import argparse
import json

from .benchmark import run_benchmark
from .pipeline import CodeGraphRAG


def main() -> None:
    parser = argparse.ArgumentParser(description="AST-first GraphRAG for code")
    sub = parser.add_subparsers(dest="command", required=True)
    index = sub.add_parser("index", help="Parse a repository and persist its graph")
    index.add_argument("repository", nargs="?", default=".")
    ask = sub.add_parser("ask", help="Retrieve targeted code context")
    ask.add_argument("question")
    ask.add_argument("--budget", type=int, default=2500)
    report = sub.add_parser("token-report", help="Compare GraphRAG context with whole-repository context")
    report.add_argument("question")
    report.add_argument("--repository", default=".")
    report.add_argument("--budget", type=int, default=2500)
    benchmark = sub.add_parser("benchmark-gemini", help="Use Gemini to compare baseline and GraphRAG token usage")
    benchmark.add_argument("question")
    benchmark.add_argument("--expected", required=True, help="Text that must appear in a correct answer")
    benchmark.add_argument("--repository", default=".")
    benchmark.add_argument("--model", default="gemini-3.6-flash")
    benchmark.add_argument("--budget", type=int, default=2500)
    benchmark.add_argument("--output", help="Write the JSON benchmark report to this path")
    args = parser.parse_args()
    rag = CodeGraphRAG()
    if args.command == "index":
        store = rag.index(args.repository)
        print(f"Indexed {len(store.graph.nodes)} nodes and {len(store.graph.edges)} edges into {rag.index_path}")
    elif args.command == "ask":
        result = rag.ask(args.question, args.budget)
        print(f"Intent: {result.query.intent}; symbol: {result.query.symbol}\nCypher: {result.cypher}\nNodes: {len(result.nodes)}; context tokens: {result.tokens}\n\n{result.context}")
    elif args.command == "token-report":
        print(json.dumps(rag.token_report(args.repository, args.question, args.budget), indent=2))
    else:
        result = run_benchmark(args.repository, args.question, args.expected, args.model, args.budget)
        payload = json.dumps(result.to_dict(), indent=2)
        if args.output:
            from pathlib import Path
            Path(args.output).write_text(payload + "\n", encoding="utf-8")
        print(payload)


if __name__ == "__main__":
    main()
