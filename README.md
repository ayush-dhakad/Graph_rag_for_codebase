# Code GraphRAG

An AST-first retrieval system that treats a codebase as a graph, rather than embedding every chunk of code. It reduces final-LLM context by retrieving only code connected to the question's target symbol.

## Architecture

`Tree-sitter AST → CodeGraph (functions/classes/imports/calls) → query translator → read-only Cypher intent → targeted source context → final LLM`

- **Structural parsing:** Tree-sitter extracts files, functions, classes, methods, and imports.
- **Graph storage:** JSON-backed in-memory graph works locally; `Neo4jGraphStore` is included for Neo4j/Memgraph deployments.
- **Question to graph query:** An optional lightweight OpenAI model produces constrained JSON that is compiled into parameterized read-only Cypher. Without `OPENAI_API_KEY`, deterministic local rules cover callers, callees, imports, definitions, and class members.
- **Context assembly:** Source for matching graph nodes is packed up to a token budget. No vector database or Sentence-Transformer is used.

## M5 Air setup

The project virtual environment is `.venv`.

```bash
source .venv/bin/activate
pip install -e .
codegraphrag index .
codegraphrag ask "What calls 'authenticate'?"
codegraphrag token-report "What calls 'authenticate'?" --repository .
codegraphrag benchmark-gemini "What calls 'assemble_context'?" --expected "CodeGraphRAG.ask"
pytest
```

On first indexing, `tree-sitter-language-pack` downloads only the grammars encountered and stores them under `.codegraphrag/tree-sitter`.

For lightweight LLM translation, set `OPENAI_API_KEY` and optionally `CODEGRAPHRAG_QUERY_MODEL` (default: `gpt-4.1-mini`). The final answer model receives the printed targeted context, not the entire repository.

## Live Gemini verification

Set `GEMINI_API_KEY` and use `benchmark-gemini`. It makes two identical Gemini calls: one receives all source files and the other receives GraphRAG context. It records Gemini's `usageMetadata` prompt/output/thought/total token counts and checks whether both answers contain an expected answer phrase. This measures actual provider-reported tokens, including the prompt instructions, model thinking, and generated answer, rather than a local tokenizer estimate.

## Token reduction measurement

`token-report` compares tokenized contents of all source files with the context returned for one question. The exact savings depend on the graph neighborhood and budget:

`reduction = 100 × (1 - targeted_context_tokens / whole_repository_tokens)`

For example, a 100,000-token repository and a 2,500-token retrieved context yields **97.5% fewer input tokens** to the final LLM (97,500 tokens saved). The command reports actual values for your repository after indexing.

## Neo4j / Memgraph

Instantiate `Neo4jGraphStore(uri, username, password)` and call `write_graph(graph)` after indexing. It uses ordinary Cypher `MERGE` statements for the fixed relationship set, so it works without APOC and is compatible with Neo4j or Memgraph.
