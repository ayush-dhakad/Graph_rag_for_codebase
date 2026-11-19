# Code GraphRAG

An AST-first retrieval system that uses a code graph to find source for structural questions. Optional semantic search supports questions that describe behavior without naming a symbol.

## Architecture

`Tree-sitter AST → CodeGraph (functions/classes/imports/calls) → query translator → read-only Cypher intent → targeted source context → final LLM`

- **Structural parsing:** Tree-sitter extracts files, functions, classes, methods, imports, and call expressions. Python imports (including aliases and relative imports) and named/namespace ES imports are read from syntax nodes rather than source-text patterns.
- **Call resolution:** Calls are linked using lexical scopes, explicit imports, `self`/`cls`/`this` receivers, and simple local constructor assignments. Nested functions own their body calls; comments and strings do not create calls. Unknown or ambiguous targets remain unlinked. This is a structural index, not a full type checker: dynamic dispatch, inheritance resolution, factory return types, anonymous callbacks, wildcard imports, and complex receiver expressions are not inferred. Language coverage depends on the supported syntax-node shapes.
- **Graph storage:** JSON-backed in-memory graph works locally; `Neo4jGraphStore` is included for Neo4j/Memgraph deployments.
- **Question to graph query:** An optional lightweight OpenAI model produces constrained JSON that is compiled into parameterized read-only Cypher. Without `OPENAI_API_KEY`, deterministic local rules cover callers, callees, imports, definitions, and class members.
- **Context assembly:** Source for matching graph nodes is packed up to a token budget. Optional embeddings are stored with the graph in the local JSON index.

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

## Semantic search

Set `OPENAI_API_KEY`, then build the semantic index and search it:

```bash
codegraphrag index . --semantic
codegraphrag ask "How does the app decide what code to send to the model?" --semantic
```

Indexing sends paths, symbol names, and bounded source excerpts to OpenAI for embeddings. Semantic queries send the question for embedding, rank code by cosine similarity, and include direct graph neighbors within the context budget.

With a semantic index, plain `ask` uses it as a fallback when symbol lookup finds no nodes. Pass `--semantic` to search by meaning first. Rebuild with `--semantic` after source changes to refresh the vectors. A graph-only reindex replaces the semantic index.

## Live Gemini verification

Set `GEMINI_API_KEY` and use `benchmark-gemini`. It makes two identical Gemini calls: one receives all source files and the other receives GraphRAG context. It records Gemini's `usageMetadata` prompt/output/thought/total token counts and checks whether both answers contain an expected answer phrase. This measures actual provider-reported tokens, including the prompt instructions, model thinking, and generated answer, rather than a local tokenizer estimate.

## Token reduction measurement

`token-report` compares tokenized contents of all source files with the context returned for one question. The exact savings depend on the graph neighborhood and budget:

`reduction = 100 × (1 - targeted_context_tokens / whole_repository_tokens)`

For example, a 100,000-token repository and a 2,500-token retrieved context yields **97.5% fewer input tokens** to the final LLM (97,500 tokens saved). The command reports actual values for your repository after indexing.

## Neo4j / Memgraph

Instantiate `Neo4jGraphStore(uri, username, password)` and call `write_graph(graph)` after indexing. It uses ordinary Cypher `MERGE` statements for the fixed relationship set, so it works without APOC and is compatible with Neo4j or Memgraph.

After changing source code or upgrading the parser, rerun `codegraphrag index .` to replace the saved graph. Existing benchmark reports describe their original run; rerun the benchmark separately to measure the new index.
