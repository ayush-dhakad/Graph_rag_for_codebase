from __future__ import annotations

import json
import os
import re

from .models import GraphQuery

SYSTEM_PROMPT = """Convert a codebase question into JSON only: {\"intent\": one of callers, callees, imports, definitions, class_members, \"symbol\": string, \"limit\": integer}. Never produce Cypher or explanation."""


class QueryTranslator:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("CODEGRAPHRAG_QUERY_MODEL", "gpt-4.1-mini")

    def translate(self, question: str) -> GraphQuery:
        if os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                response = OpenAI().chat.completions.create(
                    model=self.model,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}],
                )
                payload = json.loads(response.choices[0].message.content or "{}")
                return GraphQuery(intent=payload["intent"], symbol=payload["symbol"], limit=min(int(payload.get("limit", 12)), 30))
            except Exception as error:
                print(f"LLM translation unavailable; using local rules: {error}")
        return self._rule_based(question)

    @staticmethod
    def _rule_based(question: str) -> GraphQuery:
        lower = question.lower()
        intent = "definitions"
        if any(word in lower for word in ("what calls", "who calls", "callers", "used by")):
            intent = "callers"
        elif any(word in lower for word in ("calls from", "what does", "callees", "dependencies of")):
            intent = "callees"
        elif "import" in lower:
            intent = "imports"
        elif any(word in lower for word in ("members", "methods", "inside class")):
            intent = "class_members"
        quoted = re.search(r"['\"]([^'\"]+)['\"]", question)
        tokens = re.findall(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", question)
        ignored = {"what", "calls", "call", "the", "auth", "module", "function", "class", "does", "who", "import", "imports", "from", "of", "by", "show", "me"}
        symbol = quoted.group(1) if quoted else next((token for token in reversed(tokens) if token.lower() not in ignored), "")
        if not symbol:
            raise ValueError("Could not identify a symbol; quote it, e.g. What calls 'authenticate'?")
        return GraphQuery(intent=intent, symbol=symbol)
