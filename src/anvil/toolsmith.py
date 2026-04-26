import json
import os
import re
from typing import Optional

from anthropic import Anthropic

from anvil.models import Tool


TOOLSMITH_MODEL = os.environ.get("ANVIL_TOOLSMITH_MODEL", "claude-sonnet-4-6")


SYSTEM_PROMPT = """You are a toolsmith. You generate reusable Python tools for a local agent that cannot answer certain queries on its own (typically because the query needs live data, external computation, or capabilities outside the local model's training).

You MUST respond with a single JSON object that conforms exactly to this schema, with no prose and no code fences:

{
  "name": "snake_case_function_name",
  "description": "Natural-language description of the capability class (NOT the specific query). Written for semantic retrieval: describe what the tool does in general terms so future similar queries will match.",
  "input_schema": {
    "type": "object",
    "properties": {"param_name": {"type": "string|number|integer|boolean", "description": "..."}},
    "required": ["param_name"]
  },
  "implementation": "def snake_case_function_name(param_name):\\n    ...\\n    return result"
}

Rules:
- The implementation must define a top-level function whose name equals the `name` field exactly.
- The function parameters must match `input_schema.properties`.
- The function must return a string or JSON-serializable value (never None).
- Allowed imports ONLY: requests, datetime, json, re, math, urllib (and submodules).
- No file I/O, no subprocess, no `os` access, no network calls other than via `requests` or `urllib`.
- `description` must describe the capability broadly (e.g., "fetches current weather for a location") rather than quoting the user's specific query. Keep it CONCISE — under 20 words — because it is used for semantic retrieval, and shorter, focused descriptions match better against paraphrased queries.

INFEASIBILITY:
If the query cannot reasonably be implemented as a single Python function under the constraints above — e.g., building a game like Minecraft, building a full website or mobile app end-to-end, training an ML model from scratch, anything requiring a GUI, a real-time loop, persistent state across calls, hardware access, or credentials/paid APIs — respond INSTEAD with exactly this JSON object (no prose, no code fences):

{"infeasible": true, "reason": "<one short sentence explaining why>"}

Use this only for requests that are fundamentally outside the scope of a constrained Python script. A merely unfamiliar API or niche data source is NOT infeasible — attempt those normally.

CRITICAL ERROR-HANDLING RULE:
- DO NOT wrap calls in try/except just to return a descriptive error string (e.g., `return f"Error: {e}"`). Let exceptions propagate naturally.
- If an operation fails (network error, missing response field, bad status code, etc.), allow Python to raise the exception. The surrounding system will catch it, feed it back to you, and request a corrected tool.
- Only use try/except when you have a genuine recovery strategy (e.g., a fallback API), never as error-suppression.
- Prefer APIs that do not require authentication. If a field might be missing from a response, call `.get()` or raise KeyError — do not silently return an error message.

Example of a well-formed response:
{
  "name": "get_current_weather",
  "description": "Fetches the current weather conditions and temperature for a given city or location.",
  "input_schema": {
    "type": "object",
    "properties": {"location": {"type": "string", "description": "City name or location"}},
    "required": ["location"]
  },
  "implementation": "def get_current_weather(location):\\n    import requests\\n    r = requests.get(f'https://wttr.in/{location}?format=j1', timeout=10)\\n    r.raise_for_status()\\n    data = r.json()\\n    cur = data['current_condition'][0]\\n    return f\\"{location}: {cur['temp_F']}F, {cur['weatherDesc'][0]['value']}\\""
}
"""


REPAIR_INSTRUCTION = """The previous tool you generated FAILED. Study the error, diagnose the root cause, and return a corrected tool as a single JSON object following the same schema.

Keep the SAME tool `name` so the catalog entry is updated in place (unless the capability genuinely needs a different name).

Common failure modes to check for:
- API endpoint returned a different structure than expected (wrong keys, different nesting)
- API requires authentication that isn't available
- The tool caught an exception and returned an error string — that is forbidden; let exceptions raise
- Network / DNS error — consider a different, more reliable API
- Parameter types or names don't match how the function is actually being called
"""


class ToolsmithInfeasibleError(Exception):
    pass


def _parse_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("no JSON object found in toolsmith response")


class Toolsmith:
    def __init__(self, client: Optional[Anthropic] = None, model: str = TOOLSMITH_MODEL):
        self.client = client or Anthropic()
        self.model = model

    def _call(self, messages: list[dict]) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        return resp.content[0].text

    def _parse_into_tool(self, text: str, query: str) -> Tool:
        data = _parse_json(text)
        if data.get("infeasible") is True:
            raise ToolsmithInfeasibleError(str(data.get("reason", "")).strip())
        return Tool(
            name=data["name"],
            description=data["description"],
            input_schema=data["input_schema"],
            implementation=data["implementation"],
            created_from_query=query,
        )

    def generate_tool(self, query: str) -> Tool:
        user_msg = (
            "Generate a reusable tool that can answer the following query and any similar "
            f"queries in the same capability class.\n\nQuery: {query}"
        )
        return self._converse_for_tool(query, [{"role": "user", "content": user_msg}])

    def repair_tool(self, query: str, broken_tool: Tool, error_message: str) -> Tool:
        user_msg = (
            f"{REPAIR_INSTRUCTION}\n\n"
            f"Original user query: {query}\n\n"
            f"Previous tool name: {broken_tool.name}\n"
            f"Previous tool description: {broken_tool.description}\n"
            f"Previous input_schema: {json.dumps(broken_tool.input_schema)}\n\n"
            f"Previous implementation:\n```python\n{broken_tool.implementation}\n```\n\n"
            f"Error encountered: {error_message}\n\n"
            "Return a corrected tool as a single JSON object."
        )
        return self._converse_for_tool(query, [{"role": "user", "content": user_msg}])

    def _converse_for_tool(self, query: str, messages: list[dict]) -> Tool:
        last_error: Optional[Exception] = None
        for _ in range(2):
            text = self._call(messages)
            try:
                return self._parse_into_tool(text, query)
            except ToolsmithInfeasibleError:
                raise
            except Exception as e:
                last_error = e
                messages.append({"role": "assistant", "content": text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"The previous response could not be parsed: {e}. "
                            "Return ONLY a single valid JSON object matching the schema. "
                            "No prose, no code fences."
                        ),
                    }
                )
        raise ValueError(f"toolsmith failed to produce valid tool after retry: {last_error}")
