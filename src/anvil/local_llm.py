import json
import os
import re
from typing import Optional

import ollama

from anvil.models import Tool


MODEL = os.environ.get("ANVIL_LOCAL_MODEL", "phi3.5:3.8b")

_UNCERTAIN_PHRASES = (
    "i don't know",
    "i do not know",
    "i'm not sure",
    "i am not sure",
    "i cannot",
    "i can't",
    "unable to",
    "no information",
)

_SCHEMA_META_KEYS = {"type", "properties", "required", "items", "enum", "$schema"}


def _value_in_query(value, query: str) -> bool:
    if value is None or value == "":
        return True
    q = query.lower()
    s = str(value).strip().lower()
    if not s:
        return True
    if s in q:
        return True
    for part in re.split(r"[/_\-\s,]+", s):
        if len(part) >= 3 and part in q:
            return True
    return False


def _chat(messages, **kwargs) -> str:
    resp = ollama.chat(model=MODEL, messages=messages, **kwargs)
    return resp["message"]["content"]


def answer_directly(query: str) -> Optional[str]:
    system = (
        "You are a helpful local assistant. Answer the user's question if you know it from "
        "your training data. If you lack the knowledge or capability to answer, OR the request "
        "is a task you cannot perform as the user intends — for example, the question requires "
        "live data, the current time, web access, or external tools; or the request is for "
        "in-depth research, generating an entire codebase or application, building a game, "
        "training a model, designing a system end-to-end, or any other open-ended construction "
        "task that goes beyond a direct textual answer — "
        "respond with exactly the single word: UNKNOWN"
    )
    text = _chat([
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]).strip()

    if not text or len(text) < 2:
        return None
    stripped = text.strip().strip(".").strip()
    if stripped.upper() == "UNKNOWN":
        return None
    lowered = text.lower()
    if lowered.startswith("unknown"):
        return None
    if any(p in lowered for p in _UNCERTAIN_PHRASES):
        return None
    return text


def _parse_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"no JSON object found in: {text[:200]}")


def _build_extract_prompt(tool: Tool) -> tuple[str, str, list[str], list[str]]:
    props = tool.input_schema.get("properties", {}) or {}
    required = list(tool.input_schema.get("required", []) or [])
    param_names = list(props.keys())

    if not param_names:
        return "", "", [], []

    lines = []
    for name, info in props.items():
        t = info.get("type", "any") if isinstance(info, dict) else "any"
        desc = info.get("description", "") if isinstance(info, dict) else ""
        marker = "required" if name in required else "optional"
        lines.append(f'  - "{name}" ({t}, {marker}): {desc}')
    params_block = "\n".join(lines)

    example_keys = param_names[:2]
    example = "{" + ", ".join(f'"{k}": <value>' for k in example_keys) + "}"
    return params_block, example, param_names, required


def extract_parameters(query: str, tool: Tool) -> dict:
    params_block, example, param_names, required = _build_extract_prompt(tool)
    if not param_names:
        return {}

    system = (
        "You extract argument values to call a Python function. "
        "Respond with ONLY a flat JSON object whose keys are the parameter names shown. "
        'NEVER use the keys "type", "properties", "required", "items", or "enum" — '
        "those are schema metadata, not function arguments."
    )
    base_prompt = (
        f"Function: {tool.name}\n"
        f"Purpose: {tool.description}\n"
        f"Parameters (use these exact keys and nothing else):\n{params_block}\n\n"
        f"User query: {query}\n\n"
        f"Return a JSON object of the form {example} with values taken from the query. "
        "If a value is not in the query, make a sensible inference."
    )

    allowed = set(param_names)
    last_err: Optional[Exception] = None
    prompt = base_prompt
    for _ in range(2):
        text = _chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            format="json",
        )
        try:
            data = _parse_json(text)
            if not isinstance(data, dict):
                raise ValueError(f"expected JSON object, got {type(data).__name__}")

            if data and _SCHEMA_META_KEYS.issuperset(data.keys()):
                raise ValueError(
                    f"model echoed schema metadata instead of values "
                    f"(got keys {list(data.keys())}, expected {param_names})"
                )

            filtered = {k: v for k, v in data.items() if k in allowed}
            missing = [r for r in required if r not in filtered]
            if missing:
                raise ValueError(
                    f"missing required parameters {missing} "
                    f"(model returned keys {list(data.keys())})"
                )
            invented = [
                (r, filtered.get(r)) for r in required
                if not _value_in_query(filtered.get(r), query)
            ]
            if invented:
                raise ValueError(
                    f"required parameter(s) appear invented (not grounded in the query): {invented}"
                )
            return filtered
        except Exception as e:
            last_err = e
            prompt = (
                base_prompt
                + f"\n\nThe previous attempt was invalid: {e}. "
                f"Return ONLY keys from: {param_names}. No schema keywords."
            )
    raise ValueError(f"failed to extract parameters after retry: {last_err}")


def generate(prompt: str, system: str = "") -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return _chat(messages)


def route_tools(
    query: str, candidates: list[tuple[Tool, float]]
) -> tuple[Optional[Tool], str]:
    if not candidates:
        return None, "no candidates"

    lines = []
    for i, (tool, score) in enumerate(candidates, 1):
        lines.append(
            f"{i}. {tool.name} — {tool.description} (embedding sim={score:.2f})"
        )
    tools_block = "\n".join(lines)

    system = (
        "You are a tool router for a local agent. Given a user query and a numbered "
        "list of tools with descriptions, decide which SINGLE tool can answer the query, "
        "or answer NONE.\n\n"
        "STRICT RULES — bias hard toward NONE:\n"
        "1. Choose a tool ONLY if the user's query explicitly names the specific subject "
        "that tool operates on (a city for weather, a coin for crypto price, a location "
        "for time, an expression for math, etc.). If the subject is missing or ambiguous, "
        "choose NONE.\n"
        "2. Do NOT stretch a tool to fit a tangentially related query just because the "
        "tool description mentions related terms. Word overlap is not enough.\n"
        "3. If you would need to invent or guess a value for any of the tool's required "
        "inputs, choose NONE — that means the tool does not actually fit the query.\n"
        "4. When in doubt, choose NONE. A correct NONE is better than a wrong tool.\n\n"
        'Respond with ONLY a JSON object: {"choice": <1..N or "NONE">, "reason": "<brief>"}.'
    )
    prompt = (
        f'User query: "{query}"\n\n'
        f"Available tools:\n{tools_block}\n\n"
        "Pick the tool number that can answer the query, or NONE if none fit."
    )
    try:
        text = _chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            format="json",
        )
        data = _parse_json(text)
    except Exception as e:
        return None, f"routing call failed: {e}"

    choice = data.get("choice")
    reason = str(data.get("reason", "")).strip() or "(no reason)"

    idx: Optional[int] = None
    if isinstance(choice, int):
        idx = choice - 1
    elif isinstance(choice, str):
        s = choice.strip().upper().strip('"').strip("'")
        if s in ("NONE", "NULL"):
            return None, reason
        if s.isdigit():
            idx = int(s) - 1

    if idx is None or not (0 <= idx < len(candidates)):
        return None, f"unrecognized choice {choice!r}: {reason}"
    return candidates[idx][0], reason


def verify_tool_fit(query: str, tool: Tool) -> tuple[bool, str]:
    system = (
        "You will be given a user's query and a tool's description. "
        "Decide whether this tool's described capability ACTUALLY allows answering the user's task. "
        "Be strict: if the tool's purpose is even slightly different from what the user is asking for, answer NO. "
        "Word overlap (shared terms in the description and the query) is NOT enough — the tool's "
        "core capability must directly produce the answer the user wants. "
        'Respond with ONLY a JSON object: {"answer": "YES"|"NO", "reason": "<brief>"}.'
    )
    prompt = (
        f'User query: "{query}"\n\n'
        f"Tool name: {tool.name}\n"
        f"Tool description: {tool.description}\n\n"
        "Does this tool's described capability allow you to answer the user's task?"
    )
    try:
        text = _chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            format="json",
        )
        data = _parse_json(text)
    except Exception as e:
        return False, f"verification call failed: {e}"

    ans = str(data.get("answer", "")).strip().upper().strip('"').strip("'")
    reason = str(data.get("reason", "")).strip() or "(no reason)"
    return ans == "YES", reason


def synthesize_answer(query: str, tool_name: str, tool_output) -> str:
    system = (
        "You answer the user's question using the provided tool output as authoritative data.\n"
        "Rules:\n"
        "- Trust the tool output completely. It is up-to-date and correct.\n"
        "- Do NOT apologize, do NOT add caveats like 'please check another source', "
        "do NOT question whether the data matches the query.\n"
        "- If a location's place name in the data looks different from what the user asked "
        "(e.g. 'Phoenix Row' for a query about Durham), trust that the tool resolved the name "
        "correctly — just report the facts.\n"
        "- Respond in 1 to 3 short sentences. Speak naturally, as if you already knew the info."
    )
    prompt = (
        f"User question: {query}\n\n"
        f"Tool output:\n{tool_output}\n\n"
        "Answer:"
    )
    return _chat([
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]).strip()
