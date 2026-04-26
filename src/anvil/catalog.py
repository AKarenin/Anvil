import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from anvil.models import Tool


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class ToolCatalog:
    def __init__(self, catalog_dir: Path, embedder_model: str = "all-MiniLM-L6-v2"):
        self.catalog_dir = Path(catalog_dir)
        self.catalog_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = SentenceTransformer(embedder_model)
        self.tools: dict[str, Tool] = {}
        self._load()

    def _load(self) -> None:
        for fp in sorted(self.catalog_dir.glob("*.json")):
            try:
                data = json.loads(fp.read_text())
                tool = Tool.model_validate(data)
                self.tools[tool.name] = tool
            except Exception as e:
                print(f"[catalog] failed to load {fp.name}: {e}")

    def _embed(self, text: str) -> list[float]:
        vec = self.embedder.encode(text, convert_to_numpy=True)
        return vec.tolist()

    def _persist(self, tool: Tool) -> None:
        fp = self.catalog_dir / f"{tool.name}.json"
        fp.write_text(tool.model_dump_json(indent=2))

    def add(self, tool: Tool) -> None:
        if not tool.trigger_embedding:
            tool.trigger_embedding = self._embed(tool.description)
        self.tools[tool.name] = tool
        self._persist(tool)

    def retrieve(self, query: str, k: int = 5) -> list[tuple[Tool, float]]:
        if not self.tools:
            return []
        qv = np.array(self._embed(query))
        scored = []
        for tool in self.tools.values():
            tv = np.array(tool.trigger_embedding)
            scored.append((tool, _cosine(qv, tv)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def increment_usage(self, tool_name: str) -> None:
        tool = self.tools.get(tool_name)
        if tool is None:
            return
        tool.usage_count += 1
        self._persist(tool)

    def all_tools(self) -> list[Tool]:
        return list(self.tools.values())
