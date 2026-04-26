import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

from anvil.server import AnvilServer  # noqa: E402 — after dotenv


CATALOG_DIR = Path(os.environ.get("ANVIL_CATALOG_DIR", _REPO_ROOT / "data" / "tool_catalog"))


_server: Optional[AnvilServer] = None


def get_server() -> AnvilServer:
    global _server
    if _server is None:
        _server = AnvilServer(CATALOG_DIR)
    return _server


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_server()
    yield


app = FastAPI(title="Anvil", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
def query(req: QueryRequest):
    return get_server().handle_query(req.query).model_dump()


@app.get("/catalog")
def catalog():
    return [
        {
            "name": t.name,
            "description": t.description,
            "usage_count": t.usage_count,
            "created_from_query": t.created_from_query,
        }
        for t in get_server().catalog.all_tools()
    ]


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Anvil</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    background: #0f1115; color: #e6e6e6;
    max-width: 820px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5;
  }
  h1 { font-weight: 500; letter-spacing: 0.1em; margin-bottom: 0; }
  .subtitle { color: #888; font-size: 0.85rem; margin-bottom: 1.5rem; }
  textarea {
    width: 100%; background: #1a1d24; color: #e6e6e6;
    border: 1px solid #333; padding: 0.75rem; font: inherit;
    border-radius: 6px; resize: vertical;
  }
  textarea:focus { outline: none; border-color: #4a7; }
  button {
    background: #3b7; color: #000; border: 0; padding: 0.55rem 1.2rem;
    font: inherit; border-radius: 6px; cursor: pointer; margin-top: 0.5rem;
    font-weight: 600;
  }
  button:disabled { background: #444; color: #888; cursor: wait; }
  .result {
    background: #1a1d24; border: 1px solid #333; border-radius: 6px;
    padding: 1rem; margin-top: 1rem;
  }
  .result .output { white-space: pre-wrap; }
  .meta {
    display: flex; gap: 0.75rem; flex-wrap: wrap; font-size: 0.8rem;
    color: #aaa; margin-top: 0.75rem; padding-top: 0.6rem;
    border-top: 1px solid #2a2d34;
  }
  .tag { padding: 0.1rem 0.55rem; border-radius: 3px; font-weight: 600; font-size: 0.75rem; }
  .tag.local_direct { background: #1e3a1e; color: #8fdf8f; }
  .tag.cache_hit    { background: #1e2e3a; color: #8fcfef; }
  .tag.generated    { background: #3a311e; color: #efcf8f; }
  .tag.fallback     { background: #3a1e1e; color: #ef8f8f; }
  .q-echo { color: #666; font-size: 0.85rem; margin-bottom: 0.5rem; }
  .trace { margin-top: 0.75rem; }
  .trace summary { color: #888; font-size: 0.8rem; cursor: pointer; user-select: none; }
  .trace pre {
    background: #0b0d11; border: 1px solid #222; border-radius: 4px;
    padding: 0.5rem 0.6rem; margin: 0.4rem 0 0 0; font-size: 0.75rem;
    color: #b0b8c4; overflow-x: auto; white-space: pre-wrap; word-break: break-all;
  }
  details { margin-top: 2rem; border-top: 1px solid #2a2d34; padding-top: 1rem; }
  details summary { cursor: pointer; color: #aaa; user-select: none; }
  .tool {
    padding: 0.6rem 0.4rem; border-bottom: 1px solid #222;
  }
  .tool:last-child { border-bottom: 0; }
  .tool-name { color: #8fcfef; font-weight: 600; }
  .tool-desc { color: #bbb; font-size: 0.85rem; margin-top: 0.15rem; }
  .tool-meta { color: #666; font-size: 0.75rem; margin-top: 0.15rem; }
  .empty { color: #666; font-size: 0.85rem; padding: 0.6rem 0.4rem; }
</style>
</head>
<body>
  <h1>ANVIL</h1>
  <div class="subtitle">self-evolving local chatbot · phi3.5 + claude toolsmith</div>

  <form id="f">
    <textarea id="q" rows="2" placeholder="ask anything — e.g. 'convert 50 miles to kilometers'" autofocus></textarea>
    <button id="send" type="submit">send</button>
  </form>

  <div id="out"></div>

  <details>
    <summary>tool catalog (<span id="count">0</span>)</summary>
    <div id="catalog"></div>
  </details>

<script>
const out = document.getElementById('out');
const send = document.getElementById('send');
const qEl = document.getElementById('q');

async function loadCatalog() {
  try {
    const r = await fetch('/catalog');
    const tools = await r.json();
    document.getElementById('count').textContent = tools.length;
    const el = document.getElementById('catalog');
    if (!tools.length) {
      el.innerHTML = '<div class="empty">(empty — run a query that the local model cannot answer)</div>';
      return;
    }
    el.innerHTML = tools.map(t =>
      `<div class="tool">
        <div class="tool-name">${escapeHtml(t.name)}</div>
        <div class="tool-desc">${escapeHtml(t.description)}</div>
        <div class="tool-meta">uses: ${t.usage_count} · from: "${escapeHtml(t.created_from_query)}"</div>
      </div>`
    ).join('');
  } catch (e) {
    console.error(e);
  }
}

document.getElementById('f').onsubmit = async (e) => {
  e.preventDefault();
  const query = qEl.value.trim();
  if (!query) return;
  send.disabled = true; send.textContent = 'thinking…';
  try {
    const r = await fetch('/query', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ query }),
    });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const data = await r.json();
    const meta = [
      `<span class="tag ${data.path_taken}">${data.path_taken}</span>`,
      `${Math.round(data.latency_ms)}ms`,
    ];
    if (data.matched_tool_name) meta.push('tool: ' + escapeHtml(data.matched_tool_name));
    if (data.similarity_score != null) meta.push('sim: ' + data.similarity_score.toFixed(2));
    const el = document.createElement('div');
    el.className = 'result';
    const traceHtml = (data.trace && data.trace.length)
      ? '<details class="trace" open><summary>trace (' + data.trace.length + ' steps)</summary>' +
        '<pre>' + escapeHtml(data.trace.join('\\n')) + '</pre></details>'
      : '';
    el.innerHTML =
      '<div class="q-echo">› ' + escapeHtml(data.query) + '</div>' +
      '<div class="output">' + escapeHtml(data.output) + '</div>' +
      '<div class="meta">' + meta.join(' · ') + '</div>' +
      traceHtml;
    out.prepend(el);
    loadCatalog();
  } catch (err) {
    const el = document.createElement('div');
    el.className = 'result';
    el.textContent = 'error: ' + err.message;
    out.prepend(el);
  } finally {
    send.disabled = false; send.textContent = 'send';
    qEl.value = '';
    qEl.focus();
  }
};

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => (
    {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]
  ));
}

loadCatalog();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


def main():
    import uvicorn

    uvicorn.run(
        "anvil.web:app",
        host=os.environ.get("ANVIL_HOST", "127.0.0.1"),
        port=int(os.environ.get("ANVIL_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
