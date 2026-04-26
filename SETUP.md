# Setup

## 1. Clone and enter the repo

```
git clone <repo-url> anvil
cd anvil
```

## 2. Create a virtual environment

```
python3.11 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```
pip install -r requirements.txt
```

## 4. Pull the local model

```
ollama pull phi3.5:3.8b
```

Make sure the Ollama server is running (`ollama serve`, or the desktop app).

The local model name is configurable via `ANVIL_LOCAL_MODEL` (default `phi3.5:3.8b`).

## 5. Set your Anthropic API key

```
export ANTHROPIC_API_KEY=sk-ant-...
```

The toolsmith model is configurable via `ANVIL_TOOLSMITH_MODEL` (default `claude-sonnet-4-6`).

## 6. Verify the unit tests pass

```
pytest -m "not integration"
```

All tests should pass without Ollama or an API key available.

## 7. Run the CLI

```
PYTHONPATH=src python -m anvil.server
```

## 8. (Optional) Run integration tests

These require Ollama running, the Phi model pulled, and `ANTHROPIC_API_KEY` set.

```
pytest -m integration
```
