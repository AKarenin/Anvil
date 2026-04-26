import os
import time
from pathlib import Path
from typing import Callable, Optional

from anvil import local_llm
from anvil.catalog import ToolCatalog
from anvil.executor import ToolExecutionError, ToolExecutor
from anvil.models import QueryResult, Tool
from anvil.toolsmith import Toolsmith, ToolsmithInfeasibleError
from anvil.validator import ToolValidator, looks_like_error


MAX_REPAIR = int(os.environ.get("ANVIL_MAX_REPAIR", "1"))
SYNTHESIZE = os.environ.get("ANVIL_SYNTHESIZE", "1") not in ("0", "false", "False")


class AnvilServer:
    def __init__(self, catalog_dir: Path):
        self.catalog = ToolCatalog(Path(catalog_dir))
        self.executor = ToolExecutor()
        self.validator = ToolValidator()
        self._toolsmith: Optional[Toolsmith] = None

    @property
    def toolsmith(self) -> Toolsmith:
        if self._toolsmith is None:
            self._toolsmith = Toolsmith()
        return self._toolsmith

    def handle_query(self, query: str) -> QueryResult:
        t0 = time.time()
        trace: list[str] = []

        def log(msg: str) -> None:
            elapsed = (time.time() - t0) * 1000
            line = f"[{elapsed:7.0f}ms] {msg}"
            trace.append(line)
            print(line, flush=True)

        def finish(output: str, path: str, tool_name: Optional[str] = None,
                   score: Optional[float] = None) -> QueryResult:
            log(f"DONE → path={path}")
            return QueryResult(
                query=query,
                output=output,
                path_taken=path,  # type: ignore[arg-type]
                matched_tool_name=tool_name,
                similarity_score=score,
                latency_ms=(time.time() - t0) * 1000,
                trace=trace,
            )

        log(f"query: {query!r}")
        log(f"catalog size: {len(self.catalog.tools)} tools | "
            f"max_repair: {MAX_REPAIR} | synthesize: {SYNTHESIZE}")

        # step 1
        log("step 1/3: asking local model (phi3.5) directly")
        direct = local_llm.answer_directly(query)
        if direct is not None:
            preview = direct.replace("\n", " ")[:80]
            log(f"local model answered ({len(direct)} chars): {preview!r}")
            return finish(direct, "local_direct")
        log("local model returned UNKNOWN → not usable")

        # step 2
        log("step 2/3: semantic retrieval (top-3), then phi-as-router")
        candidates = self.catalog.retrieve(query, k=3)
        selected_tool: Optional[Tool] = None
        selected_score: Optional[float] = None

        if not candidates:
            log("  catalog is empty → going to toolsmith")
        else:
            for tool, score in candidates:
                log(f"    {tool.name:40s} sim={score:.3f}")
            log(f"asking phi to route among {len(candidates)} candidate(s)")
            chosen, reason = local_llm.route_tools(query, candidates)
            if chosen is None:
                log(f"phi routing: NONE — {reason}")
            else:
                selected_tool = chosen
                selected_score = next((s for t, s in candidates if t.name == chosen.name), None)
                log(f"phi routing: picked {chosen.name} — {reason}")

        if selected_tool is not None:
            log(f"asking phi to verify '{selected_tool.name}' fits the query")
            fits, fit_reason = local_llm.verify_tool_fit(query, selected_tool)
            if not fits:
                log(f"phi verification: NO — {fit_reason}; abandoning cache, going to toolsmith")
                selected_tool = None
                selected_score = None
            else:
                log(f"phi verification: YES — {fit_reason}")

        if selected_tool is not None:
            log(f"cache HIT via phi router: {selected_tool.name}")
            raw, err, final_tool = self._run_with_repair(
                query, selected_tool, log, already_cached=True
            )
            if raw is not None:
                self.catalog.increment_usage(final_tool.name)
                answer = self._synthesize(query, final_tool, raw, log)
                return finish(answer, "cache_hit", final_tool.name, selected_score)
            log(f"cache path failed after repair attempts: {err} → trying fresh generation")

        # step 3
        log("step 3/3: asking Claude (toolsmith) to generate a new tool")
        try:
            new_tool = self.toolsmith.generate_tool(query)
            log(f"toolsmith produced: name={new_tool.name!r}")
            log(f"  description: {new_tool.description}")
            props = list(new_tool.input_schema.get("properties", {}).keys())
            required = new_tool.input_schema.get("required", [])
            log(f"  input params: {props} (required: {required})")
        except ToolsmithInfeasibleError as e:
            reason = str(e)
            log(f"toolsmith reported infeasible: {reason!r}")
            msg = "this is impossible to do with a python script"
            if reason:
                msg = f"{msg} ({reason})"
            return finish(msg, "fallback")
        except Exception as e:
            log(f"toolsmith generate failed: {type(e).__name__}: {e}")
            return self._do_fallback(query, finish, log)

        raw, err, final_tool = self._run_with_repair(query, new_tool, log, already_cached=False)
        if raw is not None:
            self.catalog.add(final_tool)
            self.catalog.increment_usage(final_tool.name)
            log(f"saved to catalog as {final_tool.name}.json")
            answer = self._synthesize(query, final_tool, raw, log)
            return finish(answer, "generated", final_tool.name)
        log(f"generation + repair all failed (last error: {err})")
        return self._do_fallback(query, finish, log)

    def _run_with_repair(
        self,
        query: str,
        starting_tool: Tool,
        log: Callable[[str], None],
        already_cached: bool,
    ) -> tuple[Optional[object], Optional[str], Tool]:
        current = starting_tool
        last_error: Optional[str] = None

        for attempt in range(MAX_REPAIR + 1):
            if attempt > 0:
                log(f"repair attempt {attempt}/{MAX_REPAIR}: asking Claude to fix '{current.name}'")
                try:
                    current = self.toolsmith.repair_tool(query, current, last_error or "")
                    log(f"repaired tool produced: name={current.name!r}")
                except Exception as e:
                    log(f"repair failed: {type(e).__name__}: {e}")
                    return None, last_error, current

            log(f"extracting parameters for '{current.name}' (phi3.5)")
            try:
                args = local_llm.extract_parameters(query, current)
            except Exception as e:
                last_error = f"parameter extraction failed: {e}"
                log(f"extraction FAIL: {e}")
                continue
            log(f"extracted args: {args}")

            log(f"validating '{current.name}' in subprocess with those args")
            ok, err = self.validator.validate_args(current, args)
            if not ok:
                last_error = err
                log(f"validation FAIL: {err}")
                continue
            log("validation: PASS")

            log(f"executing '{current.name}' for this query")
            try:
                raw = self.executor.execute_with_args(current, args)
            except ToolExecutionError as e:
                last_error = str(e)
                log(f"execution FAIL: {e}")
                continue

            if looks_like_error(raw):
                last_error = f"tool returned error-shaped string: {str(raw)[:200]}"
                log(f"execution returned error shape: {str(raw)[:80]!r}")
                continue

            preview = str(raw).replace("\n", " ")[:100]
            log(f"raw tool output: {preview!r}")
            return raw, None, current

        return None, last_error, current

    def _synthesize(self, query: str, tool: Tool, raw_output, log: Callable[[str], None]) -> str:
        if not SYNTHESIZE:
            return str(raw_output)
        log("synthesizing natural-language answer (phi3.5 over tool output)")
        try:
            answer = local_llm.synthesize_answer(query, tool.name, raw_output)
            preview = answer.replace("\n", " ")[:80]
            log(f"synthesized: {preview!r}")
            return answer
        except Exception as e:
            log(f"synthesis failed ({e}); returning raw tool output")
            return str(raw_output)

    def _do_fallback(self, query: str, finish, log: Callable[[str], None]) -> QueryResult:
        log("falling back to direct Claude answer (no tool)")
        try:
            answer = self._claude_fallback(query)
            log(f"Claude answered ({len(answer)} chars)")
        except Exception as e:
            answer = f"Unable to answer: {e}"
            log(f"Claude fallback failed: {e}")
        return finish(answer, "fallback")

    def _claude_fallback(self, query: str) -> str:
        resp = self.toolsmith.client.messages.create(
            model=self.toolsmith.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": query}],
        )
        return resp.content[0].text


if __name__ == "__main__":
    server = AnvilServer(Path("data/tool_catalog"))
    while True:
        try:
            query = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.strip().lower() in ("exit", "quit"):
            break
        if not query.strip():
            continue
        result = server.handle_query(query)
        print(f"\n{result.output}\n")
        tail = f"[path: {result.path_taken}, latency: {result.latency_ms:.0f}ms"
        if result.matched_tool_name:
            sim = (
                f"{result.similarity_score:.2f}"
                if result.similarity_score is not None
                else "n/a"
            )
            tail += f", tool: {result.matched_tool_name} (sim={sim})"
        tail += "]\n"
        print(tail)
