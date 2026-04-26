import importlib.util
import multiprocessing as mp
import os
import tempfile
from typing import Optional

from anvil import local_llm
from anvil.models import Tool


TIMEOUT_SECONDS = 10

_ERROR_PREFIXES = (
    "error:", "error ", "failed:", "failed ", "exception:",
    "error fetching", "traceback", "unable to",
)


def looks_like_error(result) -> bool:
    if not isinstance(result, str):
        return False
    s = result.strip().lower()
    return any(s.startswith(p) for p in _ERROR_PREFIXES)


def _worker(impl_path: str, fn_name: str, args: dict, conn) -> None:
    try:
        spec = importlib.util.spec_from_file_location("anvil_tool_under_test", impl_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = getattr(mod, fn_name, None)
        if fn is None or not callable(fn):
            conn.send(("err", f"implementation did not define callable '{fn_name}'"))
            return
        result = fn(**args)
        if result is None:
            conn.send(("err", "tool returned None"))
            return
        if looks_like_error(result):
            preview = str(result)[:200]
            conn.send(("err", f"tool swallowed an exception and returned an error string: {preview}"))
            return
        conn.send(("ok", None))
    except Exception as e:
        conn.send(("err", f"{type(e).__name__}: {e}"))
    finally:
        try:
            conn.close()
        except Exception:
            pass


_CTX = mp.get_context("spawn")


class ToolValidator:
    def validate(
        self,
        tool: Tool,
        test_query: str,
        timeout: float = TIMEOUT_SECONDS,
    ) -> tuple[bool, Optional[str]]:
        try:
            args = local_llm.extract_parameters(test_query, tool)
        except Exception as e:
            return False, f"parameter extraction failed: {e}"
        return self.validate_args(tool, args, timeout)

    def validate_args(
        self,
        tool: Tool,
        args: dict,
        timeout: float = TIMEOUT_SECONDS,
    ) -> tuple[bool, Optional[str]]:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        tmp.write(tool.implementation)
        tmp.flush()
        tmp.close()

        try:
            parent_conn, child_conn = _CTX.Pipe(duplex=False)
            proc = _CTX.Process(
                target=_worker,
                args=(tmp.name, tool.name, args, child_conn),
            )
            proc.start()
            child_conn.close()
            proc.join(timeout)

            if proc.is_alive():
                proc.terminate()
                proc.join(1)
                if proc.is_alive():
                    proc.kill()
                    proc.join(1)
                return False, f"tool execution timed out after {timeout}s"

            if parent_conn.poll():
                status, payload = parent_conn.recv()
                if status == "ok":
                    return True, None
                return False, payload
            return False, f"tool process exited (code {proc.exitcode}) without returning a result"
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
