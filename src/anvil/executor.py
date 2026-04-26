import datetime as _datetime
import json as _json
import math as _math
import re as _re
import urllib as _urllib
import urllib.parse  # noqa: F401
import urllib.request  # noqa: F401
from typing import Any

import requests as _requests

from anvil import local_llm
from anvil.models import Tool


class ToolExecutionError(Exception):
    pass


_ALLOWED = {
    "requests": _requests,
    "datetime": _datetime,
    "json": _json,
    "re": _re,
    "math": _math,
    "urllib": _urllib,
}


class ToolExecutor:
    def execute(self, tool: Tool, query: str) -> Any:
        try:
            args = local_llm.extract_parameters(query, tool)
        except Exception as e:
            raise ToolExecutionError(f"parameter extraction failed: {e}") from e
        return self.execute_with_args(tool, args)

    def execute_with_args(self, tool: Tool, args: dict) -> Any:
        namespace: dict = dict(_ALLOWED)
        try:
            exec(tool.implementation, namespace)
        except Exception as e:
            raise ToolExecutionError(f"failed to compile tool '{tool.name}': {e}") from e

        fn = namespace.get(tool.name)
        if fn is None or not callable(fn):
            raise ToolExecutionError(
                f"tool implementation did not define callable '{tool.name}'"
            )
        try:
            return fn(**args)
        except Exception as e:
            raise ToolExecutionError(f"tool '{tool.name}' raised: {e}") from e
