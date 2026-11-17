# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Run a provider's operation inside its runtime and write the result back to core.

The manager passes request and result file paths to this script. It loads the
requested entrypoint and wraps its output or exception in a JSON response.
It uses only the standard library because MLIA is not installed in the runtime.
"""

import importlib
import importlib.metadata
import json
import os
import sys
import traceback
from pathlib import Path


def main() -> int:
    """Invoke a worker adapter and publish its completion envelope."""
    request_path, result_path = map(Path, sys.argv[1:])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    result = {
        key: request[key]
        for key in ("protocol", "job_id", "runtime_id", "operation", "result_schema")
    }
    try:
        if request["protocol"] != 1:
            raise ValueError("Unsupported worker protocol")
        module, function = request["entrypoint"].split(":")
        provider = importlib.import_module(module)
        adapter = getattr(provider, function)
        if not callable(adapter):
            raise TypeError("Runtime entrypoint must be callable")
        if request.get("handshake"):
            validator = getattr(provider, "validate_runtime", None)
            if validator is not None:
                validator(request)
            payload, artifacts = (
                {
                    "python": sys.version,
                    "executable": sys.executable,
                    "packages": {
                        dist.metadata["Name"]: dist.version
                        for dist in importlib.metadata.distributions()
                    },
                },
                [],
            )
        else:
            payload, artifacts = adapter(request)
        result.update(status="success", payload=payload, artifacts=artifacts)
    except Exception as exc:
        traceback.print_exc()
        result.update(
            status="error", error={"type": type(exc).__name__, "message": str(exc)}
        )
    temporary = result_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    os.replace(temporary, result_path)
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
