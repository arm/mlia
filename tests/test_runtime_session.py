# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Run model-file conversion through a profile and inspect its outputs and plan."""

import json
from pathlib import Path
from typing import Any

import pytest

from mlia.core.errors import ConfigurationError
from mlia.runtime.catalog import RuntimeDefinition, load_profile
from mlia.runtime.session import current_session, execution_session, invoke
from mlia.runtime.transformer import RuntimeTransformer
from mlia.transformers.registry import TransformRequest, transform_model
from mlia.utils.registry import Registry
from tests.test_runtime_execution import runtime


class FileConverter:
    """A provider that declares support but must run its conversion in a worker."""

    def supports(
        self,
        model: object,
        target_format: str,
        transform_options: dict[str, Any] | None = None,
    ) -> bool:
        return target_format == "tosa"

    def __call__(self, model: object, output_dir: Path, **kwargs: Any) -> Path:
        pytest.fail("Conversion must run in the selected runtime")


def test_conversion_uses_profile_options_and_records_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLIA_RUNTIME_HOME", str(tmp_path / "installed"))
    definition = runtime(
        tmp_path,
        """from pathlib import Path
def run(request):
    payload = request['payload']
    output = Path(request['output_dir']) / 'converted.tosa'
    text = Path(payload['model']).read_text() * payload['options']['copies']
    output.write_text(text + payload['options']['suffix'])
    return {}, [{'kind': 'model', 'path': output.name}]
""",
    )
    definition.data["operations"]["echo"]["result_schema"] = "model/1"
    definition.path.write_text(json.dumps(definition.data))
    definition = RuntimeDefinition.load(definition.path)
    profile_path = tmp_path / "convert.profile.toml"
    profile_path.write_text(
        'schema_version = 1\nid = "test/convert"\n'
        '[stages.convert]\noperation = "echo"\n'
        f'runtime = "{definition.name}"\n'
        '[stages.convert.options.options]\ncopies = 1\nsuffix = "!"\n'
    )
    profile = load_profile(profile_path, {definition.name: definition})
    converter = RuntimeTransformer(FileConverter(), "echo")
    registry: Registry[RuntimeTransformer] = Registry()
    registry.register(
        "unselected", RuntimeTransformer(FileConverter(), "other/convert")
    )
    registry.register("selected", converter)
    monkeypatch.setattr("mlia.transformers.registry.transformer_registry", registry)
    model = tmp_path / "input.tflite"
    model.write_text("model")
    assert converter.supports(model, "tosa")
    # An analysis failure after conversion must still close the session.
    with pytest.raises(ValueError, match="analysis failed"):
        with execution_session(profile, tmp_path / "analysis", accept_eula=False):
            result = transform_model(
                TransformRequest(model, tmp_path / "requested", "tosa", {"copies": 2})
            )
            assert result == tmp_path / "requested/converted.tosa"
            assert result.read_text() == "modelmodel!"
            raise ValueError("analysis failed")
    assert current_session() is None
    with pytest.raises(ConfigurationError, match="execution profile"):
        invoke("echo", {})
    plan = json.loads((tmp_path / "analysis/execution-plan.json").read_text())
    assert plan["profile"] == "test/convert"
    assert plan["calls"][0]["runtime"] == definition.name
    assert Path(plan["calls"][0]["artifacts"][0]["path"]).read_text() == "modelmodel!"
    assert profile.data["stages"]["convert"]["options"]["options"]["copies"] == 1
