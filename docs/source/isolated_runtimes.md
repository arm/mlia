<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Isolated backend runtimes

MLIA installs backend tools and their Python dependencies in separate virtual
environments, called runtimes. This lets you use backends with incompatible
dependencies without changing the Python environment where you installed MLIA.

Backend packages come with their MLIA plugins. The first check installs the
selected tools automatically; later checks reuse them. Setup shows progress and
obtains the required Python interpreter if it is missing. Linux is supported;
required operating-system libraries must already be available on your machine.

## Run with the defaults

Install the target plugin and any converter plugin needed for your model format,
then run a check as usual:

```bash
mlia check -t neural-technology model.vgf --performance
```

MLIA chooses the execution profile matching the backend and input format. An
execution profile selects the versions of the converters and backend tools used
by that check. To inspect those choices before installation or execution:

```bash
mlia check -t neural-technology model.vgf --performance --show-plan
```

A missing converter plugin produces an error identifying the package to install.
MLIA installs backend dependencies automatically, but does not install frontend
plugins on your behalf.

## Choose backend versions

List the available runtime IDs and the operations they provide:

```bash
mlia backend list --runtimes
```

IDs such as `arm/nx/public@VERSION` identify a particular backend variant and
version. Current runtime versions follow the corresponding plugin package version.
Several versions may be available together.

Runtime providers may set `label` in `[tool.mlia.runtime]` and in each
`[tool.mlia.runtime.operations."OPERATION_ID"]` table. MLIA uses these labels in
installation and model-run progress messages. Older runtime manifests without
labels display their IDs in a readable form.

Select a supplied execution profile by name or pass a custom TOML file:

```bash
mlia check -t neural-technology model.pt2 --performance \
  --execution-profile arm/nx-pt2
mlia check -t neural-technology model.vgf --performance \
  --execution-profile ./my-profile.toml --show-plan
```

You can also set `MLIA_EXECUTION_PROFILE` for CLI and API calls. An explicit
selection is never silently replaced with another version. If multiple defaults
match, MLIA asks you to select a profile explicitly.

To customize a supplied profile, copy its `*.profile.toml` file from the plugin's
`mlia/resources/runtimes` directory and change its ID and runtime selections.
For example, this profile chooses one estimator for VGF models. Replace `VERSION`
with a version shown by `mlia backend list --runtimes`:

```toml
schema_version = 1
id = "my-nx-vgf"
backend = "nx-performance-estimator"
inputs = ["vgf"]

[stages.execute]
operation = "nx/execute"
runtime = "arm/nx/public@VERSION"
```

Keep the converter stages when customizing a profile for converted inputs such
as TFLite or PT2. Use operation IDs listed by the installed plugins. A profile
selects implementations of those operations; it does not define a new workflow.
For each conversion, MLIA uses a converter selected in the profile that supports
the input, required output format and conversion options. If multiple selected
converters support the same request, keep only one; registry order does not
choose between them.
For a combination of backends, use `backends = ["vela", "corstone-300"]` in place
of `backend`.

A profile can accept several input formats. Add `inputs = ["pt2"]` to a stage
to select it only when the original model is a PT2 file. Stages without `inputs`
apply to every format accepted by the profile. Unselected stages do not require
their provider plugins and their runtimes are not installed. For example, an
Ethos-U profile can share its Vela stage across TFLite and PT2 inputs while
selecting the PyTorch converter stage only for PT2.
Checks with profiling data but no model skip stages that declare input formats.

Validate the edited profile with `--show-plan`, then omit that option to run the
check. Validation checks the selections; the backend still needs to accept the
contents of your model. Stage `options` provide defaults, overridden by options
supplied for the current check.

## Install, inspect and remove backends

You can install a backend before running a check:

```bash
mlia backend install nx-performance-estimator --accept-eula
mlia backend list
mlia backend uninstall nx-performance-estimator
```

A backend name selects its default version. To manage another available version,
use its complete runtime ID in the install or uninstall command. Installing a
version does not change an execution profile's selection.

`--force` rebuilds the selected environment. If replacement fails, MLIA restores
the previous installation. A rebuild may resolve newer transitive dependencies
where the backend package does not pin them.

`--path` installs a local backend package. Use the package format supplied for
that backend; a native executable archive alone may not contain the information
needed to install its dependencies. MLIA retains locally supplied runtime files
for later reinstallation, even after you remove their original location.

Runtime environments are stored under `~/.mlia/runtimes`, in directories such as
`arm/nx/public@VERSION/x86_64/venv`. Set `MLIA_RUNTIME_HOME` to use another store.
You can inspect an environment without activating it:

```bash
uv pip list --python "$HOME/.mlia/runtimes/arm/nx/public@VERSION/x86_64/venv/bin/python"
```

If you receive additional backend catalog files with a delivery, set
`MLIA_RUNTIME_CATALOG` to their directory before listing or selecting them.

## Licence acceptance

Interactive installation displays the full licence and asks for acceptance.
Declining or ending input stops installation. A previously installed backend does
not ask again unless you request a forced reinstall.

For noninteractive use, explicitly accept the applicable licences with:

- `--accept-eula` for `mlia backend install`.
- `--i-agree-to-the-contained-eula` for `mlia check`.
- `accept_eula=True` for the Python API.

`--noninteractive` prevents prompts; it does not imply acceptance.

## Python API

```python
from pathlib import Path
from mlia.api import run_advisor

output = run_advisor(
    "performance", "neural-technology", Path("model.pt2"),
    execution_profile="arm/nx-pt2", accept_eula=True,
    write_output_files=True, output_dir="analysis",
)
```

Live `nn.Module` inputs are exported using Torch in your calling environment and
`example_inputs`. The selected converter reads that exported file. If it cannot
read the export, MLIA reports the incompatibility without replacing your selection.
Torch is needed in the calling environment only when using live Torch objects.

## Results and troubleshooting

The analysis output directory contains `execution-plan.json`, which records the
selected versions and completed operations. Each `runtime-jobs` subdirectory
contains the request, result, output files and `worker.log` for one operation.
Installation diagnostics are in `install.log` inside the runtime directory.
Failed installations retain their latest log beside that directory as
`<architecture>.failed.log`; partial environments are removed, and a failed
reinstall preserves the previously working runtime.

Use those logs when installation fails or a backend rejects a model. Failed jobs
retain their diagnostics. Runtime isolation keeps Python dependencies separate;
it does not make untrusted tools or model files safe to execute.
