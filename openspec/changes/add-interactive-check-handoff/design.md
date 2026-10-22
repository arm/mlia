## Context

This change separates CLI preparation from check execution and connects an
optional terminal frontend. It defines the core contract; the `mlia-tui`
repository owns the screens, workers, transport limits, and package delivery.

## Goals / Non-Goals

**Goals:** Resolve options once through the existing CLI declarations; expose a
request that a frontend can edit before running; share validation, backend
selection, execution, persistence, and plugin dispatch; preserve ordinary CLI
behavior and resource ownership.

**Non-Goals:** Implement terminal widgets, replace the result schema, predict
automatic backend selection in a discovery API, introduce a second plugin option
protocol, or require arbitrary plugin objects to support process transport.

## Decisions

### Separate resolved requests from execution

`mlia.check` exposes `CheckSettings`, `CheckRequest`, `CheckResult`, `CheckError`,
`prepare_check`, and `execute_check`. Preparation captures resolved values and
copies MLIA-owned mappings while preserving plugin value/resource identity. It
does not select backends, install tools, create output directories, or run checks.
Completeness and cross-option execution validation belong in the shared executor.
Reconstructing a command line or repeating these rules in the frontend would
introduce a second interpretation of the same options.

The request carries model, target profile, backend choices, analysis flags,
profiling data, output base, JSON preference, EULA acceptance, noninteractive and
debug flags, resolved application settings, and dynamic plugin arguments.
Compatibility is the category default; both analysis flags remain representable.
`target_for_profile` supplies profile-to-target lookup through `mlia.api` without
exposing CLI internals or introducing a backend selection preview.

### Use the actual parser for standalone defaults

The public `mlia.cli.preparation.prepare_check_defaults()` context manager prepares the
standalone request through root/check CLI declarations, environment/configuration
loading, dynamic options, Click callbacks, and Typer conversions. The same
preparation path supplies normal CLI execution and integrated TUI launch.
Maintaining a dictionary of defaults or manually invoking plugin callbacks would
miss conversion and context behavior.

Acquire contexts before parsing and keep them alive until the frontend closes.
Release callback-owned resources on success, preparation failure, and frontend
failure. Copy declaratively prompted options before disabling their prompts for
standalone preparation. Required options lacking usable defaults/environment
values fail with an actionable configuration error. Callbacks that themselves
require terminal input are unsupported in standalone default preparation. The
normal CLI retains its existing prompt behavior; callback invocation counts are
not a new public guarantee.

### Dispatch only after shared preparation

`mlia check --tui` lazily imports `mlia_tui.launch(request)`. Missing editable
model/target fields are allowed at this boundary; syntactic option errors still
fail during parsing. The frontend waits for Run. A missing optional package
produces installation guidance. Ordinary CLI dispatch immediately invokes the
shared executor and retains strict completeness checks and lightweight help and
completion behavior. MLIA core does not depend on Textual.

### Keep execution and selection in core

`execute_check` owns completeness checks before output creation, the existing
validation/backend selection rules including profiling data, installation/EULA
questions, advisor execution, standardized persistence, analysis-plugin dispatch,
and logging cleanup. An optional prompt callback supports frontend-owned prompt
presentation; noninteractive policy suppresses installation questions without
turning an interactive frontend into an unattended command.

An optional caller-owned `output_stream` lets core construct its configured
console for captured execution. An explicit `console` takes precedence; without
either, rendered output is returned in `CheckResult.report`. Core does not close
the supplied stream. Frontends own third-party stdout/stderr containment.

The ordinary CLI retains its command-scoped plugin registry and resource-valued
options. An isolated frontend can rediscover plugins per attempt; it must not
assume parent registry instances or callback state transfer to its worker.
Serialization and runtime prompt restrictions belong to that frontend's contract,
not the core plugin protocol. No callback is rerun merely to reconstruct a worker
request.

### Return the latest accepted attempt

`CheckResult` contains canonical output, actual output directory, report, and an
explicit completed/skipped outcome with an optional skip reason. `CheckError`
carries an exit status. `launch` returns the latest result or no result and raises
the latest failure after terminal restoration. UI and cleanup failures also fail
the invocation; a prior successful result must not hide them.

| Integrated session on close | CLI outcome |
| --- | --- |
| No accepted Run | Exit 0; no JSON |
| Latest attempt completed | Exit 0; canonical JSON only when requested and present |
| Latest attempt skipped | Exit 0; reason on stderr; no JSON |
| Latest attempt failed | Error status, default 1; no JSON |
| Latest attempt cancelled or quit while active | Exit 130; no JSON |
| A later attempt succeeds after failure | Return that success |

An accepted Run clears the earlier return outcome even if worker startup fails.
Editing or cancelling a chooser/overwrite confirmation is not an attempt. An
incompatibility finding is a check result, not an execution failure. Standalone
application exit policy is owned by the companion TUI change.

## Risks / Trade-offs

- Parser coupling: the public defaults context manager deliberately depends on
  CLI declarations. Keep that dependency inside core and validate observable
  parity for dynamic options, environment values, conversions, and resource cleanup.
- Incomplete requests: early parsing cannot guarantee executability. Keep
  execution validation authoritative and ensure no output is created first.
- Resource and plugin state: ordinary CLI support can exceed worker transport
  support. Document ownership and require the TUI to reject unsupported transport
  before execution without changing values or globally narrowing plugin support.
- Editable versus published packages: editable installations can conceal missing
  APIs or incompatible plugins. Require built-package and clean-install evidence.

## Migration Plan

1. Review and harden the shared core API and ordinary CLI behavior, with focused
   behavioral tests followed by configured quality/build gates.
2. Publish a core version containing the documented API through the existing
   release workflow.
3. Set the TUI dependency floor from that actual published artifact; validate both
   entry points using clean installations before TUI publication.

## Open Questions

- Determine the actual minimum published core and target-plugin versions after
  publication; do not reserve an invented future version.
- The TUI release owner, distribution channel, and platform matrix are companion
  release decisions and must be settled before its publication.
