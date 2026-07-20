## Context

Shell completion has two different initialization needs:

- Command and option-name completion needs the Typer command tree, including
  check options registered by available backend plugins.
- Backend value completion needs the authoritative selectable names from those
  same registrations.

Fresh-process measurements showed that importing `mlia.api` pulls in backend
installation, target, advice, reporting, validation, and third-party runtime
code. On the measured core environment, current-main root completion took about
0.34 seconds. With the realistic installed plugin stack, the reworked command
tree and value-completion paths take about 0.21 to 0.24 seconds because backend
plugin implementations are imported. That latency is acceptable; the
implementation must describe it honestly rather than claim that
plugin-controlled imports remain absent.

The existing plugin architecture makes package import the initialization
boundary: importing `mlia.backend` loads backend plugins, and importing
`mlia.target` loads backend and target plugins. This change preserves that
contract.

## Goals / Non-Goals

**Goals:**

- Build command and option completions without importing the runtime-heavy MLIA
  API or target surfaces.
- Preserve plugin-provided check options during shell completion.
- Complete registered selectable backend names without querying installation
  state.
- Complete packaged target profile names without importing the target runtime.
- Preserve normal help, execution, list commands, public APIs, and plugin-owned
  initialization behavior.
- Cover the boundaries with behavior and fresh-process import tests.

**Non-Goals:**

- Do not change backend, target, or CLI plugin initialization contracts.
- Do not change plugin-owned import or lazy-loading strategies.
- Do not add a plugin metadata contract, cache, or static backend inventory.
- Do not derive selectable backend names from entry point names.
- Do not make completion installation-state-aware.
- Do not change general runtime resource discovery.
- Do not enforce wall-clock thresholds in unit tests.

## Decisions

### Keep generic command construction lightweight

Top-level imports remain the default. Imports are deferred only where their
transitive cost or runtime initialization is relevant to completion:

- `mlia.api` and operations exposed through it;
- the installation/download stack referenced only by backend configuration type
  annotations;
- command validation that imports the target registry;
- target configuration and registry access.

Small CLI helpers, execution context types, logging helpers, Rich panels and
tables, completion helpers, and UI helpers remain normal top-level imports.

Backend option metadata moves to a small backend-owned option module because it
is derived solely from registered backend configurations. The registry remains
the authoritative data source, while the option module owns the CLI-specific
representation shared by the command tree and public API. Importing that module
follows the existing `mlia.backend` package boundary, so shell completion
includes plugin-provided check options without importing `mlia.api`, backend
installation state, or the target runtime. Normal `mlia check --help` uses the
same metadata. The API directly re-exports the existing metadata types and
function for compatibility; this change does not alter their shape or the
plugin interface.

### Preserve package import as the plugin boundary

The CLI imports the backend option module and registry. Python initializes
`mlia.backend` first, so the existing package initialization loads available
backend plugins before the command tree reads registered option metadata or
completion reads registered names. The change does not add or distribute new
initialization calls across consumers.

Plugin packages remain responsible for their own import behavior. Completion may
therefore inherit any runtime imports performed by installed backend plugin
implementations.

### Use registered selectable names without installation filtering

The backend registry is the authoritative mapping from plugin registrations to
CLI backend names. Completion sorts its names and keeps configurations marked
`selectable`. It does not construct an installation manager or inspect whether
a backend is installed.

The same inventory is used for `check`, `install`, and `uninstall`. This is
stable across commands and keeps already-installed names available for
`install --force`.

Entry point names are not a suitable substitute: one plugin can register
multiple backend names, entry point and backend naming conventions can differ,
and some registered configurations are intentionally non-selectable.

### Scan packaged target profiles directly

Target-profile completion scans `resources/target_profiles/*.toml` beneath each
active `mlia.__path__` entry, then sorts and deduplicates the stems. It does not
import `mlia.target`, target configuration, the target registry, or target
plugins.

The existing runtime resource helper has broader discovery behavior.
Completion neither invokes nor changes that pre-existing path.

### Keep only the measured top-level lazy boundary

The root `mlia` package continues to expose its documented API. API functions
and dynamic constants modules are resolved on first access because resolving
them imports runtime-heavy or plugin-backed surfaces. Cheap public error classes
remain eager imports. The dynamic constants modules themselves and their
`__all__` behavior remain unchanged.

## Risks / Trade-offs

- Command-tree and backend completion latency depends on installed plugin import
  behavior. This is accepted because registered options and names are part of
  the authoritative CLI inventory.
- Direct namespace resource scanning is intentionally narrower than runtime
  resource discovery. Packaged MLIA namespace paths remain covered, while custom
  profile paths rely on shell filename fallback.
- Lazy root exports add indirection. The boundary is limited to public exports
  with demonstrated runtime or plugin initialization cost and is covered by
  compatibility tests.
