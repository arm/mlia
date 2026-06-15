## Context

MLIA's plugin system allows external packages to extend the core product without
changing the core package. The system crosses several parts of MLIA: entry
point discovery, compatibility checks, backend and target registries, CLI
command construction, and converter registration.

## Goals / Non-Goals

**Goals:**

- Provide a plugin system that can load external extensions through Python entry
  points.
- Ensure incompatible or broken plugins do not prevent MLIA from starting and
  serving valid plugins.
- Make plugin-provided targets, backends, commands, and converters available
  through the same APIs used by built-in functionality.

**Non-Goals:**

- Specify plugin packaging or publishing instructions.
- Define the behavior of individual backend, target, CLI, or converter plugins.
- Introduce a broader plugin lifecycle beyond discovery, compatibility checks,
  and registration.

## Decisions

### Separate plugin types by entry point group

The architecture separates plugin types by entry point group:

- `mlia.plugin.backend`
- `mlia.plugin.target`
- `mlia.plugin.cli`
- `mlia.plugin.converter`

This separation matters because each plugin type receives a different host
object:

- backend plugins register backend definitions into a backend registry
- target plugins register target definitions into a target registry
- CLI plugins mutate the list of CLI command descriptors
- converter plugins register named converters into the registry object passed to
  the loader

The core loader is generic, but each integration surface stays explicit.

Alternative considered: use one generic plugin group with type dispatch inside
the plugin body.

Why not chosen: separate groups keep discovery and host integration simpler.

### Use entry points for discovery and extended package paths for importability

The system uses two mechanisms that serve different purposes:

- entry point groups are the authoritative discovery mechanism for plugins
- extended package paths (`pkgutil.extend_path`) allow backend and target
  subpackages to be importable from multiple distributions

Entry points are used for discovery because they provide explicit registration
and package metadata:

- MLIA can ask for exactly one plugin group at a time
- MLIA can tell which distribution provided each plugin
- MLIA can validate external plugin compatibility against the installed core
  version before importing the plugin module
- MLIA can report discovered plugins through CLI inventory views

Extended package paths solve a different problem. They allow Python modules
under `mlia.backend.*` and `mlia.target.*` to live in multiple distributions and
still be importable through a shared package namespace. That is useful for code
organization and for backend-specific module discovery, but it is not sufficient
as the primary plugin-discovery mechanism because it does not by itself provide:

- explicit opt-in for a plugin group
- distribution-level compatibility metadata at the discovery point
- a clean way to discover CLI and converter plugins, which are not organized as
  importable backend or target subpackages

Alternative considered: discover plugins by scanning namespace packages instead
of using entry points.

Why not chosen: namespace scanning would be less explicit, would couple
discovery to package layout, and would make compatibility checks and plugin
inventory reporting harder.

### Keep the plugin contract minimal

A plugin is expected to expose:

- `plugin_interface_version`
- `register(registry_or_host)`

`register(...)` is the only integration hook. MLIA creates or provides the host
object and the plugin populates it.

Alternative considered: introduce a larger plugin lifecycle with multiple hooks.

Why not chosen: the current registry-centered architecture does not need it.

### Gate external plugins on core compatibility and interface version

The loading flow for one group is:

1. Identify whether the plugin is internal or external.
2. For external plugins, verify compatibility against the installed `mlia`
   version.
3. Import the plugin module.
4. Check `plugin_interface_version`.
5. Call `register(...)`.

The loader compares the plugin distribution against the `mlia` console script's
distribution:

- If they match, the plugin is treated as internal.
- If they do not match, the plugin is treated as external.

External plugins must declare an `mlia` dependency range that includes the
installed core version. There is a second compatibility gate at the plugin
interface layer, where MLIA currently accepts
`plugin_interface_version == "0.0.1"`.

Alternative considered: trust all installed entry points and fail fast on
incompatibility.

Why not chosen: third-party plugins should not be able to destabilize the host
process so easily.

### Initialize plugin-backed surfaces consistently

Plugin-backed surfaces should initialize their plugins at the package boundary
rather than mixing package-import loading with ad hoc loader calls buried inside
later API construction.

Examples:

- importing `mlia.backend` loads backend plugins during package initialization
- importing `mlia.target` loads backend plugins, then target plugins, during
  package initialization
- importing `mlia.cli` should load CLI plugins into the shared command
  configuration during package initialization
- backend registry access ensures backend plugins are loaded
- target registry access ensures backend plugins, then target plugins, are
  loaded
- converter plugins load only when the converter loader is called

This keeps backend, target, and CLI initialization symmetrical: each package
owns the registry or command configuration for its surface and populates it as
part of package initialization. Public APIs can still defensively ensure the
relevant plugins are present, but they should not be the primary discovery hook.

Alternative considered: load CLI plugins only when command construction asks for
the command list.

Why not chosen: it makes CLI plugin loading inconsistent with the other plugin
surfaces, spreads discovery semantics across multiple APIs, and forces command
construction to own plugin initialization rather than just consume initialized
state.

### Build around simple registries

The architecture is built around simple registries rather than a large service
container.

That has a few consequences:

- plugins mainly add named items
- host code stays explicit about which registry it depends on
- testing is straightforward because registries are easy to fake or inspect

The target registry depends on the backend registry. A target definition asserts
that its referenced backends are already registered, which is why target flows
load backend plugins before target plugins.

The CLI surface should follow the same pattern by treating its command
configuration as package-owned initialized state rather than as a fresh list
that triggers plugin loading during command construction.

Alternative considered: centralize plugin data in a more abstract service layer.

Why not chosen: the current extension model is mostly registration, not service
composition.

### Expose plugin metadata and keep registry behavior stable

The plugin registry utilities expose entry point metadata for CLI reporting:

- name
- group
- entry point value
- distribution name
- distribution version

Converter plugins are intentionally simple. They register named converters into
the registry object supplied by the loader, and duplicate registrations keep the
existing entry while logging a warning.

Alternative considered: let duplicate converter registrations replace existing
entries.

Why not chosen: preserving the existing converter favors stability over implicit
replacement.

## Risks / Trade-offs

- [Plugin compatibility checks can reject imperfect packages] → Prefer explicit
  rejection to ambiguous runtime behavior.
- [CLI plugin loading depends on package-owned initialized command state] →
  Keep command initialization explicit so command construction remains a read
  path over already-populated state.
- [The system uses both entry points and extended package paths] → Keep their
  responsibilities separate: entry points for discovery and compatibility,
  extended package paths for importability and module layout.
- [A bad plugin may still log noisy failures] → Continue isolating failures so a
  broken plugin does not abort MLIA.
