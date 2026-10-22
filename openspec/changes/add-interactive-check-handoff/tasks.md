## 1. Review the public boundary

- [x] 1.1 Review the request, settings, result, error, execution, defaults-context, and profile-lookup contracts against the companion TUI proposal; resolve naming and support commitments.
- [x] 1.2 Audit ordinary CLI callers and current plugin contracts; keep unrelated canonical plugin drift outside this change.

## 2. Harden shared preparation

- [x] 2.1 Review and complete request preparation so all declared core and dynamic options retain their resolved values and plugin resource identity without execution side effects.
- [x] 2.2 Review and complete standalone defaults through actual root/check parsing, configuration/environment loading, conversions, prompt suppression, and context lifetime management.
- [x] 2.3 Add or refine behavior tests for typed/environment defaults, required missing options, callback failure cleanup, resource lifetime, and ordinary CLI prompt preservation; reuse existing coverage where sufficient.
- [x] 2.4 Review and document named/file-backed profile lookup through the public API and verify invalid-profile errors without exposing CLI internals.

## 3. Harden shared execution and dispatch

- [x] 3.1 Review and complete the shared execution lifecycle, preserving validation before output creation, profiling-data selection, combined categories, installation/EULA policy, canonical persistence, plugin dispatch, and logging cleanup.
- [x] 3.2 Review and complete optional TUI dispatch, incomplete editable inputs, actionable missing-package handling, and preservation of supplied options without automatic execution.
- [x] 3.3 Verify observable CLI parity with and without the TUI installed: strict ordinary inputs, no-argument help/exit behavior, dynamic options, lightweight completion, text/JSON reporting, skip behavior, and command-scoped plugins.
- [x] 3.4 Verify integrated latest-attempt results, cancellation status 130, later-success recovery, fatal startup/cleanup failure, and JSON emission after terminal restoration; cover each distinct outcome without mirroring internal helpers.
- [x] 3.5 Verify ordinary CLI analysis-plugin dispatch and post-analysis failure after persistence without altering plugin discovery or compatibility checks.

## 4. Document and validate the production API

- [x] 4.1 Document public entry points, option ownership, incomplete requests, context/resource lifetime, prompt policy, result/error behavior, and frontend-specific transport limits with small usage examples.
- [x] 4.2 Run configured relevant unit/integration suites, coverage gate, static checks, documentation build, and package build; distinguish unresolved environment issues from code failures.
- [ ] 4.3 Test the built core package with representative compatible plugins and the companion TUI in a clean installation; verify both entry points and ordinary CLI behavior without editable dependencies.
- [x] 4.4 Complete independent scope, compatibility, dependency, public/private boundary, and release-readiness review; reconcile each requirement with evidence or an explicit unresolved task.

## 5. Publish and hand off dependency evidence

- [ ] 5.1 Complete the core review and release workflow through the established distribution channel and verify the published artifact includes the documented public APIs.
- [ ] 5.2 Record the actual published minimum core/plugin versions for the companion TUI dependency update and clean-install gate; do not substitute a development checkout or an invented version.
- [ ] 5.3 Reconcile the completed implementation with these specifications before archival, preserving any unfinished work explicitly.
