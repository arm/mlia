## Context

`PackagePathChecker` currently resolves one `BackendInfo.backend_path`. That
keeps existing installs simple, but cannot express a package layout where one
matched root contains a primary backend payload plus large, nested supporting
folders that also need to be installed.

`CompoundPathChecker` should remain first-match wins. Its checkers describe
alternative layouts; the selected checker should return all payload folders for
that one layout.

## Goals / Non-Goals

**Goals:**

- Preserve existing no-subfolder and single-subfolder behavior.
- Let one matched package checker describe a primary install source plus
  supporting folders.
- Copy supporting folders recursively while preserving their relative names.
- Keep the behavior local to path resolution and repository copy logic.

**Non-Goals:**

- Change `PyPackageBackendInstallation`, download, archive extraction, or EULA
  behavior.
- Add glob, wildcard, or per-file selection rules.
- Merge results from multiple successful `CompoundPathChecker` entries.

## Decisions

1. Keep `BackendInfo.backend_path` as the primary install source for
   compatibility, and add metadata for supporting folders.
2. Copy each supporting folder as a recursive directory tree into the installed
   backend directory under its relative folder name.
3. Keep `CompoundPathChecker` as a first-match selector; multi-folder copying
   comes only from the selected checker result.

## Risks / Trade-offs

- Sequence-like subfolder input can be confused with a string. Normalize strings
  separately and test both forms.
- Multi-folder copy adds a repository code path. Keep it narrow and test nested
  supporting folder contents.
