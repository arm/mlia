<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

## Context

The core and plugin repositories already provide Markdown documentation and
MkDocs builds.
The developer documentation platform consumes the same content with navigation
configuration and page metadata.

## Goals / Non-Goals

**Goals:** Publish linked documentation, retain existing build formats, and
validate content before publication.

**Non-Goals:** Runtime behavior changes or replacing existing GitHub Pages and
offline documentation.

## Decisions

- Reuse `docs/source` and add compatible page front matter for routing and order.
  Store navigation configuration in `docs/docs-config.json`; the reusable
  workflow copies it into the content directory only during the build.
- Keep one reusable publishing workflow; callers supply site and environment
  settings and pin an available workflow revision.
- Match publishing access to the repository running the workflow.
- Validate pull requests without publishing. Publish `v*.*.*` tag pushes and
  manual runs after validation, using the selected ref. Ordinary main pushes do
  not publish. Serialize publication runs independently of PR validation.
  Use `queue: max` with cancellation disabled so newer runs do not replace
  pending publications (up to 100 pending runs per concurrency group).
  PR runs use separate groups per ref and also queue without cancellation.
  Limit OIDC permission to the shared publishing job; shared validation jobs
  explicitly use read-only contents permission, including on pull requests.
- Use canonical public Arm Developer and public source-repository links in
  reader-facing content. Keep deployment details out of published READMEs.
- Preserve linked notebook assets and existing MkDocs configuration.
- Add a custom React landing page only for the developer platform through
  `indexPageDirectory`. Keep its assets in a directory excluded by MkDocs.
  Give the Markdown index a separate route on the developer platform while
  retaining its content under Documentation in GitHub Pages. Support light/dark themes,
  responsive layouts, keyboard focus, and reduced-motion preferences.
  Use the approved Arm RGB palette with Ink and Silicon White as primary
  colours, and Electric Teal and Horizon Blue as detail accents. Combine a
  CLI discovery panel, an analysis capability strip, and a
  compact plugin directory to establish a distinct MLIA visual identity.
  Use the supplied stacked full-name Ink wordmark in light mode and its official White
  variant in dark mode, preserving both artworks unchanged. Exclude both logo
  assets from the GitHub Pages output.
  Centre the introduction above a separate analysis workspace with real discovery commands
  and vertical capability links. Avoid graphics that imply unsupported graph
  transformations or example analysis results. Place quickstart and plugin directory beside
  each other on wide screens, stacking them on smaller screens.

## Risks / Trade-offs

- Markdown differences → validate both documentation builders.
- Link differences → check canonical links and generated notebook assets.
- Access mismatch → align workflow callers with per-environment access manifests.

## Migration Plan

Provision publishing access through the owning infrastructure process. Publish
the core and plugin sites after validation and verify navigation and
downloadable assets.

## Plugin landing pages

Apply the MLIA landing-page identity to core and plugin documentation sites.
Plugin sites provide their own introduction, task links, and guide directory based on
existing documentation. Use the official theme-specific logos without changing
the artwork. Keep the same exclusion and alternate Markdown route in each
repository so GitHub Pages output is preserved.

## Release destinations

Plain `vX.Y.Z` tags publish to production. Tags with prerelease suffixes publish
to staging. Manual publications offer an explicit staging/production choice,
defaulting to staging. Select the corresponding AWS account for the publishing
role; pull requests remain validation-only.

## Consistent documentation homepages

Both builders consume the same landing content JSON and official logo assets.
MkDocs renders a Material template and generates a scoped stylesheet from the
React CSS module. Keep the existing Markdown homepage available as a generated
Documentation page, including offline builds. Support both Material theme modes.

## Product wording

Use approved public product names and context in reader-facing introductions
and cross-site labels. Keep repository names, URLs, backend IDs, and profile
names stable. Retain technical placement identifiers where they are part of
the output contract.

## Landing-page navigation and task panels

Use a right arrow for same-site documentation links and an external-link
indicator for absolute external destinations. External landing links open in
a new tab with `noopener noreferrer` and an accessible announcement. Apply
the convention to both renderers, including linked cards and guide directories.

Give plugin landing pages two distinct task-focused panels based on their
existing guides, with the complete guide directory retained below.

## Landing implementation maintenance

Keep MkDocs-specific overrides in a stylesheet, scope complete compound
selectors, and resolve landing links from one page lookup. Remove unused
styles but retain responsive cascade rules whose production rendering has
been verified at desktop, tablet, mobile, and dark settings.
