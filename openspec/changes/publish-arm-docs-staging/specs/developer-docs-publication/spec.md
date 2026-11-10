<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

## ADDED Requirements

### Requirement: Publish core and plugin documentation

The system SHALL publish MLIA core and plugin documentation from their existing
Markdown sources.

#### Scenario: Release tag

- **WHEN** a `v*.*.*` release tag is pushed and documentation passes validation
- **THEN** its corresponding site is published to the configured environment

### Requirement: Validate before publishing

The workflow SHALL validate pull requests without publishing. Publishing MUST
require validation success and either a release-tag push or a manual run.

#### Scenario: Documentation pull request

- **WHEN** a documentation pull request runs
- **THEN** validation runs without invoking publication or assuming its role

### Requirement: Repository-scoped publishing access

Access manifests SHALL authorize the repository running each publishing caller
for its matching site ID in the configured environment.

#### Scenario: Matching workflow identity

- **WHEN** a repository publishes its documentation
- **THEN** its configured role authorizes that repository and site

### Requirement: Preserve existing documentation builds

Existing GitHub Pages and offline builds SHALL remain usable. Linked notebook
downloads SHALL match the source assets.

#### Scenario: Download a walkthrough

- **WHEN** a reader follows a notebook download link
- **THEN** the original notebook asset is available in the generated site

### Requirement: Public documentation references

Reader-facing documentation SHALL use canonical public documentation and source
links. It MUST NOT include private infrastructure references or preview-site URLs.

#### Scenario: Open a documentation reference

- **WHEN** a reader follows a README or cross-site documentation link
- **THEN** the link targets the canonical public documentation site

### Requirement: Reuse publishing jobs

The documentation callers SHALL use one reusable publishing workflow with a pinned
available revision and caller-supplied site settings.

#### Scenario: Shared workflow invocation

- **WHEN** a caller validates or publishes its documentation
- **THEN** the shared workflow uses the caller's site and environment settings

### Requirement: Independent and release-driven publishing

The workflow SHALL support manual publication of the selected ref and automatic
publication for `v*.*.*` tag pushes, matching GitHub Pages triggers. Ordinary
branch pushes MUST NOT publish documentation. Publication runs SHALL be
serialized per repository with `queue: max` and cancellation disabled, while
pull-request validation remains independent. Up to 100 pending publications
SHALL be retained instead of replaced by newer runs in the same group.

#### Scenario: Manual documentation publication

- **WHEN** a user manually runs the documentation workflow on a selected ref
- **THEN** that ref is validated and published independently of a release

#### Scenario: Closely spaced publications

- **WHEN** a publication is running and two more publications enter its group
- **THEN** both pending publications remain queued without replacing each other

#### Scenario: Ordinary main-branch update

- **WHEN** a commit is pushed to main without a release tag
- **THEN** it does not trigger developer documentation publication

### Requirement: Separate developer-platform landing page

All core and plugin developer-platform documentation sites SHALL have custom
landing pages with repository-specific introductions and task links. The core site SHALL also
provide a plugin directory. GitHub Pages SHALL render the same landing content, branding, and layout through
a MkDocs template. The original Markdown homepage SHALL remain available as a
Documentation page. React source files MUST NOT be copied into the published site.

#### Scenario: Open the developer site

- **WHEN** a reader opens the developer-platform site root
- **THEN** the custom landing page appears with working links to documentation

#### Scenario: Build GitHub Pages

- **WHEN** the MkDocs site is built
- **THEN** the landing page is rendered with shared content and assets, while original
  Markdown remains available under Documentation

#### Scenario: Browse on a narrow screen or in dark mode

- **WHEN** the landing page is displayed on a mobile viewport or in dark mode
- **THEN** content remains readable, reachable, and free of horizontal overflow

### Requirement: Route publication by release type

Plain `vX.Y.Z` tags SHALL publish to production. Prerelease tags SHALL publish
to staging. Each destination SHALL use its matching AWS publishing role. Manual
runs SHALL offer an explicit destination and default to staging.

#### Scenario: Stable release

- **WHEN** tag `v1.2.3` triggers publication
- **THEN** validation precedes publication to production using its publishing role

#### Scenario: Prerelease

- **WHEN** tag `v1.2.3rc1` or `v1.2.3-rc.1` triggers publication
- **THEN** validation precedes publication to staging using its publishing role
