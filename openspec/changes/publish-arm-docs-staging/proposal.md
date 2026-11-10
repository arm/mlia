<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

## Why

MLIA and its plugins need accessible documentation on Arm Developer while
preserving their existing documentation formats.

## What Changes

- Publish MLIA core and plugin documentation.
- Reuse existing Markdown with compatible navigation metadata.
- Add canonical public documentation links to READMEs and navigation.
- Validate documentation before publication through one shared workflow.
- Preserve GitHub Pages, offline builds, and notebook downloads.
- Add consistent landing pages for the developer platform and GitHub Pages,
  keeping the original Markdown available under Documentation.

## Capabilities

### New Capabilities

- `developer-docs-publication`: Validate and publish the core and plugin documentation sites.

### Modified Capabilities

None.

## Impact

Documentation metadata, links, and workflow callers in the core and plugin
repositories.
Publishing access is configured separately for each deployment environment.
