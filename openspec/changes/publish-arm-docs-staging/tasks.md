<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

## 1. Content

- [x] 1.1 Add compatible navigation metadata to the core and plugin documentation trees.
- [x] 1.2 Add canonical public links and preserve notebook downloads.
- [x] 1.3 Keep reader-facing READMEs free of deployment details.

## 2. Publishing

- [x] 2.1 Configure the documentation callers to use one pinned shared workflow.
- [x] 2.2 Align per-environment publishing access with caller identities.
- [x] 2.3 Preserve existing GitHub Pages and offline documentation builds.

## 3. Validation

- [x] 3.1 Validate core and plugin sites with both documentation builders.
- [x] 3.2 Validate workflows, navigation schemas, and notebook downloads.
- [x] 3.3 Remove private references and preview links from public prose.

## 4. Publishing schedule

- [x] 4.1 Match release-tag and manual triggers to GitHub Pages across all callers.
- [x] 4.2 Keep pull requests validation-only and serialize publication runs.
- [x] 4.3 Validate the scheduling change and update the existing draft PRs.

## 5. Shared homepage

- [x] 5.1 Improve the core Markdown homepage and align navigation entry points.
- [x] 5.2 Build both documentation formats and run relevant repository checks.
- [x] 5.3 Remove the repository-local container preview and its instructions.

## 6. Developer-platform landing page

- [x] 6.1 Add a custom landing page, diagram, responsive styling, and task links.
- [x] 6.2 Exclude custom assets from MkDocs and preserve its homepage output.
- [x] 6.3 Validate builds and inspect desktop, mobile, and dark-mode rendering.

## 7. MLIA visual identity

- [x] 7.1 Give the landing page a distinct model-analysis visual identity.
- [x] 7.2 Validate the revised build, responsive layout, and preserved MkDocs output.

## 8. Approved branding

- [x] 8.1 Apply the published Arm RGB palette and primary-colour hierarchy.
- [x] 8.2 Add the supplied MLIA logo without changing the logo artwork.
- [x] 8.3 Validate the approved-colour build, contrast, and responsive rendering.

- [x] 8.4 Verify the supplied logo loads and retains its aspect ratio in both themes.

## 9. Distinct landing page structure

- [x] 9.1 Separate the introduction, CLI discovery workspace, and toolkit directory.
- [x] 9.2 Verify responsive layout, theme logos, navigation, and build compatibility.

## 10. Plugin landing pages

- [x] 10.1 Add dedicated landing pages for plugin documentation sites.
- [x] 10.2 Use existing guides for content and retain Markdown homepages for MkDocs.
- [x] 10.3 Validate core and plugin sites, navigation, theme logos, and responsive layouts.

## 11. Learning Path entry points

- [x] 11.1 Promote relevant Learning Paths in plugin heroes and dedicated sections.
- [x] 11.2 Use approved public product names in landing-page copy.
- [x] 11.3 Validate both builds and relevant repository checks.

## 12. Stacked logo

- [x] 12.1 Use the official full-name stacked Ink and White logos across core and plugin landing pages.
- [x] 12.2 Verify asset imports, dimensions, and documentation builds.

## 13. Release destinations

- [x] 13.1 Route stable release tags to production and prerelease tags to staging across all documentation callers.
- [x] 13.2 Select the matching publishing role and allow an explicit manual destination.
- [x] 13.3 Validate workflow syntax and exercise release-routing cases.

## 14. Landing page background

- [x] 14.1 Give core and plugin landing pages an explicit full-width light/dark background.
- [x] 14.2 Validate builds and background rendering independently of the host background.

## 15. Consistent GitHub Pages landing pages

- [x] 15.1 Share landing content, logos, and styling between Docusaurus and MkDocs.
- [x] 15.2 Render MkDocs homepages with the matching layout and retain the original Markdown as Documentation.
- [x] 15.3 Validate core and plugin sites, offline links, theme switching, and browser layouts.

## 16. Product wording

- [x] 16.1 Review core and plugin landing pages against product naming guidance.
- [x] 16.2 Use approved public product names and context in landing introductions.
- [x] 16.3 Rebuild documentation, verify wording, and refresh demo packages.

## 17. Landing-page feedback

- [x] 17.1 Add paired plugin panels and task-focused documentation entry points.
- [x] 17.2 Standardise internal and external indicators and open external destinations in new tabs.
- [x] 17.3 Verify both renderers, navigation, offline output, and refreshed demos.

## 18. Landing code cleanup

- [x] 18.1 Remove unused CSS while retaining production-sensitive cascade rules.
- [x] 18.2 Simplify page-link lookup and keep MkDocs-specific styles outside Python.
- [x] 18.3 Verify builds and compare rendered styles against the pre-cleanup baseline.

## 19. Publication queue review

- [x] 19.1 Preserve pending publications using the maximum concurrency queue.
- [x] 19.2 Check path filters against the current shared-workflow location.
- [x] 19.3 Validate workflow configuration and queue policy across callers.

## 20. Runner and permission review

- [x] 20.1 Encode scalar runner labels for callers using scalar repository variables.
- [x] 20.2 Verify the pinned shared workflow limits OIDC to non-PR publishing jobs.
- [x] 20.3 Check runner encoding, workflow YAML, and front-matter compatibility.

## 21. Documentation script location

- [x] 21.1 Move the landing hook into `docs/scripts/` and update its stylesheet path.
- [x] 21.2 Verify strict online and offline MkDocs builds and repository checks.

## 22. Dedicated runner variable

- [x] 22.1 Use the JSON-formatted ARM64 runner variable across documentation callers.
- [x] 22.2 Validate direct runner decoding and shared-workflow runner inputs.
