<!---
SPDX-FileCopyrightText: Copyright 2022-2023, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->
# MLIA Releases

These are the release notes for all MLIA releases. They document all of the
main feature changes, interface changes and issues that have been fixed.

The version numbering adheres to the [semantic versioning](https://semver.org/)
scheme.

### Trademarks and Copyrights

* Arm®, Cortex®-A, Ethos™-U, Corstone™ are registered trademarks or trademarks
  of Arm® Limited (or its subsidiaries) in the U.S. and/or elsewhere.
* TensorFlow™ is a trademark of Google® LLC.

## Release 0.7.0

### Internal changes

* Update the operator list of TensorFlow Lite Delegate
  for Arm NN 23.05 (MLIA-881)
* Generalize installation routines for backends by using registry (MLIA-825)
* Add contribution guidelines
* Update Vela (MLIA-757) and mypy (MLIA-849) dependencies

### Issues fixed

* Have the TFLite compability check display tables for all
  targets (MLIA-711)

## Release 0.6.0

### Feature changes

* Add metadata to the output console display and json file
  (MLIA-741/2 and MLIA-775)
* Support using TOML format for target profile and loading custom target
  profiles (MLIA-590)
* Support displaying a table of supported targets, backends and advice
  in the help text (MLIA-737)

### Interface changes

* **Breaking change:** Implement new CLI changes (MLIA-650)
* Create an interface to load/save/verify target profiles (MLIA-591)
* Remove the --output parameter (MLIA-782)
* Add the CLI parameter --output-dir (MLIA-461)

### Issues fixed

* Fix issues so that all available backends can be listed using CLI (MLIA-770)
* Provide a command for backend installation when it is not available and fix
  the issue of connection timeout (MLIA-460)

### Internal changes

* Update the compatibility list of TensorFlow Lite delegates operators
  for Arm NN 22.11 (MLIA-709)
* Expand the use of target/backend registries to avoid hardcoded names and
  cache target profiles (MLIA-769)
* Create an API to get target information (MLIA-589)
* Remove previous backend configuration mechanism (MLIA-595)
* Upgrade dependencies with Vela 3.6 (MLIA-669)
* Move tosa-checker functions into a separate module
* Migrate e2e tests to tox (MLIA-676)

## Release 0.5.0

### Feature changes

* Add TensorFlow Lite compatibility check for Cortex-A (MLIA-433)
* Add operator compatibility for Cortex-A (MLIA-411)

### Interface changes

* Remove support for CSV output (MLIA-275)
* Add "mlia-backend" command for managing backends (MLIA-649)
* Add performance for Ethos-U65-256 target profile (MLIA-618)

### Issues fixed

* Fix hyperlinks in README.md (MLIA-630)
* Fix TOSA checker dependency (MLIA-622)
* Fix backend install for Corstone-300 on AVH/VHT (MLIA-647)
* Fix --supported-ops-report flag (MLIA-688)

### Internal changes

* Update generic inference runner to 22.08 (MLIA-671)
* Use importlib for getting package version (MLIA-670)
* Make python syntax consistent across codebase
* Use tox to run additional project tasks (MLIA-571)
* Simplify typing in the source code (MLIA-386)
* Define incident response plan (MLIA-496)
* Enable testing for aarch64 (MLIA-584/MLIA-599)

### Known issues

* TOSA operator compatibility might raise a few errors after "Checking operator
  compatibility". These are known errors due to some issues present in
  TensorFlow 2.9 but they don't affect the correct result of the operator
  compatibility (MLIA-722/MLIA-723). This will be fixed in a future release of
  MLIA.

## Release 0.4.0

### Feature changes

* Add TOSA operator compatibility via tosa-checker python package
  (MLIA-548/549/579)

### Interface changes

* Update CLI to allow the usage of the TOSA checker (MLIA-550).

### Issues fixed

* Fix the issue that no performance information is shown for
  TensorFlow Lite files when the mode 'all_tests' is used (MLIA-552)
* Specify cache arena size in the Vela memory profiles (MLIA-316)

### Internal changes

* Merge the deprecated AIET interface for backend execution into MLIA
  (MLIA-546/551)
* Add pre-commit configuration (MLIA-529)
* Upgrade Vela version from 3.3.0 to 3.4.0 (MLIA-507)
* Update TensorFlow to version 2.8 (MLIA-569)

## Release 0.3.0

* Ethos-U operator compatibility, performance estimation and optimization
  advice
* Arm IP support:
   * Ethos-U55 via Corstone-300 and Corstone-310
   * Ethos-U65 via Corstone-300

  Note: Corstone-310 is available on AVH only.
