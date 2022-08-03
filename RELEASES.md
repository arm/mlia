<!---
SPDX-FileCopyrightText: Copyright 2022, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->
# MLIA Releases

These are the release notes for all MLIA releases. They document all of the
main feature changes, interface changes and issues that have been fixed.

The version numbering adheres to the [semantic versioning](https://semver.org/)
scheme.

### Trademarks and Copyrights

Arm®, Ethos™-U, Corstone™ are registered trademarks or trademarks of Arm®
Limited (or its subsidiaries) in the U.S. and/or elsewhere.
TensorFlow™ is a trademark of Google® LLC.

## Release 0.4.0

### Feature changes

* Add TOSA operator compatibility via tosa-checker python package
  (MLIA-548/549/579)

### Interface changes

* Update CLI to allow the usage of the TOSA checker (MLIA-550).

### Issues fixed

* Fix the issue that no performance information is shown for TFLite files when
  the mode 'all_tests' is used (MLIA-552)
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
