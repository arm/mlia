# Running E2E tests

E2E tests could be launched locally using script `check-e2e.sh`
in the root of the project.

Next environment variables should be set before run:

* ARMLMD_LICENSE_FILE - license info

* MLIA_E2E_CONFIG - path to the E2E tests configuration directory

* AIET_ARTIFACT_PATH - path to the AIET artifact which will be used for
  AIET installation

Directory which MLIA_E2E_CONFIG points to should have two subfolders:

* systems - artifacts for the AIET systems should be placed here
* software - artifacts for the AIET software should be placed here

## Example layout for the E2E configuration directory

```
e2e_config/
    systems/
        fvp_corstone_sse-300_ethos-u55-21.08.0-SNAPSHOT.tar.gz
        sgm775_ethosu_platform-21.03.0.tar.gz
        sgm775_ethosu_platform-21.08.0-SNAPSHOT-oss.tar.gz
    software/
        ethosu_eval_platform_release_aiet-21.08.0-SNAPSHOT.tar.gz
        ethosU65_eval_app-21.08.0-SNAPSHOT.tar.gz
    aiet-21.9.0rc2-py3-none-any.whl
```

## Example E2E tests launch

```shell
export ARMLMD_LICENSE_FILE=<actual value for the license env variable>
export MLIA_E2E_CONFIG=e2e_config
export AIET_ARTIFACT_PATH=e2e_config/aiet-21.9.0rc2-py3-none-any.whl
./check-e2e.sh
```
