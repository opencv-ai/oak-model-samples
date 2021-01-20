#!/bin/bash

test_status=0
uhubctl_location="2"
uhubctl_action="2"
models=($(ls -d */))

for package_folder in "${models[@]}"; do
    uhubctl -l $uhubctl_location -a $uhubctl_action
    cd ${package_folder} && python3.7 -m pip install --no-deps . -f https://artifacts.luxonis.com/artifactory/luxonis-python-snapshot-local/depthai/
    python3.7 -m pytest -s
    if [ $? -ne 0 ]; then
        echo -e "Failed test: " ${package_folder}
        test_status=1
    fi
    python3.7 -m pip uninstall -y ${package_folder}
    cd ${CI_PROJECT_DIR}
done

exit $test_status
