#!/bin/bash

models=($(ls -d */))

for model_folder in "${models[@]}"; do
    cd ${model_folder} && python3.7 -m pip install --no-deps . -f https://artifacts.luxonis.com/artifactory/luxonis-python-snapshot-local/depthai/
    cd ${CI_PROJECT_DIR}
done
