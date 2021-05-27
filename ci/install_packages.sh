#!/bin/bash

models=($(ls -d */))

for model_folder in "${models[@]}"; do
    python3.7 -m pip install --no-cache-dir --no-cache -U ${1} wheel_packages/${model_folder::-1}*.whl -f https://artifacts.luxonis.com/artifactory/luxonis-python-snapshot-local/depthai/
    cp -r oak_inference_utils ${model_folder}
done
