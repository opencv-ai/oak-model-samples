#!/bin/bash

models=($(ls -d */))

for model_folder in "${models[@]}"; do
    cd ${model_folder} && python3.7 setup.py build_ext bdist_wheel --dist-dir ${CI_PROJECT_DIR}/wheel_packages
    cd ${CI_PROJECT_DIR}
done
