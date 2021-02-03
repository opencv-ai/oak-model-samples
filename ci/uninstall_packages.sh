#!/bin/bash

models=($(ls -d */))

for model_folder in "${models[@]}"; do
    cd ${model_folder} && python3.7 -m pip uninstall -y ${model_folder::-1}
    cd ${CI_PROJECT_DIR}
done
