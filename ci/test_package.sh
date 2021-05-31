#!/bin/bash

models=($(ls -d */))
retries=3
test_status=0

for model_folder in "${models[@]}"; do
    if [[ -d "$model_folder" ]] &&  [[ $(basename ${model_folder}) != *"ci"* ]] && [[ $(basename ${model_folder}) != *"docs"* ]]; then
      python3 -m pip install pip --upgrade
      python3.7 -m pip install --no-cache-dir --no-cache -U  wheel_packages/${model_folder::-1}*.whl -f https://artifacts.luxonis.com/artifactory/luxonis-python-snapshot-local/depthai/
      cp -r oak_inference_utils ${model_folder}
      cd ${model_folder}
      for ((iteration_amount = 1; iteration_amount <= ${retries}; iteration_amount++)); do
          python3.7 -m pytest -s -vv
          if [ $? == 0 ]; then
              test_status=1
              echo -e "Failed test for: " ${model_folder::-1}
          fi
      done
      python3.7 -m pip uninstall -y ${model_folder::-1}
      cd ${CI_PROJECT_DIR}
    fi
done

cd ${CI_PROJECT_DIR}
exit $test_status