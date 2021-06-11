#!/bin/bash

models=($(ls -d */))
retries=3
test_status=0

for model_folder in "${models[@]}"; do
    if [[ -d "$model_folder" ]] &&  [[ $(basename ${model_folder}) != @(ci|docs|oak_inference_utils|wheel_packages) ]]; then
      python3.7 -m pip install pip --upgrade
      python3.7 -m pip install --no-cache-dir --no-cache -U  wheel_packages/${model_folder::-1}*.whl
      if  [[ $LATEST_DEPTHAI == 'true' ]];
      then
        python3.7 -m pip install -U depthai
      fi
      cd ${model_folder}
      for ((iteration_amount = 1; iteration_amount <= ${retries}; iteration_amount++)); do
          python3.7 -m pytest -s -vv
          if [ $? -ne 0 ]; then
              test_status=1
              echo -e "Failed test for: " ${model_folder::-1}
          else
              break
          fi
      done
      python3.7 -m pip uninstall -y ${model_folder::-1}
      cd ${CI_PROJECT_DIR}
    fi
done

cd ${CI_PROJECT_DIR}
python3.7 -m pip cache purge
exit $test_status
