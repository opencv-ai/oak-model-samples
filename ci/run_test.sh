#!/bin/bash

retries=3

for ((iteration_amount = 1; iteration_amount <= ${retries}; iteration_amount++)); do
    python3.7 -m pytest -s -vv
    if [ $? == 0 ]; then
        exit 0
    fi
done
exit 1
