#!/bin/bash
set -o allexport
source .env
set +o allexport

python ./src/main.py