#!/bin/bash
#docker build -t terraforming-mars-bot:latest -f Dockerfile.train .
git reset
git pull
docker stop tms-bot-train || true
docker rm tms-bot-train || true
echo "SERVER_BASE_URL=${SERVER_BASE_URL}"
echo "START_LR=${START_LR}"
echo "CONTINUE_TRAIN=${CONTINUE_TRAIN}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "RUN_NAME=${RUN_NAME}"
if [ -z "${CONTINUE_TRAIN}" ]; then
    CONTINUE_TRAIN=""
if [ -z "${START_LR}" ]; then
    START_LR=""
if [ -z "${MODEL_PATH}" ]; then
    MODEL_PATH=""
if [ -z "${RUN_NAME}" ]; then
    RUN_NAME=""

docker run --name tms-bot-train -v $(pwd):/data -e SERVER_BASE_URL="${SERVER_BASE_URL}" \
  -e START_LR="${START_LR}" -e CONTINUE_TRAIN="${CONTINUE_TRAIN}" -e MODEL_PATH="${MODEL_PATH}" -e RUN_NAME="${RUN_NAME}" \
   terraforming-mars-bot:latest  python3 ppo.py
