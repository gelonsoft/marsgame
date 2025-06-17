#!/bin/bash
#docker build -t terraforming-mars-bot:latest -f Dockerfile.train .
git reset
git pull
docker stop tms-bot-train || true
docker rm tms-bot-train || true
echo "SERVER_BASE_URL=${SERVER_BASE_URL}"
docker run --name tms-bot-train -v $(pwd):/data -e SERVER_BASE_URL="${SERVER_BASE_URL}" terraforming-mars-bot:latest  python3 ppo.py
