#!/bin/bash
#docker build -t terraforming-mars-bot:latest -f Dockerfile.train .
docker stop tms-bot-train || true
docker run --name tms-bot-train -v $(pwd):/data terraforming-mars-bot:latest python3 ppo.py