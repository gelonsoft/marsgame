#!/bin/bash
docker stop tms-bot-train || true
docker rm tms-bot-train || true
docker build -t terraforming-mars-bot:latest -f Dockerfile.train .