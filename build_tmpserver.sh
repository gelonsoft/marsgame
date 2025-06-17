#!/bin/bash
docker stop tms || true
docker rm tms || true
docker build -t terraforming-mars-server:latest -f Dockerfile.tmserver . &&  docker run -d -p 9976:9976 --name tms terraforming-mars-server:latest
# cleanup_tmserver.sh