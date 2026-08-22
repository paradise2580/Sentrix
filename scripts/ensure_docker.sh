#!/bin/bash
# Restarts the Docker daemon if it isn't already running. Needed because
# this sandbox doesn't persist background processes across separate tool
# invocations — NOT something a real server/CI runner needs to worry about.
if ! docker ps > /dev/null 2>&1; then
    dockerd > /tmp/dockerd.log 2>&1 &
    for i in $(seq 1 15); do
        sleep 1
        docker ps > /dev/null 2>&1 && break
    done
fi
docker ps > /dev/null 2>&1 && echo "Docker is up" || echo "Docker FAILED to start"
