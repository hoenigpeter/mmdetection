#!/bin/bash

docker run \
--gpus all \
-it \
--shm-size=8gb --env="DISPLAY" \
--volume="/home/hoenig/temp/mmdetection:/mmdetection" \
--name=mmdetectionv0 mmdetection