#!/bin/bash

docker run \
--gpus all \
-it \
--shm-size=8gb --env="DISPLAY" \
--volume="/home/hoenig/temp/mmdetection:/mmdetection" \
--volume="/hdd2/mmdetection:/mmdetection/work_dirs_hdd" \
--volume="/ssd3/datasets_bop:/mmdetection/data" \
--name=mmdetectionv0 mmdetection