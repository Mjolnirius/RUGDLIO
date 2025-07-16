#!/bin/bash

# Conda aktiv lassen
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dfliom_env

# ROS2 ohne Conda-Störung sourcen
CONDA_LD_PATH="$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v 'conda' | tr '\n' ':')

source /opt/ros/humble/setup.bash
source ~/dfliom_ws/install/setup.bash

export LD_LIBRARY_PATH="$CONDA_LD_PATH"

# Pythonpath falls nötig
export PYTHONPATH=$HOME/miniconda3/envs/dfliom_env/lib/python3.10/site-packages:$PYTHONPATH
export PYTHONPATH=$HOME/dfliom_ws/src/FeatureLIOM:$PYTHONPATH

# Argumente setzen
INPUT_BAG="/home/thor_unix_2204/dfliom_ws/src/FeatureLIOM/input_data/dataset_quadhard_full/quadhard.db3"
OUTPUT_BAG="/home/thor_unix_2204/dfliom_ws/src/FeatureLIOM/input_data/dataset_quadhard_full/quadhard_dfliom.db3"
TOPIC="/os_cloud_node/points"
START_IDX=-1
END_IDX=-1

# 🛡️ Zwinge Systembibliothek statt Conda
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# Python-Skript starten
python3 $HOME/dfliom_ws/src/FeatureLIOM/keypoint_node/run_featureliom_on_rosbag.py \
  "$INPUT_BAG" \
  "$OUTPUT_BAG" \
  "$TOPIC" \
  "$START_IDX" \
  "$END_IDX"
