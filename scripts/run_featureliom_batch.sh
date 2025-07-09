#!/bin/bash

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate dfliom_env

# ROS2 setup
source /opt/ros/humble/setup.bash
source ~/dfliom_ws/install/setup.bash

# PYTHONPATH setzen (falls nötig für ruamel, torch, etc.)
export PYTHONPATH=$HOME/miniconda3/envs/dfliom_env/lib/python3.10/site-packages:$PYTHONPATH
export PYTHONPATH=$HOME/dfliom_ws/src/FeatureLIOM:$PYTHONPATH


# Batch-Verarbeitung direkt starten (kein ROS nötig)
python3 $HOME/dfliom_ws/src/FeatureLIOM/keypoint_node/batch_runner.py \
  --input_dir $HOME/dfliom_ws/src/FeatureLIOM/input_data/dataset_quadhard_full/quadhard_seq_50_79 \
  --output_dir $HOME/dfliom_ws/src/FeatureLIOM/input_data/dataset_quadhard_full/DFLIOM_output_new_quadhard_seq_50_79 \
  --use_model \
  --mode bimodal \
  --lidar_range 70.0    # To be changed