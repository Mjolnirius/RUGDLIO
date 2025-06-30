#!/bin/bash

# ROS2 setup
source /opt/ros/humble/setup.bash
source ~/dfliom_ws/install/setup.bash

# PYTHONPATH setzen
export PYTHONPATH=$HOME/miniconda3/envs/dfliom_env/lib/python3.10/site-packages:$PYTHONPATH
export PYTHONPATH=$HOME/dfliom_ws/src/FeatureLIOM:$PYTHONPATH

# FeatureLIOM-Node starten
ros2 run FeatureLIOM extract --ros-args -r /dliom/odom_node/compress:=/os_cloud_node/points
