#!/bin/bash

# Conda-Umgebung aktivieren
#source ~/miniconda3/etc/profile.d/conda.sh
#conda activate dfliom_env


# ROS2 sourcen
source /opt/ros/humble/setup.bash
source ~/dfliom_ws/install/setup.bash

# LD_LIBRARY Workaround
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# PYTHONPATH setzen
export PYTHONPATH=$HOME/miniconda3/envs/dfliom_env/lib/python3.10/site-packages:$PYTHONPATH
export PYTHONPATH=$HOME/dfliom_ws/src/FeatureLIOM:$PYTHONPATH

# Argumente definieren
INPUT_BAG="/home/thor_unix_2204/dfliom_ws/src/FeatureLIOM/input_data/dataset_quadhard_full/quadhard.db3"
OUTPUT_BAG_BASE="/home/thor_unix_2204/dfliom_ws/src/FeatureLIOM/input_data/dataset_quadhard_full/quadhard_dfliom.db3"
TOPIC="/os_cloud_node/points"
#START_IDX=5
#END_IDX=10

# Versionierung falls Datei existiert
OUTPUT_BAG="$OUTPUT_BAG_BASE"
i=1
while [[ -e $OUTPUT_BAG ]]; do
  OUTPUT_BAG="${OUTPUT_BAG_BASE%.db3}_V${i}.db3"
  ((i++))
done

# Sicherstellen, dass Zielordner existiert
mkdir -p "$(dirname "$OUTPUT_BAG")"

# Start-Info ausgeben
echo "Starte dfliom_rosbag.py mit:"
echo "  Eingabe: $INPUT_BAG"
echo "  Ausgabe: $OUTPUT_BAG"
echo "  Topic  : $TOPIC"
#echo "  Bereich: $START_IDX bis $END_IDX"

# Python-Skript starten
python3 $HOME/dfliom_ws/src/FeatureLIOM/keypoint_node/dfliom_rosbag.py \
  "$INPUT_BAG" \
  "$OUTPUT_BAG" \
  "$TOPIC"
#  "$START_IDX" \
#  "$END_IDX"
