# LiDAR Inertial Odometry and Mapping Using Learned Registration-Relevant Features

## Update

* Our paper has been accepted to IEEE ICRA 2025! Looking forward to meeting fellow researchers in Atlanta.
* Code for feature extraction (Python ROS node) has been released!
* Code for DLIOM (C++) will be released separately.

## About

This repository contains the official implementation of the feature extractor network and ros2 node proposed in paper "LiDAR Inertial Odometry and Mapping Using Learned Registration-Relevant Features". It achieves robust and efficient real-time LiDAR Inertial Odometry using a light-weight neural network based feature extractor, as opposed to previous feature-based methods that relies on hand-crafted heuristics and parameters. More detailed maps are shown in the last section.

<br>
<p align='center'>
    <img src="./images/map_neu.png" alt="NEU Campus" width="360" height="230"/>
    <img src="./images/map.png" alt="Newer College Short" width="300"  height="230"/>
</p>

## Dependencies

* ROS2 Humble
* Ubuntu 22.04
* Core python packages and version can be found in `requirements.txt`.
* Install `octree_handler`: ```cd submodules/octree_handler && pip3 install -U .```

## Running

### Setup Machine
```
mkdir -p ~/dfliom_ws/src
cd ~/dfliom_ws/src

conda create -n dfliom_env python=3.10 -y
conda activate dfliom_env

sudo apt update && sudo apt upgrade -y
sudo apt install software-properties-common curl gnupg lsb-release -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | \
  sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install ros-humble-desktop -y

conda deactivate 
sudo apt install python3-rosdep -y

sudo rosdep init
rosdep update

conda activate dfliom_env

cd ~/dfliom_ws/src
git clone https://github.com/neu-autonomy/FeatureLIOM.git #bzw. mein git

cd ~/dfliom_ws/src/FeatureLIOM
git submodule update --init --recursive

cd submodules/octree_handler
pip install -U .
cd ../../

pip uninstall ruamel.yaml -y
pip install --no-cache-dir ruamel.yaml

pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118

#install special wheels for cu118 (ToDo)

pip install -r requirements.txt

# .sh file ansonsten benutzen # --------------------------------------------
cd ~/dfliom_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select FeatureLIOM
source install/setup.bash

python -c "import site; print(site.getsitepackages()[0])"
export PYTHONPATH=/home/thor_unix_2204/miniconda3/envs/dfliom_env/lib/python3.10/site-packages:$PYTHONPATH
export PYTHONPATH=$HOME/dfliom_ws/src/FeatureLIOM:$PYTHONPATH

ros2 run FeatureLIOM extract --ros-args -r /dliom/odom_node/compress:=/os_cloud_node/points
```

### Building
Build the ROS package using `Colcon`:

```
colcon build --packages-select FeatureLIOM
source install/setup.bash
ros2 run FeatureLIOM extract
```
The node listens to the specified `pcd_topic`, and publishes keypoint **indices** on `downsampled_topic`. This implementation assumes the odometry code maintains the point ordering until the dense point cloud gets compressed.

Quickstart: Use the following shell script:

```
cd ~/dfliom_ws
conda activate dfliom_env
~/dfliom_ws/src/FeatureLIOM/scripts/start_featureliom.sh
```
```
rviz2
# set frame id to: ToDo (odom, os_sensor, map) 
# listen to /PointRec/dropped_cloud_dfliom & /PointRec/compressed_cloud_dfliom
```
```
ros2 bag play ~/dfliom_ws/src/FeatureLIOM/input_data/dataset_quadhard_full/quadhard.db3  --loop   --rate 0.5
```
For the local (non-ROS) variant:
```
cd ~/dfliom_ws
conda activate dfliom_env
~/dfliom_ws/src/FeatureLIOM/scripts/run_featureliom_batch.sh
```


## Generated Maps

We present detailed maps of the Northeastern University Campus and Newer College Dataset in this section.

### Northeastern University Main Campus (727.50m)

<p align='center'>
    <img src="./images/main_campus/main_campus_map.png" alt="NEU Campus" width="720"/>
</p>

<p align='center'>
    <img src="./images/main_campus/main_campus_zoom_law_school.png" alt="Law School_LED" width="360"/>
    <img src="./images/main_campus/main_campus_zoom_Shillman_Hall.png" alt="Shillman Hall" width="360"/>
</p>

<p align='center'>
    <img src="./images/main_campus/main_campus_zoom_BofA.png" alt="BofA and Forsyth" width="360"/>
    <img src="./images/main_campus/main_campus_zoom_Forsyth_Sidewalk.png" alt="Forsyth Sidewalk" width="360"/>
</p>

<p align='center'>
    <img src="./images/main_campus/main_campus_zoom_Egan.png" alt="Egan Research Center" width="360"/>
    <img src="./images/main_campus/main_campus_zoom_Egan_Sidewalk.png" alt="Egan Sidewalk" width="360"/>
</p>

<p align='center'>
    <img src="./images/main_campus/main_campus_zoom_Snell.png" alt="Snell Library" width="360"/>
    <img src="./images/main_campus/main_campus_zoom_Willis.png" alt="Willis Hall" width="360"/>
</p>

<p align='center'>
    <img src="./images/main_campus/main_campus_zoom_near_Snell.png" alt="Near Snell Library" width="360"/>
    <img src="./images/main_campus/main_campus_zoom_Willis_car.png" alt="Willis Hall Car" width="360"/>
</p>

<p align='center'>
    <img src="./images/main_campus/main_campus_zoom_painting.png" alt="Painting on Meserve Hall" width="360"/>
    <img src="./images/main_campus/main_campus_zoom_ground_pattern.png" alt="Pattern on ground" width="360"/>
</p>

### Northeastern University ISEC and Columbus Garage (548.32m)

<p align='center'>
    <img src="./images/exp/exp_map.png" alt="NEU ISEC" width="720"/>
</p>

<p align='center'>
    <img src="./images/exp/exp_zoom_bikes.png" width="360"/>
    <img src="./images/exp/exp_zoom_sign_and_pedestrian.png" width="360"/>
</p>

<p align='center'>
    <img src="./images/exp/exp_zoom_parked_cars.png" width="360"/>
    <img src="./images/exp/exp_zoom_columbus_garage.png" width="360"/>
</p>

<p align='center'>
    <img src="./images/exp/exp_zoom_squashbuster.png" width="360"/>
    <img src="./images/exp/exp_zoom_soccer.png" width="360"/>
</p>

<p align='center'>
    <img src="./images/exp/exp_zoom_columbus_back_stairs_and_car.png" width="360"/>
    <img src="./images/exp/exp_zoom_columbus_back.png" width="360"/>
</p>

<p align='center'>
    <img src="./images/exp/exp_zoom_station.png" width="360"/>
    <img src="./images/exp/exp_zoom_bt_isec_and_garage.png" width="360"/>
</p>

<p align='center'>
    <img src="./images/exp/exp_zoom_isec.png" width="360"/>
    <img src="./images/exp/exp_zoom_dorm.png" width="360"/>
</p>

### Northeastern University ISEC Bridge

<p align='center'>
    <img src="./images/bridge/bridge_map.png" width="720"/>
</p>

<p align='center'>
    <img src="./images/bridge/bridge_zoom_exp_ramp.png" width="360"/>
    <img src="./images/bridge/bridge_zoom_exp_door.png" width="360"/>
</p>

<p align='center'>
    <img src="./images/bridge/bridge_zoom_left.png" width="360"/>
    <img src="./images/bridge/bridge_zoom_right.png" width="360"/>
</p>

### Newer College Dataset

| <img src="./images/newer_college/short.png" width="360"/> | <img src="./images/newer_college/long.png" width="360"/> |
| ----- | ----- |
| *Newer College Short* | *Newer College Long* |

| <img src="./images/newer_college/mount.png" width="360"/> | <img src="./images/newer_college/park.png" width="360"/> |
| ----- | ----- |
| *Newer College Mount* | *Newer College Park* |

| <img src="./images/newer_college/quad_w_dyn.png" width="360"/> | <img src="./images/newer_college/quad_hard.png" width="360" height="237"/> |
| ----- | ----- |
| *Newer College Quad with Dynamics* | *Newer College Quad Hard* |

| <img src="./images/newer_college/quad_medium.png" width="360"/> | <img src="./images/newer_college/quad_easy.png" width="360" height="237"/> |
| ----- | ----- |
| *Newer College Quad Medium* | *Newer College Quad Easy* |

| <img src="./images/newer_college/math_easy.png" width="360"/> | <img src="./images/newer_college/math_medium.png" width="360" height="237"/> |
| ----- | ----- |
| *Newer College Math Easy* | *Newer College Math Medium* |

| <img src="./images/newer_college/math_hard.png" width="360"/> | <img src="./images/newer_college/cloister.png" width="360" height="237"/> |
| ----- | ----- |
| *Newer College Math Hard* | *Newer College Cloister* |

## Acknowledgement

We would also like to thank Alexander Estornell, Sahasrajit Anantharamakrishnan, and Yash Mewada for setting up the scout robot hardware, and Hanna Zhang, Yanlong Ma, Kenny Chen, and Nakul Joshi for help with data collection and comments.
