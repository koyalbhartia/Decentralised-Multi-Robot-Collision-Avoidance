# Decentralised-Multi-Robot-Collision-Avoidance via Deep Reinforecment Learning

## Dependencies

- python2.7
- [ROS Kinetic](http://wiki.ros.org/kinetic)
- [mpi4py](https://mpi4py.readthedocs.io/en/stable/)
- [Stage](http://rtv.github.io/Stage/)
- [PyTorch](http://pytorch.org/)

## File List
- worlds: Folder containing the different Stage world files with the initial poisitons of the robots.
- policy: Folder is where the trained policies get saved
- model/net.py: Python file with the CNN Architecture
- model/ppo.py : Python file updatig policy and value loss
- videos: Folder containing some videos displaying training/testing
- graphs: Folder contains graphs generated during training
- log: Folder containing sample of log files generated during training which are used to plot graphs
- ppo_scene/ppo_warehouse.py : Python file where the on-policy (PPO) training process of the robots starts.
- scene_world.py / warehouse_world.py : Python file which takes care of the RL setup described in the report. Some of these tasks include fetching observations, publishing command velocities, reward designing etc.
- warehouse_test/scene_test.py : Python file which uses the trained policy to test on the given environment.

## Train
`stage_ros-add_pose_and_crash` package is used instead of the default package provided by ROS.
```
mkdir -p catkin_ws/src
git clone https://github.com/koyalbhartia/Decentralised-Multi-Robot-Collision-Avoidance.git
cp stage_ros-add_pose_and_crash catkin_ws/src
cd catkin_ws
catkin_make
source devel/setup.bash
```
### To train
Large environemnt
```
rosrun stage_ros_add_pose_and_crash stageros worlds/warehouse.world
mpiexec -np 40 python ppo_warehouse.py
```
Small Environment
```
rosrun stage_ros_add_pose_and_crash stageros worlds/scene1.world
mpiexec -np 44 python ppo_scene.py
```
### To train on large environment, run the following command:

## Test
```
rosrun stage_ros_add_pose_and_crash stageros worlds/warehouse.world
mpiexec -np 40 python warehouse_test.py
```
```
rosrun stage_ros_add_pose_and_crash stageros worlds/scene1.world
mpiexec -np 44 python scene_test.py
```
