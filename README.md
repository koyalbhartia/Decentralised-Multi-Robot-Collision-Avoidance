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
- ppo1/ppo2.py : Python file where the on-policy (PPO) training process of the robots starts.
- scene1_world.py / warehouse_world.py : Python file which takes care of the RL setup described in the report. Some of these tasks include fetching observations, publishing command velocities, reward designing etc.
- warehouse_test/scene1_test.py : Python file which uses the trained policy to test on the given environment.

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
### To train on small environment, run the following command:
To train on obstacleless space
```
rosrun stage_ros_add_pose_and_crash stageros worlds/scene1_stage1.world
mpiexec -np 44 python ppo1.py
```
```
rosrun stage_ros_add_pose_and_crash stageros worlds/scene1_stage2.world
mpiexec -np 44 python ppo1.py
```
### To train on large environment, run the following command:
```
rosrun stage_ros_add_pose_and_crash stageros worlds/warehouse.world
mpiexec -np 44 python ppo2.py
```
## Test
```
rosrun stage_ros_add_pose_and_crash stageros worlds/warehouse.world
mpiexec -np 45 python warehouse_test.py
```
```
rosrun stage_ros_add_pose_and_crash stageros worlds/scene1_stage2.world
mpiexec -np 44 python scene1_test.py
```
