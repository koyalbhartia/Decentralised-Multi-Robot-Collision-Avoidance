# Decentralised-Multi-Robot-Collision-Avoidance via Deep Reinforecment Learning

## Dependencies

- python2.7
- [ROS Kinetic](http://wiki.ros.org/kinetic)
- [mpi4py](https://mpi4py.readthedocs.io/en/stable/)
- [Stage](http://rtv.github.io/Stage/)
- [PyTorch](http://pytorch.org/)

## Train
`stage_ros-add_pose_and_crash` package is used instead of the default package provided by ROS.
```
mkdir -p catkin_ws/src
cp stage_ros-add_pose_and_crash catkin_ws/src
cd catkin_ws
catkin_make
source devel/setup.bash
```
### To train on small environment, run the following command:
To train on obstacleless space
```
rosrun stage_ros_add_pose_and_crash stageros worlds/scene1_stage1.world
mpiexec -np 44 python ppo_update1.py
```
```
rosrun stage_ros_add_pose_and_crash stageros worlds/scene1_stage2.world
mpiexec -np 44 python ppo_update1.py
```
### To train on large environment, run the following command:
```
rosrun stage_ros_add_pose_and_crash stageros worlds/warehouse.world
mpiexec -np 44 python ppo_update2.py
```
## Test
```
rosrun stage_ros_add_pose_and_crash stageros worlds/warehouse.world
mpiexec -np 44 python warehouse_test.py
```
```
rosrun stage_ros_add_pose_and_crash stageros worlds/warehouse.world
mpiexec -np 50 python warehouse_test.py
```
