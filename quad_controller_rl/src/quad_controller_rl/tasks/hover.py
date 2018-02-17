"""Takeoff task."""

import numpy as np
from gym import spaces
from quad_controller_rl.tasks.base_task import BaseTask
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench


class Hover(BaseTask):
    """Simple task where the goal is to lift off the ground and reach a target height."""

    def __init__(self):
        # super().__init__()
        BaseTask.__init__(self)
        self.name = 'hover'
        # env is cube_size x cube_size x cube_size
        cube_size = 300.0
        max_vel = 50
        # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w>
        self.observation_space = spaces.Box(
            np.array([-cube_size / 2, -cube_size / 2,       0.0,
                      # -1.0, -1.0, -1.0, -1.0,
                      -max_vel, -max_vel, -max_vel]),
            np.array([ cube_size / 2,  cube_size / 2, cube_size,
                       # 1.0,  1.0,  1.0,  1.0,
                       max_vel, max_vel, max_vel]))
        # print("Takeoff(): observation_space = {}".format(self.observation_space))

        max_force = 20.0
        max_force_z = 40.0
        max_torque = 1.0
        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force,  0.0, -max_torque, -max_torque, -max_torque]),
            np.array([ max_force,  max_force,  max_force_z,  max_torque,  max_torque,  max_torque]))
        # print("Takeoff(): action_space = {}".format(self.action_space))

        # Task-specific parameters
        self.max_duration = 5.0  # secs
        self.max_error_position = 20.0  # distance units
        self.max_velocity = 30.0
        self.target_z = 20.0
        self.target_position = np.array([0.0, 0.0, self.target_z])
        self.weight_position = 0.7
        # self.target_orientation = np.array([0.0, 0.0, 0.0, 1.0])  # target orientation quaternion (upright)
        # self.weight_orientation = 0.1
        self.target_velocity = np.array([0.0, 0.0, 0.0])  # target velocity (ideally should stay in place)
        self.weight_velocity = 0.3

        self.last_timestamp = None
        self.last_position = None
        self.timestep = 0

    def reset(self):
        # Reset episode-specific variables
        self.last_timestamp = None
        self.last_position = None
        self.timestep = 0
        # slight random position around the target
        p = self.target_position + np.random.normal(0.5, 0.1, size=3)
        # drop off from the target height
        return Pose(
            position=Point(*p),
            orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
        ), Twist(
            linear=Vector3(0.0, 0.0, 0.0),
            angular=Vector3(0.0, 0.0, 0.0)
        )

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):

        # Prepare state vector (pose, orientation, velocity only; ignore angular_velocity, linear_acceleration)
        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        # orientation = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        if self.last_timestamp is None:
            velocity = np.array([0.0, 0.0, 0.0])
        else:  # prevent divide by zero
            velocity = (position - self.last_position) / max(timestamp - self.last_timestamp, 1e-03)
        # state = np.concatenate([position, orientation, velocity])  # combined state vector
        state = np.concatenate([position, velocity])  # combined state vector
        self.last_timestamp = timestamp
        self.last_position = position

        # Euclidean distance from target position vector
        error_position = np.linalg.norm(self.target_position - position)
        # Euclidean distance from target orientation quaternion (a better comparison may be needed)
        # error_orientation = np.linalg.norm(self.target_orientation - orientation)
        # Euclidean distance from target velocity vector
        error_velocity = np.linalg.norm(self.target_velocity - velocity)

        # agent is hovering about the target.
        # reward = zero for matching target z, -ve as you go farther, upto -10
        # reward = -min(abs(self.target_z - pose.position.z), 10.0)
        done = False

        reward = -(self.weight_position * error_position +
                   # self.weight_orientation * error_orientation +
                   self.weight_velocity * error_velocity)

        # distance penalty
        # reward = -1.0 * error_position
        # squared distance penalty
        # reward = -1.0 * error_position**2
        # gravity wells
        # inverse law.
        # reward = 20.0 / (-1.0 * reward + 1)
        # inverse square law.
        # agent is closer than default starting distance. give it some reward.
        # inverse_square_reward = 250.0 / (error_position + 4) ** 2
        # attempt a smooth transition on the boundary between height and radial rewards.
        # blend_mask = (10 - error_position) / 10.0
        # reward += inverse_square_reward * blend_mask

        if error_position > self.max_error_position:
            dt = self.last_timestamp / self.timestep
            total_steps = self.max_duration / dt
            remaining_steps = total_steps - self.timestep
            reward -= remaining_steps * 10.0  # extra penalty, agent strayed too far
            # reward += remaining_steps * 2.0  # extra penalty, agent strayed too far
            done = True
        elif timestamp > self.max_duration:
            reward += 50.0  # extra reward, agent made it to the end
            done = True

        # print(timestamp, reward)
        # Take one RL step, passing in current state and reward, and obtain action
        # Note: The reward passed in here is the result of past action(s)
        # Note: action = <force; torque> vector
        action_new = self.agent.step(state, reward, done)
        # print(action)
        # if done:
        #     self.i_episode += 1
        # Convert to proper force command (a Wrench object) and return it
        self.timestep += 1
        if action_new is not None:
            # action_new = action * self.action_space.high
            # flatten, clamp to action space limits
            # action_new = np.clip(action_new0.flatten(), self.action_space.low, self.action_space.high)
            # if np.any(action_new != action_new0):
            #     print("  action clip : " + (3 * '{:>7.3f} ').format(*action[0:3]))
            #     print("  action_new0 : " + (3 * '{:>7.3f} ').format(*action_new0[0:3]))
            #     print("  action_new  : " + (3 * '{:>7.3f} ').format(*action_new[0:3]))
            return Wrench(force=Vector3(action_new[0], action_new[1], action_new[2]),
                          torque=Vector3(action_new[3], action_new[4], action_new[5])), done
        else:
            print('Wrench()', done)
            return Wrench(), done
