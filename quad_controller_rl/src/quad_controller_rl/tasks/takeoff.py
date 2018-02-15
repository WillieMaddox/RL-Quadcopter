"""Takeoff task."""

import numpy as np
from gym import spaces
from quad_controller_rl.tasks.base_task import BaseTask
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench


class Takeoff(BaseTask):
    """Simple task where the goal is to lift off the ground and reach a target height."""

    def __init__(self):
        # super().__init__()
        BaseTask.__init__(self)

        # env is cube_size x cube_size x cube_size
        cube_size = 300.0
        # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w>
        self.observation_space = spaces.Box(
            np.array([-cube_size / 2, -cube_size / 2,       0.0, -1.0, -1.0, -1.0, -1.0]),
            np.array([ cube_size / 2,  cube_size / 2, cube_size,  1.0,  1.0,  1.0,  1.0]))
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
        self.max_force = max_force_z
        # target height (z position) to reach for successful takeoff
        self.target_z = 10.0
        self.target_position = np.array([0, 0, self.target_z])
        # self.target_x = 0.0
        # self.target_y = 0.0
        # self.i_episode = -10  # do initial rollout to fill up some experience.
        # self.param_noise_adaption_interval = 10
        # self.timestep = 0

    def reset(self):

        # We are doing this here so the timestamp doesn't muck things up.
        # if self.i_episode >= 0 and self.i_episode % 15 == 0:
        #     print('DDPG.learn()', self.i_episode, self.agent.i_episode)
        #     for i in range(50):
        #         if i % self.param_noise_adaption_interval == 0:
        #             distance = self.agent.adapt_param_noise()
        #         loss = self.agent.learn()
        #         if i % self.param_noise_adaption_interval == 0:
        #             fmt = '{}: loss = {}, distance = {}, stdev = {}'
        #             print(fmt.format(i, loss, distance, self.agent.param_noise.current_stddev))

            # print('loss =', loss)

        # self.timestep = 0

        # Nothing to reset; just return initial condition
        # drop off from a slight random height
        return Pose(
            position=Point(0.0, 0.0, np.random.normal(1.0, 0.1)),
            orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
        ), Twist(
            linear=Vector3(0.0, 0.0, 0.0),
            angular=Vector3(0.0, 0.0, 0.0)
        )

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        # Prepare state vector (pose only; ignore angular_velocity, linear_acceleration)
        state = np.array([
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w
        ])
        # fmt = '{:3} vel: ({:>7.4f}, {:>7.4f}, {:>7.4f})'
        # print(fmt.format(self.timestep, angular_velocity.x, angular_velocity.y, angular_velocity.z))
        # fmt = '{:3} acc: ({:>7.4f}, {:>7.4f}, {:>7.4f})'
        # print(fmt.format(self.timestep, linear_acceleration.x, linear_acceleration.y, linear_acceleration.z))

        # Compute reward / penalty and check if this episode is complete
        distance_to_target = np.linalg.norm(self.target_position - state[:3])

        # agent hits target perfectly.
        if distance_to_target < 1:
            reward = 200  # bonus reward
            done = True
        # agent has crossed the target height
        elif pose.position.z >= self.target_z:
            reward = 100  # bonus reward
            done = True
        # agent has run out of time
        elif timestamp > self.max_duration:
            reward = -100.0  # extra penalty
            done = True
        else:
            # reward = zero for matching target z, -ve as you go farther, upto -10
            reward = -min(abs(self.target_z - pose.position.z), 10.0)
            done = False
            # agent is closer than default starting distance. give it some reward.
            if distance_to_target < 10:
                inverse_square_reward = 250.0 / (distance_to_target + 4) ** 2
                # distance penalty
                # reward = -1.0 * distance_to_target
                # squared distance penalty
                # reward = -1.0 * distance_to_target**2
                # gravity wells
                # inverse law.
                # reward = 30.0 / (distance_to_target + 2)
                # inverse square law.
                # reward = 250.0 / (distance_to_target + 4) ** 2
                # attempt a smooth transition on the boundary between height and radial rewards.
                blend_mask = (10 - distance_to_target) / 10.0

                reward += inverse_square_reward * blend_mask

        # Take one RL step, passing in current state and reward, and obtain action
        # Note: The reward passed in here is the result of past action(s)
        # Note: action = <force; torque> vector
        action_new = self.agent.step(state, reward, done)
        # print(action)
        # if done:
        #     self.i_episode += 1
        # Convert to proper force command (a Wrench object) and return it
        # self.timestep += 1
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
