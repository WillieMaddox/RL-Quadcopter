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
        self.mass = 2.0  # This is the mass of the Quadcopter. (kg)
        self.gravity = -9.81  # Gravity on earth. (m/s^2)
        # env is cube_size x cube_size x cube_size
        cube_size = 300.0
        # max_vel = 30
        max_acc = 5
        # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w>
        self.observation_space = spaces.Box(
            np.array([
                -cube_size/2, -cube_size/2, 0.0,
                # -1.0, -1.0, -1.0, -1.0,
                # -max_vel, -max_vel, -max_vel,
                -max_acc, -max_acc, -max_acc
            ]),
            np.array([
                cube_size/2,  cube_size/2, cube_size,
                # 1.0,  1.0,  1.0,  1.0,
                # max_vel, max_vel, max_vel,
                max_acc, max_acc, max_acc
            ]))

        max_force = 10.0
        min_force_z = 10.0
        max_force_z = 30.0
        max_torque = 1.0
        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        self.action_space = spaces.Box(
            np.array([
                -max_force, -max_force,  min_force_z,
                -max_torque, -max_torque, -max_torque]),
            np.array([
                max_force,  max_force,  max_force_z,
                max_torque,  max_torque,  max_torque]))

        # Task-specific parameters
        self.target_z = 20.0
        self.max_duration = 5.0  # secs
        self.max_distance = self.target_z  # distance units
        self.max_speed = 10.0  # should be 30 but it never has enough time to go that fast
        self.max_accel = 5.0
        self.max_reward = 1
        self.target_position = np.array([0.0, 0.0, self.target_z])
        self.weight_position = 0.7
        # self.target_velocity = np.array([0.0, 0.0, 0.0])  # target velocity (ideally should stay in place)
        self.weight_velocity = 0.0
        # self.target_acceleration = np.array([0.0, 0.0, 0.0])
        self.weight_acceleration = 0.25
        self.prev_timestamp = None
        self.prev_position = None
        # self.prev_position_new = None
        # self.prev_velocity = None
        # self.prev_velocity_new = np.array([0.0, 0.0, 0.0])
        # self.prev_action = None
        self.timestep = 0
        self.perturb_position = None

    def reset(self):
        # Reset episode-specific variables
        self.prev_timestamp = None
        self.prev_position = None
        # self.prev_position_new = None
        # self.prev_velocity = None
        # self.prev_velocity_new = np.array([0.0, 0.0, 0.0])
        # self.prev_action = None
        self.timestep = 0
        # slight random position around the target
        self.perturb_position = np.random.normal(0.5, 0.5, size=3)
        p = self.target_position + self.perturb_position
        # drop off from the target height
        return Pose(
            position=Point(*p),
            orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
        ), Twist(
            linear=Vector3(0.0, 0.0, 0.0),
            angular=Vector3(0.0, 0.0, 0.0)
        )

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):

        # Prepare state vector (pose, velocity, linear_acceleration only; ignore angular_velocity)
        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        # orientation = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        if self.prev_timestamp is None:
            velocity = np.array([0.0, 0.0, 0.0])
            # velocity = -self.perturb_position
        else:
            dt = max(timestamp - self.prev_timestamp, 1e-03)
            dx = position - self.prev_position
            velocity = dx / dt

        # if self.prev_timestamp is None:
        #     acceleration = np.array([0.0, 0.0, 0.0])
        # else:
        #     acceleration = (velocity - self.prev_velocity) / max(timestamp - self.prev_timestamp, 1e-03)

        acceleration = np.array([linear_acceleration.x, linear_acceleration.y, linear_acceleration.z])

        # if self.prev_timestamp < 0.1:
        #     # velocity_new = np.array([0.0, 0.0, 0.0])
        #     velocity_new = (position - self.prev_position) / max(timestamp - self.prev_timestamp, 1e-03)
        # else:
        #     velocity_new = acceleration_new * (timestamp - self.prev_timestamp) + self.prev_velocity_new

        # if self.prev_timestamp < 0.1:
        #     position_new = position
        # else:
        #     position_new = velocity_new * (timestamp - self.prev_timestamp) + self.prev_position_new

        # if self.prev_action is None:
        #     acceleration_target = acceleration_new
        # else:
        #     acceleration_target = self.prev_action[:3] / self.mass + np.array([0.0, 0.0, self.gravity])

        state = np.concatenate([position, acceleration])
        # state = np.concatenate([position_new, velocity_new, acceleration_new])  # combined state vector

        position_vector = position - self.target_position
        distance = np.linalg.norm(position_vector)
        p_hat = position_vector / distance
        distance_scaled = distance / self.max_distance
        position_reward_scaled = 2 * max(1 - distance_scaled, 0) - 1
        position_reward = self.max_reward * position_reward_scaled

        speed = np.linalg.norm(velocity)
        v_hat = velocity / speed
        velocity_direction_scaled = np.dot(-p_hat, v_hat)  # reward if v is pointing toward target
        speed_scaled = speed / self.max_speed
        velocity_reward_scaled = velocity_direction_scaled * (1 - abs(speed_scaled - distance_scaled))
        if self.prev_timestamp is None:
            velocity_reward_scaled = 0
        velocity_reward = self.max_reward * velocity_reward_scaled

        # self.prev_position_new = position_new
        # self.prev_velocity_new = velocity_new
        # position_reward_new = np.linalg.norm(position_new - self.target_position)
        # velocity_reward_new = np.linalg.norm(velocity_new - self.target_velocity)

        accel = np.linalg.norm(acceleration)
        a_hat = acceleration / accel
        accel_direction_scaled = np.dot(-p_hat, a_hat)  # reward if v is pointing toward target
        accel_scaled = accel / self.max_accel
        accel_reward_scaled = accel_direction_scaled * (1 - abs(accel_scaled - distance_scaled))
        if self.prev_timestamp is None:
            accel_reward_scaled = 0
        accel_reward = self.max_reward * accel_reward_scaled

        # acceleration_reward_target = np.linalg.norm(acceleration_target - self.target_acceleration)
        # acceleration_reward_new = np.linalg.norm(acceleration_new - self.target_acceleration)

        self.prev_timestamp = timestamp
        self.prev_position = position
        # self.prev_velocity = velocity

        done = False

        # reward = zero for matching target z, -ve as you go farther, upto -10
        # reward = -min(abs(self.target_z - pose.position.z), 10.0)
        weighted_position_reward = self.weight_position * position_reward
        weighted_acceleration_reward = self.weight_acceleration * accel_reward
        reward = weighted_position_reward + weighted_acceleration_reward

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

        # if reward < -self.max_reward:
        if distance > self.max_distance:
            # dt = self.prev_timestamp / self.timestep
            # # print(dt)
            # total_steps = self.max_duration / dt
            # # print(total_steps)
            # remaining_steps = total_steps - self.timestep
            # # print(remaining_steps)
            # reward -= remaining_steps * self.max_reward  # extra penalty, agent strayed too far
            reward -= 2.0 * self.max_reward  # extra penalty, agent strayed too far
            done = True
        elif np.max(np.abs(acceleration)) > self.max_accel:
            reward -= 2.0 * self.max_reward  # extra penalty, agent strayed too far
            done = True
        if timestamp > self.max_duration:
            reward += 2.0 * self.max_reward  # extra reward, agent made it to the end
            done = True

        # Take one RL step, passing in current state and reward, and obtain action
        # Note: The reward passed in here is the result of past action(s)
        action_new = self.agent.step(state, reward, done)

        # if not done or self.timestep != 1:
        #
        # else:
        #     action_new = None
        # self.prev_action = action_new

        # fmt = [
        #     "[{:3}",
        #     "|{:5.3f}",
        #     "||{:7.3f},{:7.3f},{:7.3f}",
        #     "|{:6.3f}",
        #     "|{:6.3f}",
        #     "|{:6.3f}",
        #     "|{:6.3f}",
        #     "||{:7.3f},{:7.3f},{:7.3f}",
        #     "|{:6.3f}",
        #     "|{:6.3f}",
        #     "|{:6.3f}",
        #     "|{:6.3f}",
        #     "||{:7.3f},{:7.3f},{:7.3f}",
        #     "|{:6.3f}",
        #     "|{:6.3f}",
        #     "|{:6.3f}",
        #     "|{:6.3f}",
        #     "||{:7.3f}",
        #     "|{:7.3f},{:7.3f},{:7.3f}],"
        # ]
        # print(''.join(fmt).format(
        #     self.timestep,
        #     timestamp,
        #     *position,
        #     distance,
        #     distance_scaled,
        #     position_reward_scaled,
        #     position_reward,
        #     *velocity,
        #     speed,
        #     speed_scaled,
        #     velocity_reward_scaled,
        #     velocity_reward,
        #     *acceleration,
        #     accel,
        #     accel_scaled,
        #     accel_reward_scaled,
        #     accel_reward,
        #     reward,
        #     *action_new[:3]
        # ))

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
