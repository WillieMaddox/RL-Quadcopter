"""Policy search agent."""

import os
from copy import copy
import numpy as np
import pandas as pd

from quad_controller_rl import util
from quad_controller_rl.noise import OUNoise
from quad_controller_rl.noise import AdaptiveParamNoiseSpec
from quad_controller_rl.replay_buffer import ReplayBuffer
from quad_controller_rl.replay_buffer import PrioritizedReplayBuffer
from quad_controller_rl.agents.base_agent import BaseAgent
from quad_controller_rl.agents.actor_critic import Actor
from quad_controller_rl.agents.actor_critic import Critic
# from quad_controller_rl.agents.actor_critic import get_perturbed_actor_updates

EXPLORE = 1
EXPLOIT = 2
RANDOM = 3


class DDPG(BaseAgent):
    """Sample agent that searches using Deep Deterministic Policy Gradients."""

    def __init__(self, task):
        super().__init__(task)
        # BaseAgent.__init__(self, task)

        # Task (environment) information
        self.task = task  # should contain observation_space and action_space

        self.state_size = np.prod(self.task.observation_space.shape)
        # self.state_size = 3  # position only

        # self.action_size = np.prod(self.task.action_space.shape)
        self.action_size = 3  # force only

        # self.state_range = self.task.observation_space.high - self.task.observation_space.low
        # self.action_range = self.task.action_space.high[:3] - self.task.action_space.low[:3]

        self.state_low = self.task.observation_space.low[:self.state_size]
        self.state_high = self.task.observation_space.high[:self.state_size]
        self.state_range = self.state_high - self.state_low

        self.action_low = self.task.action_space.low[:self.action_size]
        self.action_high = self.task.action_space.high[:self.action_size]
        self.action_range = self.action_high - self.action_low

        # print(self.action_low)
        # print(self.action_high)

        # OU Noise
        self.ou_noise = OUNoise(self.action_size)
        # self.ou_noise = None

        # Param noise
        # self.param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.1)
        self.param_noise = None

        layer_norm = self.param_noise is not None
        # Actor (Policy) Model
        self.actor = Actor(self.state_size, self.action_size, self.action_low, self.action_high, name='local_actor', layer_norm=layer_norm)

        # Critic (Value) Model
        self.critic = Critic(self.state_size, self.action_size, name='local_critic', layer_norm=layer_norm)  # , reuse=True)

        # Initialize target model parameters with local model parameters
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high, name='target_actor', layer_norm=layer_norm)
        self.actor_target.model.set_weights(self.actor.model.get_weights())
        self.critic_target = Critic(self.state_size, self.action_size, name='target_critic', layer_norm=layer_norm)
        self.critic_target.model.set_weights(self.critic.model.get_weights())

        if self.param_noise is not None:
            # self.param_noise_stddev = self.param_noise.current_stddev
            self.param_noise_adaption_interval = 55
            # Configure perturbed actor.
            self.perturbed_actor = Actor(self.state_size, self.action_size, self.action_low, self.action_high, name='perturbed_actor', layer_norm=layer_norm)
            self.perturbed_actor.model.set_weights(self.actor.model.get_weights())
            # Configure separate copy for stddev adoption.
            self.adaptive_actor = Actor(self.state_size, self.action_size, self.action_low, self.action_high, name='adaptive_actor', layer_norm=layer_norm)
            self.adaptive_actor.model.set_weights(self.actor.model.get_weights())

        # Replay Buffer
        self.buffer_size = 75000
        self.batch_size = 64

        max_timesteps = 1000000
        self.i_timestep = 0

        self.prioritized_replay = False
        if self.prioritized_replay:
            prioritized_replay_alpha = 0.6
            prioritized_replay_beta0 = 0.4
            prioritized_replay_beta_iters = None
            self.prioritized_replay_eps = 1e-6

            self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size, alpha=prioritized_replay_alpha)
            if prioritized_replay_beta_iters is None:
                prioritized_replay_beta_iters = max_timesteps
            self.beta_schedule = util.LinearSchedule(
                prioritized_replay_beta_iters,
                initial_p=prioritized_replay_beta0,
                final_p=1.0)
        else:
            self.replay_buffer = ReplayBuffer(self.buffer_size)
            self.beta_schedule = None

        exploration_fraction = 0.5
        exploration_final_eps = 0.6
        # Create the schedule for exploration starting from 1.
        self.exploration = util.LinearSchedule(
            schedule_timesteps=int(exploration_fraction * max_timesteps),
            initial_p=1.0,
            final_p=exploration_final_eps)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.001  # for soft update of target parameters

        # Training variables
        self.rollout_only = 4
        self.training_interval = 25
        self.learn_delay = 0
        self.i_episode = 1
        self.i_explore_episode = 1
        self.i_exploit_episode = 1
        self.n_explore_episodes = 50
        self.n_exploit_episodes = 2
        self.save_weights_episode = 50
        self.stabilize = True
        self.policy = None

        self.explore = not self.stabilize
        self.best_explore_score = -np.inf
        self.best_exploit_score = -np.inf
        self.state_rms = util.RunningMeanStd(shape=self.state_size)
        self.action_rms = util.RunningMeanStd(shape=self.action_size)

        # Episode variables
        self.prev_state = None
        self.prev_action = None
        self.prev_q = None
        self.total_reward = 0.0
        self.prev_actions = []
        self.prev_qs = []
        self.count = 0
        self.train_count = 0
        self.adapt_count = 0
        self.actor_loss = 0.0
        self.critic_loss = 0.0

        if self.param_noise is not None:
            self.perturb_actor(self.perturbed_actor.model)

        out_basename = "{}_{}".format(util.get_timestamp(), task.name)
        # path to actor model weights
        self.actor_filename = os.path.join(util.get_param('out'), out_basename + "_actor.h5")
        # path to critic model weights
        self.critic_filename = os.path.join(util.get_param('out'), out_basename + "_critic.h5")
        # path to experience replay
        self.replay_buffer_filename = os.path.join(util.get_param('out'), out_basename + "_replay.pkl")
        # path to explore episode stats CSV file
        self.explore_stats_filename = os.path.join(util.get_param('out'), out_basename + "_explore_stats.csv")
        # path to exploit episode stats CSV file
        self.exploit_stats_filename = os.path.join(util.get_param('out'), out_basename + "_exploit_stats.csv")
        # path to training stats CSV file
        self.training_stats_filename = os.path.join(util.get_param('out'), out_basename + "_training_stats.csv")
        # specify columns to save for explore/exploit
        self.stats_columns = [
            'episode',
            'episode2',
            'count',
            'exploration_eps',
            'total_reward',
            'Fx_mean', 'Fy_mean', 'Fz_mean',
            'Fx_std', 'Fy_std', 'Fz_std',
            'Q_mean', 'Q_std']
        # specify columns to save for train/learn
        self.training_stats_columns = [
            'episode', 'n_dones', 'replay_idx',
            'Fx_mean', 'Fy_mean', 'Fz_mean',
            'Fx_std', 'Fy_std', 'Fz_std',
            'Fx_next_mean', 'Fy_next_mean', 'Fz_next_mean',
            'Fx_next_std', 'Fy_next_std', 'Fz_next_std',
            'Fx_norm_mean', 'Fy_norm_mean', 'Fz_norm_mean',
            'Fx_norm_std', 'Fy_norm_std', 'Fz_norm_std',
            'Fx_next_norm_mean', 'Fy_next_norm_mean', 'Fz_next_norm_mean',
            'Fx_next_norm_std', 'Fy_next_norm_std', 'Fz_next_norm_std',
            'grad_x_mean', 'grad_y_mean', 'grad_z_mean',
            'grad_x_std', 'grad_y_std', 'grad_z_std',
            'Q_next_mean', 'Q_next_std',
            'Q_mean', 'Q_std',
            'reward_mean', 'reward_std',
            'reward_min', 'reward_max',
            'actor_loss', 'critic_loss']

    def write_training_stats(self, actions, actions_next,
                             actions_norm, actions_next_norm,
                             q_targets_next, q_targets,
                             rewards, dones, action_gradients):

        """Write training episode stats to CSV file."""
        avg_a = np.mean(actions, axis=0)
        std_a = np.std(actions, axis=0)
        avg_ax = np.mean(actions_next, axis=0)
        std_ax = np.std(actions_next, axis=0)
        avg_an = np.mean(actions_norm, axis=0)
        std_an = np.std(actions_norm, axis=0)
        avg_axn = np.mean(actions_next_norm, axis=0)
        std_axn = np.std(actions_next_norm, axis=0)
        avg_ag = np.mean(action_gradients, axis=0)
        std_ag = np.std(action_gradients, axis=0)
        avg_qn = np.mean(q_targets_next, axis=0)
        std_qn = np.std(q_targets_next, axis=0)
        avg_q = np.mean(q_targets, axis=0)
        std_q = np.std(q_targets, axis=0)
        avg_r = np.mean(rewards, axis=0)
        std_r = np.std(rewards, axis=0)
        min_r = np.min(rewards, axis=0)
        max_r = np.max(rewards, axis=0)

        stats = [
            self.i_episode, np.sum(dones),
            self.replay_buffer._next_idx,
            *avg_a,
            *std_a,
            *avg_ax,
            *std_ax,
            *avg_an,
            *std_an,
            *avg_axn,
            *std_axn,
            *avg_ag,
            *std_ag,
            *avg_qn, *std_qn,
            *avg_q, *std_q,
            *avg_r, *std_r,
            *min_r, *max_r,
            self.actor_loss[0], self.critic_loss
        ]

        if self.policy == EXPLORE:
            df_stats = pd.DataFrame([stats], columns=self.training_stats_columns)  # single-row dataframe
            df_stats.to_csv(self.training_stats_filename, mode='a', index=False,
                            header=not os.path.isfile(self.training_stats_filename))  # write header first time only

    def write_stats(self, i_episode):
        """Write single episode stats to CSV file."""
        # print(self.prev_actions)
        avg_force = np.mean(self.prev_actions, axis=0)
        # print(avg_force)
        std_force = np.std(self.prev_actions, axis=0)
        stats = [
            self.i_episode,
            i_episode,
            self.count,
            self.exploration.value(self.i_timestep),
            self.total_reward,
            *avg_force,
            *std_force,
            np.mean(self.prev_qs), np.std(self.prev_qs)
        ]

        if self.policy == EXPLORE:
            df_stats = pd.DataFrame([stats], columns=self.stats_columns)  # single-row dataframe
            df_stats.to_csv(self.explore_stats_filename, mode='a', index=False,
                            header=not os.path.isfile(self.explore_stats_filename))  # write header first time only
        elif self.policy == EXPLOIT:
            df_stats = pd.DataFrame([stats], columns=self.stats_columns)  # single-row dataframe
            df_stats.to_csv(self.exploit_stats_filename, mode='a', index=False,
                            header=not os.path.isfile(self.exploit_stats_filename))  # write header first time only

    def update_score(self):
        score = self.total_reward / float(self.count) if self.count else 0.0
        if self.policy == EXPLORE:
            self.best_explore_score = max(score, self.best_explore_score)
            fmt = "DDPG.explore({}): t = {:3d}, score = {:7.3f} (best = {:7.3f})"
            print(fmt.format(self.i_explore_episode, self.count, score, self.best_explore_score))
        elif self.policy == EXPLOIT:
            self.best_exploit_score = max(score, self.best_exploit_score)
            fmt = "DDPG.exploit({}): t = {:3d}, score = {:7.3f} (best = {:7.3f})"
            print(fmt.format(self.i_exploit_episode, self.count, score, self.best_exploit_score))

    def perturb_actor(self, perturbed_model):

        perturbed_actor_weights = perturbed_model.get_weights()
        # perturbed_actor_params = perturbed_model.trainable_weights

        actor_weights = self.actor.model.get_weights()
        actor_params = self.actor.model.trainable_weights
        for i, ap, in enumerate(actor_params):

            if 'layer_norm' in ap.name:
                # print('  {} <- {}'.format(perturbed_actor_params[i].name, ap.name),
                #       perturbed_actor_params[i].shape, perturbed_actor_weights[i].shape, actor_weights[i].shape, ap.shape)
                perturbed_actor_weights[i] = actor_weights[i]
            else:
                # print(perturbed_actor_params[i].name, perturbed_actor_params[i].shape)
                noise = np.random.randn(*ap.shape) * self.param_noise.current_stddev
                # print('  {} <- {} + noise'.format(perturbed_actor_params[i].name, ap.name),
                #       perturbed_actor_params[i].shape, perturbed_actor_weights[i].shape, actor_weights[i].shape, ap.shape, noise.shape)
                perturbed_actor_weights[i] = actor_weights[i] + noise
                # print(actor_weights[i])
                # print(noise)
                # print(perturbed_actor_weights[i])

        perturbed_model.set_weights(perturbed_actor_weights)

    def reset_episode_vars(self):
        self.prev_state = None
        self.prev_action = None
        self.prev_q = None
        self.total_reward = 0.0
        self.prev_actions = []
        self.prev_qs = []
        self.count = 0
        self.train_count = 0
        self.adapt_count = 0
        self.actor_loss = 0.0
        self.critic_loss = 0.0
        if self.ou_noise is not None:
            self.ou_noise.reset()
        if self.param_noise is not None:

            # if len(self.replay_buffer) < self.batch_size:
            #     return 0.
            #
            # ffmt = '{:9.5f}'
            # states = self.replay_buffer.sample(self.batch_size)[0]
            # states = util.normalize(states, self.state_rms_real)
            # actions = self.actor.model.predict_on_batch(states)
            # fmt = [
            #     '[' + ' '.join([ffmt, ffmt, ffmt]) + ']',
            #     '[' + ' '.join([ffmt, ffmt, ffmt]) + ']',
            #     ffmt,
            # ]
            # print('(perturbed) before: ' + ' '.join(fmt).format(
            #     *np.mean(actions, axis=0),
            #     *np.std(actions, axis=0),
            #     self.param_noise.current_stddev,
            # ))

            self.perturb_actor(self.perturbed_actor.model)

            # perturbed_actions = self.perturbed_actor.model.predict_on_batch(states)
            # fmt = [
            #     '[' + ' '.join([ffmt, ffmt, ffmt]) + ']',
            #     '[' + ' '.join([ffmt, ffmt, ffmt]) + ']',
            #     ffmt,
            # ]
            # print('(perturbed) after : ' + ' '.join(fmt).format(
            #     *np.mean(perturbed_actions, axis=0),
            #     *np.std(perturbed_actions, axis=0),
            #     self.param_noise.current_stddev,
            # ))

    def preprocess_state(self, state):
        """Reduce state vector to relevant dimensions."""
        return state[:self.state_size]  # position only

    def postprocess_action(self, action):
        """Return complete action vector."""
        complete_action = np.zeros(self.task.action_space.shape)  # shape: (6,)
        complete_action[:self.action_size] = action  # linear force only
        return complete_action

    def act(self, state, compute_q=True):
        """Returns actions for given state(s) as per current policy."""

        state = np.reshape(state, [-1, self.state_size])
        self.state_rms.update(state)
        state_norm = util.normalize(state, self.state_rms)

        if self.explore and self.param_noise is not None:
            actor = self.perturbed_actor
        else:
            actor = self.actor

        action = actor.model.predict(state_norm)

        action = action.flatten()
        if self.explore and self.ou_noise is not None:
            action += self.ou_noise.sample()
        action = np.clip(action, self.action_low, self.action_high)
        # action = np.clip(action.flatten(), self.action_low, self.action_high)

        action_norm = np.reshape(action, [-1, self.action_size])
        self.action_rms.update(action_norm)
        action_norm = util.normalize(action_norm, self.action_rms)

        q = self.critic.model.predict([state_norm, action_norm]) if compute_q else None

        # ffmt = '{:8.4f}'
        # explore = 'explore' if self.explore else 'exploit'
        # fmt = [
        #     '({}) step{:>2}:',
        #     '[' + ' '.join([ffmt, ffmt, ffmt, ffmt, ffmt, ffmt]) + ']',
        #     '[' + ' '.join([ffmt, ffmt, ffmt]) + ']',
        #     '[' + ' '.join([ffmt, ffmt, ffmt]) + ']',
        #     '{:9.4f}',
        #     # '[' + ' '.join([ffmt, ffmt, ffmt]) + ']',
        #     # '[' + ' '.join([ffmt, ffmt, ffmt]) + ']',
        #     # '[' + ' '.join([ffmt, ffmt, ffmt]) + ']',
        #     # '[' + ' '.join([ffmt, ffmt, ffmt]) + ']',
        # ]
        # print(' '.join(fmt).format(
        #     explore,
        #     self.count,
        #     *state[0],
        #     *action[0],
        #     *action_norm[0],
        #     *q[0]
        #     # *self.state_rms.mean,
        #     # *self.state_rms.std,
        #     # *self.action_rms.mean,
        #     # *self.action_rms.std,
        # ))

        return action, q

    def adapt_param_noise(self):
        if self.param_noise is None:
            return 0.

        if len(self.replay_buffer) < self.batch_size:
            return 0.

        if self.i_episode < self.rollout_only:
            return 0.

        ffmt = '{:9.5f}'
        # Perturb a separate copy of the policy to adjust the scale for the next "real" perturbation.
        states = self.replay_buffer.sample(self.batch_size)[0]
        # states = np.vstack([e.state for e in experiences if e is not None])
        # print('1' * 50)
        # print(states)
        # states = util.normalize(states, self.state_rms_real)
        # print('2' * 50)
        # print(states)
        # states = np.clip(states, self.state_low, self.state_high)
        # print('4' * 50)
        # print(states)
        actions = self.actor.model.predict_on_batch(states)
        # adaptive_actions = self.adaptive_actor.model.predict_on_batch(states)
        # fmt = [
        #     # '[' + ' '.join([ffmt, ffmt, ffmt]) + ']',
        #     '[' + ' '.join([ffmt, ffmt, ffmt]) + ']',
        #     # '[' + ' '.join([ffmt, ffmt, ffmt]) + ']',
        #     '[' + ' '.join([ffmt, ffmt, ffmt]) + ']',
        #     # '[' + ' '.join([ffmt, ffmt, ffmt]) + ']',
        #     # '[' + ' '.join([ffmt, ffmt, ffmt]) + ']',
        #     # '[' + ' '.join([ffmt, ffmt, ffmt]) + ']',
        #     # '[' + ' '.join([ffmt, ffmt, ffmt]) + ']',
        #     ffmt,
        # ]
        # print('(adaptive)  before: ' + ' '.join(fmt).format(
        #     # *actions[0],
        #     *np.mean(actions, axis=0),
        #     # *np.median(actions, axis=0),
        #     *np.std(actions, axis=0),
        #     # *adaptive_actions[0],
        #     # *np.mean(adaptive_actions, axis=0),
        #     # *np.median(adaptive_actions, axis=0),
        #     # *np.std(adaptive_actions, axis=0),
        #     self.param_noise.current_stddev,
        # ))

        self.perturb_actor(self.adaptive_actor.model)

        # actions = self.actor.model.predict_on_batch(states)
        adaptive_actions = self.adaptive_actor.model.predict_on_batch(states)
        # fmt = [
        #     # '[' + ' '.join([ffmt, ffmt, ffmt]) + ']',
        #     # '[' + ' '.join([ffmt, ffmt, ffmt]) + ']',
        #     # '[' + ' '.join([ffmt, ffmt, ffmt]) + ']',
        #     # '[' + ' '.join([ffmt, ffmt, ffmt]) + ']',
        #     # '[' + ' '.join([ffmt, ffmt, ffmt]) + ']',
        #     '[' + ' '.join([ffmt, ffmt, ffmt]) + ']',
        #     # '[' + ' '.join([ffmt, ffmt, ffmt]) + ']',
        #     '[' + ' '.join([ffmt, ffmt, ffmt]) + ']',
        #     # '{:9.5f} ',
        # ]
        # print('(adaptive)  after : ' + ' '.join(fmt).format(
        #     # *actions[0],
        #     # *np.mean(actions, axis=0),
        #     # *np.median(actions, axis=0),
        #     # *np.std(actions, axis=0),
        #     # *adaptive_actions[0],
        #     *np.mean(adaptive_actions, axis=0),
        #     # *np.median(adaptive_actions, axis=0),
        #     *np.std(adaptive_actions, axis=0),
        #     # self.param_noise.current_stddev,
        # ))

        actions = (actions - self.action_low) / self.action_range
        adaptive_actions = (adaptive_actions - self.action_low) / self.action_range
        # print('1', actions)
        # print('2', adaptive_actions)
        # print('3', np.square(actions - adaptive_actions))
        # print('4', np.mean(np.square(actions - adaptive_actions), axis=0))
        # print('5', np.sqrt(np.mean(np.square(actions - adaptive_actions), axis=0)))
        mean_distance = np.sqrt(np.mean(np.square(actions - adaptive_actions)))
        # print('5', mean_distance)

        # fmt = [
        #     # '[' + ' '.join([ffmt, ffmt, ffmt]) + ']',
        #     # '[' + ' '.join([ffmt, ffmt, ffmt]) + ']',
        #     # '[' + ' '.join([ffmt, ffmt, ffmt]) + ']',
        #     # '[' + ' '.join([ffmt, ffmt, ffmt]) + ']',
        #     # '[' + ' '.join([ffmt, ffmt, ffmt]) + ']',
        #     '[' + ' '.join([ffmt, ffmt, ffmt]) + ']',
        #     # '[' + ' '.join([ffmt, ffmt, ffmt]) + ']',
        #     '[' + ' '.join([ffmt, ffmt, ffmt]) + ']',
        #     ffmt,
        # ]
        # print('(adaptive)  scale : ' + ' '.join(fmt).format(
        #     # *actions[0],
        #     # *np.mean(actions, axis=0),
        #     # *np.median(actions, axis=0),
        #     # *np.std(actions, axis=0),
        #     # *adaptive_actions[0],
        #     *np.mean(adaptive_actions, axis=0),
        #     # *np.median(adaptive_actions, axis=0),
        #     *np.std(adaptive_actions, axis=0),
        #     mean_distance,
        # ))

        self.param_noise.adapt(mean_distance)
        return mean_distance

    def step(self, state, reward, done):

        # Reduce state vector
        state = self.preprocess_state(state)
        # print((len(state)*'{:8.3f} ').format(*state))

        # Transform state vector: scale to [0.0, 1.0]
        # state = (state - self.state_low) / self.state_range
        # print((len(state)*'{:8.3f} ').format(*state))

        if self.policy == RANDOM:
            # Choose whether we want to explore or exploit. Favor exploration initially.
            self.explore = np.random.random() < self.exploration.value(self.i_timestep)
        else:
            self.explore = self.policy == EXPLORE

        # Choose an action
        action, q = self.act(state, compute_q=True)

        # Skip the very first episode.  It's kinda unstable.
        if self.stabilize:
            self.count += 1
            if done:
                self.stabilize = False
                self.explore = True
                self.reset_episode_vars()
            action = self.postprocess_action(action)
            return action

        # Save experience / reward
        if self.prev_state is not None and self.prev_action is not None:
            # print((len(state) * '{:8.3f} ').format(*state))
            # self.replay_buffer.add(self.prev_state, self.prev_action, reward, state, done)
            if self.policy == RANDOM:
                self.replay_buffer.add(self.prev_state, self.prev_action, reward, state, done)
                self.learn_delay += 1
            #
            #     if self.explore:
            #         self.train_count += 1
            #         if self.train_count % self.training_interval == 0:
            #             self.learn(beta=0.0)
            #     else:
            #         self.adapt_count += 1
            #         if self.param_noise and self.adapt_count % self.param_noise_adaption_interval == 0:
            #             self.adapt_param_noise()
            #             self.learn(beta=0.0)
            #
            if self.policy == EXPLORE:
                self.replay_buffer.add(self.prev_state, self.prev_action, reward, state, done)
                self.learn_delay += 1
            #
            #     if self.param_noise and self.count % self.param_noise_adaption_interval == 0:
            #         self.adapt_param_noise()
            #         self.learn(beta=0.0)
            #
            #     if self.count % self.training_interval == 0:
            #         self.learn(beta=0.0)

            self.total_reward += reward
            self.prev_actions.append(self.prev_action)
            self.prev_qs.append(self.prev_q)

        self.count += 1
        self.prev_state = state
        self.prev_action = action
        self.prev_q = q

        if done:
            self.update_score()

            if self.policy == EXPLORE:
                self.write_stats(self.i_explore_episode)
                self.learn()

                if self.i_explore_episode % self.save_weights_episode == 0:
                    self.actor.model.save_weights(self.actor_filename)
                    self.critic.model.save_weights(self.critic_filename)
                    print('Experiences:', self.replay_buffer._next_idx)
                    self.replay_buffer.save_pkl(self.replay_buffer_filename)

                self.explore = self.i_explore_episode % self.n_explore_episodes != 0
                self.i_explore_episode += 1
                rv = np.random.random()
                if rv > 0.25:
                    self.policy = EXPLORE
                    # print('Begin EXPLORE')
                elif rv > 0.10:
                    self.policy = EXPLOIT
                    # print('Begin EXPLOIT')
                else:
                    self.policy = RANDOM
                    # print('Begin RANDOM')

            elif self.policy == EXPLOIT:
                self.write_stats(self.i_exploit_episode)
                self.adapt_param_noise()

                self.explore = self.i_exploit_episode % self.n_exploit_episodes == 0
                self.i_exploit_episode += 1
                self.policy = EXPLORE
                # print('Begin EXPLORE')

            else:
                self.policy = EXPLORE
                # print('Begin EXPLORE')

            self.reset_episode_vars()
            self.i_episode += 1

        self.i_timestep += 1
        action = self.postprocess_action(action)
        # print((len(action)*'{:8.3f} ').format(*action))

        return action

    def learn(self, beta=0.0):
        """Update policy and value parameters using given batch of experience tuples."""

        # Learn, if enough samples are available in replay_buffer
        if len(self.replay_buffer) < self.batch_size:
            return

        if self.learn_delay < self.batch_size:
            return
        else:
            self.learn_delay = 0

        if self.i_episode < self.rollout_only:
            return

        if self.prioritized_replay:
            experience = self.replay_buffer.sample(self.batch_size, beta=self.beta_schedule.value(self.i_timestep))
            states, actions, rewards, states_next, dones, weights, batch_idxes = experience
        else:
            states, actions, rewards, states_next, dones = self.replay_buffer.sample(self.batch_size, beta=beta)
            weights, batch_idxes = np.ones_like(rewards), None

        states_norm = util.normalize(states, self.state_rms)
        actions_norm = util.normalize(actions, self.action_rms)
        states_next_norm = util.normalize(states_next, self.state_rms)
        # self.state_rms.update(states)

        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target.model.predict_on_batch(states_next_norm)
        actions_next_norm = util.normalize(actions_next, self.action_rms)
        # q_targets_next tells us how much reward we expect to get by taking actions_next in states_next.
        q_targets_next = self.critic_target.model.predict_on_batch([states_next_norm, actions_next_norm])
        # Compute Q targets for current states
        q_targets = rewards + self.gamma * q_targets_next * (1 - dones)  # future_rewards

        # Train local critic model
        self.critic_loss = self.critic.model.train_on_batch(x=[states_norm, actions_norm], y=q_targets)

        # Train local actor model
        action_gradients = self.critic.get_action_gradients([states_norm, actions_norm, 0])
        action_gradients = np.reshape(action_gradients, (-1, self.action_size))
        self.actor_loss = self.actor.set_action_gradients([states_norm, action_gradients, 1])  # custom training function

        if self.prioritized_replay:
            q_hat = self.critic_target.model.predict_on_batch([states_norm, actions_norm])
            td_errors = q_targets - q_hat
            new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
            # print(weights)
            # ffmt = '{:9.5f}'
            # fmt2 = [
            #     '{:>9}',
            #     ffmt,
            #     ffmt,
            # ]
            # for i in range(self.batch_size):
            #     print(' '.join(fmt2).format(
            #         batch_idxes[i],
            #         td_errors[i, 0],
            #         new_priorities[i, 0],
            #     ))

            self.replay_buffer.update_priorities(batch_idxes, new_priorities)

        # Soft-update target models
        self.soft_update(self.critic.model, self.critic_target.model)
        self.soft_update(self.actor.model, self.actor_target.model)

        # ffmt = '{:9.5f}'
        # fmt = [
        #     'DDPG.learn({},{:>3}):  ',
        #     ffmt,
        #     ffmt,
        #     ffmt,
        #     ffmt,
        #     'actor loss = {:7.4f},',
        #     'critic loss = {:7.4f},'
        # ]
        # print(' '.join(fmt).format(
        #     self.i_episode,
        #     self.count,
        #     *np.mean(q_targets_next, axis=0),
        #     *np.std(q_targets_next, axis=0),
        #     *np.mean(q_targets, axis=0),
        #     *np.std(q_targets, axis=0),
        #     self.actor_loss[0],
        #     self.critic_loss
        # ))

        self.write_training_stats(actions, actions_next, actions_norm, actions_next_norm, q_targets_next, q_targets, rewards, dones, action_gradients)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)


