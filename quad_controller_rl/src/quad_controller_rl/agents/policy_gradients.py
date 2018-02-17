"""Policy search agent."""

import os
from copy import copy
import numpy as np
import pandas as pd
from quad_controller_rl import util
from quad_controller_rl.noise import OUNoise
from quad_controller_rl.noise import AdaptiveParamNoiseSpec
from quad_controller_rl.replay_buffer import ReplayBuffer
from quad_controller_rl.agents.base_agent import BaseAgent
from quad_controller_rl.agents.actor_critic import Actor
from quad_controller_rl.agents.actor_critic import Critic
# from quad_controller_rl.agents.actor_critic import get_perturbed_actor_updates


class DDPG(BaseAgent):
    """Sample agent that searches using Deep Deterministic Policy Gradients."""

    def __init__(self, task):
        super().__init__(task)
        # BaseAgent.__init__(self, task)

        # Task (environment) information
        self.task = task  # should contain observation_space and action_space

        # self.state_size = 3  # position only
        self.action_size = 3  # force only

        self.state_size = np.prod(self.task.observation_space.shape)
        # self.action_size = np.prod(self.task.action_space.shape)

        self.state_range = self.task.observation_space.high - self.task.observation_space.low
        # self.action_range = self.task.action_space.high[:3] - self.task.action_space.low[:3]

        # print("Original spaces: {}, {}\nConstrained spaces: {}, {}".format(
        #     self.task.observation_space.shape, self.task.action_space.shape,
        #     self.state_size, self.action_size))

        self.action_low = self.task.action_space.low[:3]  # / self.task.action_space.high[:3]
        self.action_high = self.task.action_space.high[:3]  # / self.task.action_space.high[:3]

        # print(self.action_low)
        # print(self.action_high)

        layer_norm = True
        # Actor (Policy) Model
        self.actor = Actor(self.state_size, self.action_size, self.action_low, self.action_high, name='local_actor', layer_norm=layer_norm)

        # Critic (Value) Model
        self.critic = Critic(self.state_size, self.action_size, name='local_critic', layer_norm=layer_norm)#, reuse=True)

        # Initialize target model parameters with local model parameters
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high, name='target_actor', layer_norm=layer_norm)
        self.actor_target.model.set_weights(self.actor.model.get_weights())
        self.critic_target = Critic(self.state_size, self.action_size, name='target_critic', layer_norm=layer_norm)
        self.critic_target.model.set_weights(self.critic.model.get_weights())

        # Noise process
        # self.action_noise = OUNoise(self.action_size)
        self.action_noise = None
        # Param noise
        self.param_noise = AdaptiveParamNoiseSpec(initial_stddev=3.0, desired_action_stddev=3.0)
        # self.param_noise = None
        self.param_noise_adaption_interval = 2
        if self.param_noise is not None:
            # self.param_noise_stddev = self.param_noise.current_stddev
            # Configure perturbed actor.
            self.perturbed_actor = Actor(self.state_size, self.action_size, self.action_low, self.action_high, name='perturbed_actor', layer_norm=layer_norm)
            self.perturbed_actor.model.set_weights(self.actor.model.get_weights())
            # Configure separate copy for stddev adoption.
            self.adaptive_actor = Actor(self.state_size, self.action_size, self.action_low, self.action_high, name='adaptive_actor', layer_norm=layer_norm)
            self.adaptive_actor.model.set_weights(self.actor.model.get_weights())

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.001  # for soft update of target parameters

        # Training variables
        self.i_episode = 0

        self.explore_schedule = np.array([47, 3, 47, 3])
        lo = np.cumsum(self.explore_schedule)[::2]
        hi = np.cumsum(self.explore_schedule)[1::2]
        self.is_explore_episode = np.ones(sum(self.explore_schedule), dtype=np.bool_)
        for i, j in zip(lo, hi):
            self.is_explore_episode[i:j] = False
        self.explore = self.is_explore_episode[self.i_episode]
        self.explore = True
        self.best_rollout_score = -np.inf
        self.best_eval_score = -np.inf

        # Episode variables
        self.last_state = None
        self.last_action = None
        self.total_reward = 0.0
        self.count = 0
        self.actor_loss = 0
        self.critic_loss = 0

        self.save_weights_episode = 100
        # path to actor model weights
        out_basename = "{}_{}".format(util.get_timestamp(), task.name)
        self.actor_filename = os.path.join(util.get_param('out'), out_basename + "_actor.h5")
        # path to actor model weights
        self.critic_filename = os.path.join(util.get_param('out'), out_basename + "_critic.h5")
        # path to episode stats CSV file
        self.stats_filename = os.path.join(util.get_param('out'), out_basename + "_stats.csv")
        # specify columns to save
        self.stats_columns = ['episode', 'total_reward']
        print("Saving stats {} to {}".format(self.stats_columns, self.stats_filename))

    def update_score(self):
        score = self.total_reward / float(self.count) if self.count else 0.0
        if self.explore:
            self.best_rollout_score = max(score, self.best_rollout_score)
            fmt = "DDPG.rollout(): t = {:4d}, score = {:7.3f} (best = {:7.3f})"
            print(fmt.format(self.count, score, self.best_rollout_score))
        else:
            self.best_eval_score = max(score, self.best_eval_score)
            fmt = "DDPG.eval()   : t = {:4d}, score = {:7.3f} (best = {:7.3f})"
            print(fmt.format(self.count, score, self.best_eval_score))

    def write_stats(self, stats):
        """Write single episode stats to CSV file."""
        df_stats = pd.DataFrame([stats], columns=self.stats_columns)  # single-row dataframe
        df_stats.to_csv(self.stats_filename, mode='a', index=False,
                        header=not os.path.isfile(self.stats_filename))  # write header first time only

    def get_perturbed_actor_updates(self):

        perturbed_actor_weights = self.perturbed_actor.model.get_weights()

        actor_weights = self.actor.model.get_weights()
        actor_params = self.actor.model.trainable_weights
        for i, ap, in enumerate(actor_params):
            if 'layer_norm' in ap.name:
                perturbed_actor_weights[i] = actor_weights[i]
            else:
                noise = np.random.randn(*ap.shape) * self.param_noise.current_stddev
                perturbed_actor_weights[i] = actor_weights[i] + noise

        self.perturbed_actor.model.set_weights(perturbed_actor_weights)

    def reset_episode_vars(self):
        self.last_state = None
        self.last_action = None
        self.total_reward = 0.0
        self.count = 0
        self.actor_loss = 0
        self.critic_loss = 0
        if self.action_noise is not None:
            self.action_noise.reset()
        if self.param_noise is not None:

            # experiences = self.memory.sample(batch_size=self.batch_size)
            # states = np.vstack([e.state for e in experiences if e is not None])

            # actions = self.actor.model.predict_on_batch(states)
            # perturbed_actions = self.perturbed_actor.model.predict_on_batch(states)
            # print('(perturbed) before:', actions[0], perturbed_actions[0])

            self.get_perturbed_actor_updates()

            # actions = self.actor.model.predict_on_batch(states)
            # perturbed_actions = self.perturbed_actor.model.predict_on_batch(states)
            # print('(perturbed) after :', actions[0], perturbed_actions[0])

    def preprocess_state(self, state):
        """Reduce state vector to relevant dimensions."""
        return state[:3]  # position only

    def postprocess_action(self, action):
        """Return complete action vector."""
        complete_action = np.zeros(self.task.action_space.shape)  # shape: (6,)
        complete_action[:3] = action  # linear force only
        return complete_action

    def pi(self, state, apply_noise=True, compute_q=True):
        """Returns actions for given state(s) as per current policy."""

        state = np.reshape(state, [-1, self.state_size])

        if self.param_noise is not None and apply_noise:
            actor = self.perturbed_actor
        else:
            actor = self.actor

        # print(self.count, self.actor.model.predict(state)[0], self.perturbed_actor.model.predict(state)[0])
        # print('   state      :' + (len(state[0]) * '{:>7.3f} ').format(*state[0]))
        action = actor.model.predict(state)
        # print('  action      :' + (len(action[0]) * '{:>7.3f} ').format(*action[0]))
        q = self.critic.model.predict([state, action]) if compute_q else None
        # print('q             :' + (len(q[0]) * '{:>7.3f} ').format(*q[0]))

        action = action.flatten()
        # print('  action      :' + (len(action) * '{:>7.3f} ').format(*action))
        if self.action_noise is not None and apply_noise:
            action += self.action_noise.sample()
        action = np.clip(action, self.action_low, self.action_high)
        # print('  action      :' + (len(action) * '{:>7.3f} ').format(*action))
        return action, q

    def act(self, state):
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor.model.predict(state)
        print('************************************************')
        print('  force before  ' + (len(action[0])*'{:>7.3f} ').format(*action[0]))
        # add some noise for exploration
        noise = self.action_noise.sample() if self.explore else np.zeros((self.action_size,))
        action = action + noise
        print('     noise      ' + (len(noise)*'{:>7.3f} ').format(*noise))
        # action = action + self.action_noise.sample() if self.explore else action
        # if done:
        #     print('force   ' + (len(action[0]) * '{:>7.3f} ').format(*action[0]))
        #     print('noise   ' + (len(noise) * '{:>7.3f} ').format(*noise))
        # norm = np.linalg.norm(actions)
        # actions = actions / norm * min(norm, self.task.max_force)
        print('  force after   ' + (len(action[0])*'{:>7.3f} ').format(*action[0]))

        # flatten, clamp to action space limits
        action_clipped = np.clip(action.flatten(), self.action_low, self.action_high)
        # if np.any(action_clipped != action):
        #     self.noise.reset()

        return action_clipped

    def get_adaptive_actor_updates(self):

        adaptive_actor_weights = self.adaptive_actor.model.get_weights()

        actor_weights = self.actor.model.get_weights()
        actor_params = self.actor.model.trainable_weights
        # adaptive_actor_params = adaptive_actor.model.trainable_weights
        for i, ap, in enumerate(actor_params):
            # print(adaptive_actor_params[i].name, adaptive_actor_params[i].shape)
            if 'layer_norm' in ap.name:
                adaptive_actor_weights[i] = actor_weights[i]
            else:
                noise = np.random.randn(*ap.shape) * self.param_noise.current_stddev
                adaptive_actor_weights[i] = actor_weights[i] + noise

        self.adaptive_actor.model.set_weights(adaptive_actor_weights)

    def adapt_param_noise(self):
        if self.param_noise is None:
            return 0.

        # Perturb a separate copy of the policy to adjust the scale for the next "real" perturbation.
        experiences = self.memory.sample(batch_size=self.batch_size)
        states = np.vstack([e.state for e in experiences if e is not None])

        # actions = self.actor.model.predict_on_batch(states)
        # adaptive_actions = self.adaptive_actor.model.predict_on_batch(states)
        # print('(adaptive)  before:', actions[0], adaptive_actions[0])

        self.get_adaptive_actor_updates()

        actions = self.actor.model.predict_on_batch(states)
        adaptive_actions = self.adaptive_actor.model.predict_on_batch(states)
        # print('(adaptive)  after :', actions[0], adaptive_actions[0])
        mean_distance = np.sqrt(np.mean(np.square(actions - adaptive_actions)))

        self.param_noise.adapt(mean_distance)
        return mean_distance

    def step(self, state, reward, done):

        # Transform state vector: scale to [0.0, 1.0]
        state = (state - self.task.observation_space.low) / self.state_range
        # state = state / self.task.observation_space.high
        # print((len(state)*'{:8.3f} ').format(*state))

        # Reduce state vector
        # state = self.preprocess_state(state)
        # print((len(state)*'{:8.3f} ').format(*state))

        # Choose an action
        action, q = self.pi(state, apply_noise=self.explore, compute_q=True)

        # Save experience / reward
        if self.last_state is not None and self.last_action is not None:
            if self.explore:
                self.memory.add(self.last_state, self.last_action, reward, state, done)
                # self.learn()
            self.total_reward += reward
            self.count += 1

        # if self.explore:
        #     self.learn()

        print('{:6} {:3} {:9.5f}'.format(self.memory.idx, self.count, q[0][0]))
        self.last_state = state
        self.last_action = action

        if done:
            if self.explore:
                self.learn()

            self.update_score()
            self.write_stats([self.i_episode, self.total_reward])

            if np.mod(self.i_episode, self.save_weights_episode) == 0:
                self.actor.model.save_weights(self.actor_filename)
                self.critic.model.save_weights(self.critic_filename)

            self.i_episode += 1
            idx = self.i_episode % sum(self.explore_schedule)
            self.explore = self.is_explore_episode[idx]
            # print(self.i_episode, idx, self.explore)
            self.reset_episode_vars()

        action = self.postprocess_action(action)
        # print((len(action)*'{:8.3f} ').format(*action))

        return action

    def learn(self):
        """Update policy and value parameters using given batch of experience tuples."""

        # Learn, if enough samples are available in memory
        if len(self.memory) < self.batch_size:
            return

        # if self.count % self.param_noise_adaption_interval == 0:
        #     distance = self.adapt_param_noise()
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        experiences = self.memory.sample(self.batch_size, prob_last=0)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        states_next = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        # Q_targets_next = critic_target(next_state, actor_target(next_state))
        # print('   states_next:' + (len(states_next[0]) * '{:>7.3f} ').format(*states_next[0]))
        actions_next = self.actor_target.model.predict_on_batch(states_next)
        # print('  actions_next:' + (len(actions_next[0]) * '{:>7.3f} ').format(*actions_next[0]))
        q_targets_next = self.critic_target.model.predict_on_batch([states_next, actions_next])
        print('q_targets_next: {:>7.4f} {:>7.4f}'.format(min(q_targets_next[:, 0]), max(q_targets_next[:, 0])))
        # Compute Q targets for current states
        q_targets = rewards + self.gamma * q_targets_next * (1 - dones)  # future_rewards

        # print(q_targets.shape)
        # print(states.shape)
        # print(self.action_size)
        # print(actions.shape)
        # print('   states     :' + (len(states[0]) * '{:>7.3f} ').format(*states[0]))
        # print('  actions     :' + (len(actions[0]) * '{:>7.3f} ').format(*actions[0]))

        # Train local critic model
        self.critic_loss = self.critic.model.train_on_batch(x=[states, actions], y=q_targets)

        action_gradients = self.critic.get_action_gradients([states, actions, 0])
        action_gradients = np.reshape(action_gradients, (-1, self.action_size))

        # Train local actor model
        self.actor_loss = self.actor.set_action_gradients([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic.model, self.critic_target.model)
        self.soft_update(self.actor.model, self.actor_target.model)

        fmt = '{}: actor loss = {:7.4f}, critic loss = {:7.4f}, distance = {:7.4f}, stdev = {:7.4f}'
        print(fmt.format(self.i_episode, self.actor_loss[0], self.critic_loss, distance, self.param_noise.current_stddev))

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
