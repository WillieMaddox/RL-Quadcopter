"""
Replay Buffer.

Note: This circular buffer is only one possible way of managing finite memory.
Another approach is to randomly overwrite elements.
And you can even control the probability of each experience tuple being kept/overwritten using a priority value.
This can also be used when sampling to implement prioritized experience replay.
"""
import time
import json
import random
import pickle
from operator import attrgetter
from collections import namedtuple
import numpy as np

from .segment_tree import SumSegmentTree, MinSegmentTree

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    """Fixed-size circular buffer to store experience tuples."""

    def __init__(self, max_size=1000):
        """Initialize a ReplayBuffer object."""
        self._max_size = max_size  # maximum size of buffer
        self._next_idx = 0  # current index into circular buffer
        self._storage = []  # internal memory (list)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self._storage)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""

        s0 = state.tolist() if isinstance(state, np.ndarray) else state
        a0 = action.tolist() if isinstance(action, np.ndarray) else action
        s1 = next_state.tolist() if isinstance(next_state, np.ndarray) else next_state
        e = Experience(s0, a0, reward, s1, done)
        # Note: If memory is full, start overwriting from the beginning
        if len(self._storage) < self._max_size:
            self._storage.append(e)
        else:
            self._storage[self._next_idx] = e
        if self._next_idx + 1 == self._max_size:
            # print('*' * 80)
            # print('*' * 80)
            # print('Storage is full')
            # print('*' * 80)
            # print('*' * 80)
            # print(len(self._storage))
            # print(self._storage[0])
            # print(self._storage[-1])
            # t0 = time.time()
            self.sort()
            # print(len(self._storage), time.time() - t0)
            # print(self._storage[0])
            # print(self._storage[-1])
        self._next_idx = (self._next_idx + 1) % self._max_size

    def sort(self):
        self._storage = sorted(self._storage, key=attrgetter('reward'))

    def _encode_sample(self, idxes):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in idxes:
            e = self._storage[i]
            states.append(np.array(e.state, copy=False))
            actions.append(np.array(e.action, copy=False))
            rewards.append(e.reward)
            next_states.append(np.array(e.next_state, copy=False))
            dones.append(e.done)
        return \
            np.array(states), \
            np.array(actions), \
            np.array(rewards).astype(np.float32).reshape(-1, 1), \
            np.array(next_states), \
            np.array(dones).astype(np.uint8).reshape(-1, 1)

    def sample(self, batch_size=64, beta=0.0):
        """Randomly sample a batch of experiences from memory."""
        # Note: Make sure beta is set to zero if you are only training
        # at the end of each episode and not during.

        # grab the indices for batch_size randomly selected experiences
        sample_indices = random.sample(range(len(self._storage)), k=batch_size)

        # should the newest experience be included in the sample?
        if beta > 0 and random.random() < beta:
            # is the index of the newest experience included in the sample...?
            if self._next_idx-1 not in sample_indices:
                # get the index to a random sample
                idx = random.randint(0, batch_size-1)
                # print(len(sample_indices), idx, self.size, self.idx-1, sample_indices)
                # replace the sample at idx with newest experience
                sample_indices[idx] = self._next_idx-1
            # ...now we're sure it is.
            # sample = [self._storage[i] for i in sample_indices]
        else:
            # fall back to the original (default) behavior
            # sample = random.sample(self._storage, k=batch_size)
            pass

        return self._encode_sample(sample_indices)

    def save_json(self, filename):
        json_dict = {'idx': self._next_idx, 'experiences': self._storage}
        with open(filename, 'w') as ofs:
            json.dump(json_dict, ofs)

    def save_pkl(self, filename):
        with open(filename, 'wb') as ofs:
            pickle.dump((self._next_idx, self._storage), ofs, pickle.HIGHEST_PROTOCOL)

    def load_pkl(self, filename):
        with open(filename, 'rb') as ifs:
            self._next_idx, self._storage = pickle.load(ifs)

    def update_priorities(self, idxes, priorities):
        raise NotImplementedError("{} must override update_priorities()".format(self.__class__.__name__))


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    """
    def __init__(self, size, alpha):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)

        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha > 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            # TODO(szymon): should we ensure no repeats?
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size=64, beta=0.5):
        """Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.


        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert 0 <= beta <= 1

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


def test_run():
    """
    Test run ReplayBuffer implementation.

    Replay buffer: size = 10
    0 Experience(state=10, action=0, reward=0, next_state=11, done=2)
    1 Experience(state=11, action=1, reward=1, next_state=12, done=3)
    2 Experience(state=12, action=0, reward=-1, next_state=13, done=0)
    3 Experience(state=13, action=1, reward=0, next_state=14, done=1)
    4 Experience(state=14, action=0, reward=1, next_state=15, done=2)
    5 Experience(state=5, action=1, reward=1, next_state=6, done=1)
    6 Experience(state=6, action=0, reward=-1, next_state=7, done=2)
    7 Experience(state=7, action=1, reward=0, next_state=8, done=3)
    8 Experience(state=8, action=0, reward=1, next_state=9, done=0)
    9 Experience(state=9, action=1, reward=-1, next_state=10, done=1)

    Random batch: size = 5
    Experience(state=7, action=1, reward=0, next_state=8, done=3)
    Experience(state=5, action=1, reward=1, next_state=6, done=1)
    Experience(state=9, action=1, reward=-1, next_state=10, done=1)
    Experience(state=13, action=1, reward=0, next_state=14, done=1)
    Experience(state=12, action=0, reward=-1, next_state=13, done=0)
    """
    buf = ReplayBuffer(10)  # small buffer to test with

    # Add some sample data with a known pattern:
    #     state: i, action: 0/1, reward: -1/0/1, next_state: i+1, done: 0/1
    for i in range(15):  # more than maximum size to force overwriting
        buf.add(i, i % 2, i % 3 - 1, i + 1, i % 4)

    # Print buffer size and contents
    print("Replay buffer: size =", len(buf))  # maximum size if full
    for i, e in enumerate(buf.memory):
        print(i, e)  # should show circular overwriting

    # Randomly sample a batch
    batch = buf.sample(5)
    print("\nRandom batch: size =", len(batch))  # maximum size if full
    for e in batch:
        print(e)


if __name__ == "__main__":
    test_run()
