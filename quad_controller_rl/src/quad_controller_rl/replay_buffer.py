"""
Replay Buffer.

Note: This circular buffer is only one possible way of managing finite memory.
Another approach is to randomly overwrite elements.
And you can even control the probability of each experience tuple being kept/overwritten using a priority value.
This can also be used when sampling to implement prioritized experience replay.
"""
import json
import random
import pickle
from collections import namedtuple
import numpy as np

Experience = namedtuple("Experience",
                        field_names=["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    """Fixed-size circular buffer to store experience tuples."""

    def __init__(self, max_size=1000):
        """Initialize a ReplayBuffer object."""
        self.max_size = max_size  # maximum size of buffer
        self.size = 0  # current size of buffer
        self.idx = 0  # current index into circular buffer
        self.memory = []  # internal memory (list)
        # self.outfile = 'current.pkl'

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""

        s0 = state.tolist() if isinstance(state, np.ndarray) else state
        a0 = action.tolist() if isinstance(state, np.ndarray) else action
        s1 = next_state.tolist() if isinstance(state, np.ndarray) else next_state
        e = Experience(s0, a0, reward, s1, done)
        # Note: If memory is full, start overwriting from the beginning
        if self.size < self.max_size:
            self.memory.append(e)
            self.size += 1
        else:
            self.memory[self.idx] = e
        self.idx = (self.idx + 1) % self.max_size

        # if self.size in (999, 1000) or self.idx in (999, 1000, 0, 1, 2, 3, 4):
        #     print('*************ADD*******************')
        #     print(self.memory[self.idx - 1])

    def sample(self, batch_size=64, prob_last=0):
        """Randomly sample a batch of experiences from memory."""
        # Note: Make sure prob_last is set to zero if you are only training
        # at the end of each episode and not during.

        # should the newest experience be included in the sample?
        if prob_last > 0 and random.random() < prob_last:
            # grab the indices for batch_size randomly selected experiences
            sample_indices = random.sample(range(self.size), k=batch_size)
            # is the index of the newest experience included in the sample...?
            if self.idx-1 not in sample_indices:
                # get the index to a random sample
                idx = random.randint(0, batch_size-1)
                # print(len(sample_indices), idx, self.size, self.idx-1, sample_indices)
                # replace the sample at idx with newest experience
                sample_indices[idx] = self.idx-1
            # ...now we're sure it is.
            sample = [self.memory[i] for i in sample_indices]
            # if self.size in (999, 1000) or self.idx in (999, 1000, 0, 1):
            #     print('************SAMPLE******************')
            #     print(self.memory[self.idx-1])
        else:
            # fall back to the original (default) behavior
            sample = random.sample(self.memory, k=batch_size)
        return sample

    def save_json(self, filename):
        json_dict = {'idx': self.idx, 'experiences': self.memory}
        with open(filename, 'w') as ofs:
            json.dump(json_dict, ofs)

    def save_pkl(self, filename):
        with open(filename, 'wb') as ofs:
            pickle.dump((self.idx, self.memory), ofs, pickle.HIGHEST_PROTOCOL)

    def load_pkl(self, filename):
        with open(filename, 'rb') as ifs:
            self.idx, self.memory = pickle.load(ifs)
        self.size = len(self.memory)

    def __len__(self):
        """Return the current size of internal memory."""
        return self.size


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
