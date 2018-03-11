"""Utility functions."""

import pandas as pd
import rospy

from datetime import datetime
import numpy as np


def get_param(name):
    """Return parameter value specified in ROS launch file or via command line, e.g. agent:=DDPG."""
    return rospy.get_param(name)


def get_timestamp(t=None, format='%Y-%m-%d_%H-%M-%S'):
    """Return timestamp as a string; default: current time, format: YYYY-DD-MM_hh-mm-ss."""
    if t is None:
        t = datetime.now()
    return t.strftime(format)


def plot_stats(csv_filename, columns=['total_reward'], **kwargs):
    """Plot specified columns from CSV file."""
    df_stats = pd.read_csv(csv_filename)
    df_stats[columns].plot(**kwargs)


def normalize(x, stats):
    # if stats is None:
    #     return x
    # return (x - stats.mean) / stats.std
    return x if stats is None else (x - stats.mean) / stats.std


def denormalize(x, stats):
    # if stats is None:
    #     return x
    # return x * stats.std + stats.mean
    return x if stats is None else x * stats.std + stats.mean


class RunningMeanStd(object):
    """
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py
    """
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.std = np.ones(shape, 'float64')
        self.count = epsilon
        self.call_flag = False

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.std = np.sqrt(new_var)
        self.count = new_count


class LinearSchedule(object):
    """
    https://github.com/openai/baselines/blob/master/baselines/common/schedules.py
    """
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.

        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


if __name__ == '__main__':
    ls = LinearSchedule(10, 0.78)
    for i in range(20):
        print(ls.value(i))