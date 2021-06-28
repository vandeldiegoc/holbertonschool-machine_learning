#!/usr/bin/env python3
"""modile """
import numpy as np
import gym


def q_init(env):
    """function that initializes the Q-table"""
    action_size = env.action_space.n
    state_size = env.observation_space.n
    q_table = np.zeros(shape=(state_size, action_size))
    return q_table
