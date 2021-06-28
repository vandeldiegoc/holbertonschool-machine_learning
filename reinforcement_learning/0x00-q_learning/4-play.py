#!/usr/bin/env python3
""" module """
import numpy as np


def play(env, Q, max_steps=100):
    """trained agent play an episode"""
    env.reset()
    state = 0
    env.render()
    for i in range(max_steps):
        action = np.argmax(Q[state, :])
        state, reward, done, _ = env.step(action)
        env.render()
        if done:
            break
    return reward
