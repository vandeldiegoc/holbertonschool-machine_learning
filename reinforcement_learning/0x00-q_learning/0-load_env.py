#!/usr/bin/env python3
"""module"""
import gym


def load_frozen_lake(desc=None, map_name=None,
                     is_slippery=False):
    """that loads the pre-made FrozenLakeEnv
       evnironment from OpenAI’s gym"""
    frozen = gym.make("FrozenLake-v0", desc=desc,
                      map_name=map_name, is_slippery=is_slippery)
    return frozen
