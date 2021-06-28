#!/usr/bin/env python3
"""module """
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """ that performs Q-learning:"""
    rewards_per_episode = []
    max_epsilon = epsilon

    for e in range(episodes):
        current_state = env.reset()
        done = False
        rewards_current_episode = 0

        for step in range(max_steps):

            action = epsilon_greedy(Q, current_state, epsilon)
            next_state, reward, done, _ = env.step(action)

            if done is True and reward == 0:
                reward = -1

            Q[current_state, action] = (1 - alpha) * \
                Q[current_state, action] + alpha * \
                (reward + gamma*max(Q[next_state, :]))

            rewards_current_episode += reward
            current_state = next_state

            if done:
                break

        rewards_per_episode.append(rewards_current_episode)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * \
            np.exp(-epsilon_decay * e)
    return Q, rewards_per_episode
