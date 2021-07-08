#!/usr/bin/env python3
import gym
import numpy as np
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.callbacks import FileLogger
from rl.processors import Processor
from rl.memory import SequentialMemory
import tensorflow.keras as k
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Permute
from tensorflow.keras.optimizers import Adam
from PIL import Image


INPUT_SHAPE = (84, 84)


class AtariProcessor(Processor):
    def process_observation(self, observation):
        # (height, width, channel)
        assert observation.ndim == 3
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')
        # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')
        # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`.
        # In this case, however,
        # we would need to store a `float32` array
        # instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


def create_cnn(action):
    model = k.models.Sequential()
    model.add(Permute((2, 3, 1), input_shape=(window, 84, 84)))
    model.add(Conv2D(32, 8, strides=4, activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(action, activation='linear'))
    return model

if __name__ == '__main__':

    env = gym.make("Breakout-v0")
    env.reset()
    nb_actions = env.action_space.n
    input_shape = env.observation_space.shape
    print(input_shape)
    window = 4
    model = create_cnn(nb_actions)
    model.summary()

    memory = SequentialMemory(limit=1000000, window_length=4)
    processor = AtariProcessor()
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
                                  value_max=1., value_min=.1,
                                  value_test=.05,
                                  nb_steps=1000000)

    dqn = DQNAgent(model=model, nb_actions=nb_actions,
                   policy=policy, memory=memory,
                   processor=processor,
                   nb_steps_warmup=50000, gamma=.99,
                   target_model_update=10000, train_interval=4,
                   delta_clip=1.)

    dqn.compile(Adam(lr=.00025), metrics=['mae'])

    dqn.fit(env, nb_steps=775000, log_interval=10000)
    dqn.save_weights('policy.h5', overwrite=True)
