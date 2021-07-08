#!/usr/bin/env python3
import gym
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
import tensorflow.keras as K
from keras.models import load_model

create_q_model = __import__('train').create_q_model
AtariProcessor = __import__('train').AtariProcessor


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
                   processor=processor, nb_steps_warmup=50000,
                   gamma=.99, target_model_update=10000,
                   train_interval=4, delta_clip=1.)

    dqn.compile(Adam(lr=.00025), metrics=['mae'])

    dqn.load_weights('policy.h5')
    dqn.test(env, nb_episodes=10, visualize=False)
