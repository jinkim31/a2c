import tensorflow
import tensorflow as tf
import keras.models
import keras.layers
import keras.optimizers
import numpy as np
import matplotlib as plt
import gym


class Actor(keras.Model):

    def __init__(self, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_bound = action_bound

        self.h1 = keras.layers.Dense(64, activation='relu')
        self.h2 = keras.layers.Dense(32, activation='relu')
        self.h3 = keras.layers.Dense(16, activation='relu')
        self.mu = keras.layers.Dense(action_dim, activation='tanh')
        self.std = keras.layers.Dense(action_dim, activation='softplus')

    def call(self, state):
        x = self.h1(state)
        x = self.h2(x)
        x = self.h3(x)
        mu = self.mu(x)
        std = self.std(x)

        # bound mu. tanh makes mu to be a vector of values between -1 and 1. simply multiplying by action_bound returns
        # bound value
        mu = keras.layers.Lambda(lambda x: x * self.action_bound)(mu)

        return [mu, std]


class Critic(keras.Model):
    def __init__(self):
        super(Critic, self).__init__()

        self.h1 = keras.layers.Dense(64, activation='relu')
        self.h2 = keras.layers.Dense(32, activation='relu')
        self.h3 = keras.layers.Dense(16, activation='relu')
        self.v = keras.layers.Dense(1, activation='linear')

    def call(self, state):
        x = self.h1(state)
        x = self.h2(x)
        x = self.h3(x)
        v = self.v(x)
        return v


class A2CAgent:
    def __init__(self, env):
        # hyperparameters
        self.gamma = 0.95
        self.BATCH_SIZE = 32
        self.LEARNING_RATE_ACTOR = 0.001
        self.LEARNING_RATE_CRITIC = 0.01

        # env
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]
        self.std_bound = [1e-2, 1.0]

        # actor & critic network
        self.actor = Actor(self.action_dim, self.action_bound)
        self.actor.build(input_shape=(None, self.state_dim))
        self.critic = Critic()
        self.critic.build(input_shape=(None, self.state_dim))
        self.actor.summary()
        self.critic.summary()

        # optimizers
        self.opt_actor = keras.optimizers.adam_v2.Adam(self.LEARNING_RATE_ACTOR)
        self.opt_critic = keras.optimizers.adam_v2.Adam(self.LEARNING_RATE_CRITIC)

        # history
        self.history_reward = []



if __name__ == '__main__':
    env = gym.make('Pendulum-v1')
    A2CAgent(env)
