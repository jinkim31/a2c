import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Lambda
from keras.optimizers import adam_v2

import numpy as np
import matplotlib.pyplot as plt


# actor network
class Actor(Model):

    def __init__(self, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_bound = action_bound

        self.h1 = Dense(64, activation='relu')
        self.h2 = Dense(32, activation='relu')
        self.h3 = Dense(16, activation='relu')
        self.mu = Dense(action_dim, activation='tanh')
        self.std = Dense(action_dim, activation='softplus')

    def call(self, state):
        x = self.h1(state)
        x = self.h2(x)
        x = self.h3(x)
        mu = self.mu(x)
        std = self.std(x)

        # Scale output to [-action_bound, action_bound]
        mu = Lambda(lambda x: x * self.action_bound)(mu)

        return [mu, std]


# critic network
class Critic(Model):
    def __init__(self):
        super(Critic, self).__init__()

        self.h1 = Dense(64, activation='relu')
        self.h2 = Dense(32, activation='relu')
        self.h3 = Dense(16, activation='relu')
        self.v = Dense(1, activation='linear')

    def call(self, state):
        x = self.h1(state)
        x = self.h2(x)
        x = self.h3(x)
        v = self.v(x)
        return v


class A2Cagent(object):

    def __init__(self, env):

        # hyperparameters
        self.GAMMA = 0.95
        self.BATCH_SIZE = 32
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001

        # environment
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]
        self.std_bound = [1e-2, 1.0]

        # networks
        self.actor = Actor(self.action_dim, self.action_bound)
        self.critic = Critic()
        self.actor.build(input_shape=(None, self.state_dim))
        self.critic.build(input_shape=(None, self.state_dim))
        self.actor.summary()
        self.critic.summary()

        # optimizers
        self.actor_opt = adam_v2.Adam(self.ACTOR_LEARNING_RATE)
        self.critic_opt = adam_v2.Adam(self.CRITIC_LEARNING_RATE)

        # save the results
        self.save_epi_reward = []

    # log of gaussian probability density function
    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / var - 0.5 * tf.math.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def get_action(self, state):
        mu_a, std_a = self.actor(state)
        mu_a = mu_a.numpy()[0]  # unpack outer []
        std_a = std_a.numpy()[0]
        std_a = np.clip(std_a, self.std_bound[0], self.std_bound[1])
        # sample action
        action = np.random.normal(mu_a, std_a, size=self.action_dim)
        return action

    ## train the actor network
    def actor_learn(self, states, actions, advantages):

        with tf.GradientTape() as tape:
            # policy pdf
            mu_a, std_a = self.actor(states, training=True)
            log_policy_pdf = self.log_pdf(mu_a, std_a, actions)

            # loss function and its gradients
            loss_policy = log_policy_pdf * advantages
            loss = tf.reduce_sum(-loss_policy)

        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))

    ## single gradient update on a single batch data
    def critic_learn(self, states, td_targets):
        with tf.GradientTape() as tape:
            td_hat = self.critic(states, training=True)
            loss = tf.reduce_mean(tf.square(td_targets - td_hat))

        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))

    ## computing targets: y_k = r_k + gamma*V(x_k+1)
    def td_target(self, rewards, next_v_values, dones):
        td_targets = np.zeros(next_v_values.shape)
        for i in range(next_v_values.shape[0]):  # number of batch
            if dones[i]:
                td_targets[i] = rewards[i]
            else:
                td_targets[i] = rewards[i] + self.GAMMA * next_v_values[i]
        return td_targets

    ## load actor wieghts
    def load_weights(self, path):
        self.actor.load_weights(path + 'pendulum_actor.h5')
        self.critic.load_weights(path + 'pendulum_critic.h5')

    ## convert (list of np.array) to np.array
    def unpack_batch(self, batch):
        unpack = batch[0]
        for idx in range(len(batch) - 1):
            unpack = np.append(unpack, batch[idx + 1], axis=0)
        return unpack

    def train(self, max_episode_num):

        for ep in range(int(max_episode_num)):

            # batches. use mutable array for efficiency. use unpack() to convert back to immutable tf tensor.
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = [], [], [], [], []

            time, episode_reward, done = 0, 0, False
            state = self.env.reset()

            while not done:
                # self.env.render()

                # first layer of actor(dense) requires 2 dim tensor at least. add a rank on state with []
                action = self.get_action(tf.convert_to_tensor([state], dtype=tf.float32))

                # action is sampled value from gaussian(mu, std)
                action = np.clip(action, -self.action_bound, self.action_bound)

                # observe
                next_state, reward, done, _ = self.env.step(action)

                # normalize reward to be within [0, 1]
                normalized_reward = (reward + 8) / 8

                # append to the batches(mutable arrays for efficiency)
                # batches can be converted back to immutable tf tensors using unpack()
                # [1 2 3].append([4 5 6]) returns [1 2 3 4 5 6] which eliminates borders between states, actions, etc.
                # use [] to address the problem
                batch_state.append([state])
                batch_action.append([action])
                batch_reward.append([normalized_reward])
                batch_next_state.append([next_state])
                batch_done.append([done])

                # check if batches are full
                if len(batch_state) < self.BATCH_SIZE:
                    # batch not full yet
                    state = next_state
                    episode_reward += reward
                    time += 1
                    continue

                # batches full. not continued.
                # unpack batches
                states = self.unpack_batch(batch_state)
                actions = self.unpack_batch(batch_action)
                train_rewards = self.unpack_batch(batch_reward)
                next_states = self.unpack_batch(batch_next_state)
                dones = self.unpack_batch(batch_done)

                # clear batches
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = [], [], [], [], []

                # get next state value of each transition
                next_v_values = self.critic(tf.convert_to_tensor(next_states, dtype=tf.float32))

                # get td target of each transition
                td_targets = self.td_target(train_rewards, next_v_values.numpy(), dones)

                # train critic
                self.critic_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                  tf.convert_to_tensor(td_targets, dtype=tf.float32))

                # compute advantages
                v_values = self.critic(tf.convert_to_tensor(states, dtype=tf.float32))
                next_v_values = self.critic(tf.convert_to_tensor(next_states, dtype=tf.float32))
                advantages = train_rewards + self.GAMMA * next_v_values - v_values

                # train actor
                self.actor_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                 tf.convert_to_tensor(actions, dtype=tf.float32),
                                 tf.convert_to_tensor(advantages, dtype=tf.float32))

                # update current state
                state = next_state
                episode_reward += reward
                time += 1

            ## display rewards every episode
            print('Episode: ', ep + 1, 'Time: ', time, 'Reward: ', episode_reward)

            self.save_epi_reward.append(episode_reward)

            ## save weights every episode
            if ep % 10 == 0:
                self.actor.save_weights("./save_weights/pendulum_actor.h5")
                self.critic.save_weights("./save_weights/pendulum_critic.h5")

        np.savetxt('./save_weights/pendulum_epi_reward.txt', self.save_epi_reward)
        print(self.save_epi_reward)

    ## save them to file if done
    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()
