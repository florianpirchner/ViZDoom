#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from vizdoom import *
import itertools as it
from random import sample, randint, random
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
import tensorflow as tf
from tqdm import trange
import cv2

# Q-learning settings
learning_rate = 0.00025
# learning_rate = 0.0001
discount_factor = 0.99
epochs = 20
startEpoch = 10
learning_steps_per_epoch = 2000
replay_memory_size = 10000

# NN learning settings
batch_size = 64

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 12
resolution = (60, 90)

episodes_to_watch = 10

model_savefile = "/Users/florianpirchner/Work/dev/tensorflow/git/ViZDoom/examples/python/savedModels_4frames/model.ckpt"
save_model = True
load_model = True
skip_learning = True
trainConvNets = False
# Configuration file path
#config_file_path = "../../scenarios/simpler_basic.cfg"
config_file_path = "../../scenarios/defend_the_line.cfg"

global imageStack
imageStack = np.stack((np.zeros(resolution), np.zeros(resolution), np.zeros(resolution), np.zeros(resolution)), axis=2)


# config_file_path = "../../scenarios/rocket_basic.cfg"
# config_file_path = "../../scenarios/basic.cfg"

# Converts and down-samples the input image
def preprocess(img):
    global imageStack
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    img = np.reshape(img, list(resolution) + [1])

    imageStack = np.append(img, imageStack[:, :, :3], axis=2)
    return imageStack

class ReplayMemory:
    def __init__(self, capacity):
        channels = 4
        state_shape = (capacity, resolution[0], resolution[1], channels)
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        self.s1[self.pos, :, :, :] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, :, :, :] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]

def create_network(session, available_actions_count):
    # Create the input variables
    ###s1_ = tf.placeholder(tf.float32, [None] + list(resolution) + [1], name="State")
    s1_ = tf.placeholder("float", [None] + list(resolution) + [4], name="State")
    a_ = tf.placeholder(tf.int32, [None], name="Action")
    target_q_ = tf.placeholder(tf.float32, [None, available_actions_count], name="TargetQ")

    # Add 2 convolutional layers with ReLu activation
    conv1 = tf.contrib.layers.convolution2d(s1_, num_outputs=32, kernel_size=[3, 3], stride=[2, 2],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1), trainable = trainConvNets)
    print("conv1 Shape ", conv1.shape)

    maxPool1 = tf.contrib.layers.max_pool2d(conv1, kernel_size=(2,2))
    print("maxPool1 Shape ", maxPool1.shape)

    
    conv2 = tf.contrib.layers.convolution2d(maxPool1, num_outputs=64, kernel_size=[3, 3], stride=[1, 1],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1), trainable = trainConvNets)
    print("conv2 Shape ", conv2.shape)
    
    conv3 = tf.contrib.layers.convolution2d(conv1, num_outputs=64, kernel_size=[2, 2], stride=[2, 2],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1), trainable = trainConvNets)
    print("conv3 Shape ", conv3.shape)
    
    conv3_flat = tf.contrib.layers.flatten(conv3)
    print("conv3_flat Shape ", conv3_flat.shape)

    fc1 = tf.contrib.layers.fully_connected(conv3_flat, num_outputs=128, activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            biases_initializer=tf.constant_initializer(0.1))
    print("fc1 Shape ", fc1.shape)

    q = tf.contrib.layers.fully_connected(fc1, num_outputs=available_actions_count, activation_fn=None,
                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                          biases_initializer=tf.constant_initializer(0.1))
    print("q Shape ", q.shape)

    best_a = tf.argmax(q, 1)

    loss = tf.losses.mean_squared_error(q, target_q_)

    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    # Update the parameters according to the computed gradient using RMSProp.
    train_step = optimizer.minimize(loss)

    def function_learn(s1, target_q):
        print("Run back propagation")
        feed_dict = {s1_: s1, target_q_: target_q}
        l, _ = session.run([loss, train_step], feed_dict=feed_dict)
        return l

    def function_get_q_values(state):
        return session.run(q, feed_dict={s1_: state})

    def function_get_best_action(state):
        return session.run(best_a, feed_dict={s1_: state})

    def function_simple_get_best_action(state):
        return function_get_best_action(state.reshape([1, resolution[0], resolution[1], 4]))[0]
        #return function_get_best_action(state)

    return function_learn, function_get_q_values, function_simple_get_best_action


def learn_from_memory():
    """ Learns from a single transition (making use of replay memory).
    s2 is ignored if s2_isterminal """

    # Get a random minibatch from the replay memory and learns from it.
    ##print("memory size", memory.size)
    if memory.size > batch_size:
        s1, a, s2, isterminal, r = memory.get_sample(batch_size)

        q2 = np.max(get_q_values(s2), axis=1)
        ##print("next q2", q2)
        target_q = get_q_values(s1)
        ##print("target_q", target_q)
        # target differs from q only for the selected action. The following means:
        # target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r
        target_q[np.arange(target_q.shape[0]), a] = r + discount_factor * (1 - isterminal) * q2
        ##print("target_q reshape", target_q)
        learn(s1, target_q)


def perform_learning_step(epoch):
    """ Makes an action according to eps-greedy policy, observes the result
    (next state, reward) and learns from the transition"""

    def exploration_rate(epoch):
        """# Define exploration rate change over time"""
        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = 0.1 * epochs  # 10% of learning time
        eps_decay_epochs = 0.6 * epochs  # 60% of learning time

        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            # Linear decay
            return start_eps - (epoch - const_eps_epochs) / \
                               (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return end_eps

    s1 = preprocess(game.get_state().screen_buffer)
    #cv2.imshow('s1', s1[:, :, 0])
    #cv2.imshow('s2', s1[:, :, 1])
    #cv2.imshow('s3', s1[:, :, 2])
    #cv2.imshow('s4', s1[:, :, 3])
    #cv2.waitKey(1)
    
    
    # With probability eps make a random action.
    eps = exploration_rate(epoch)
    if random() <= eps:
        a = randint(0, len(actions) - 1)
        print("best random action ", a)
    else:
        # Choose the best action according to the network.
        a = get_best_action(s1)
        print("best action ", a)
    reward = game.make_action(actions[a], frame_repeat)
    print("reward", reward)

    isterminal = game.is_episode_finished()
    s2 = preprocess(game.get_state().screen_buffer) if not isterminal else None

    # Remember the transition that was just experienced.
    memory.add_transition(s1, a, s2, isterminal, reward)

    learn_from_memory()


# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    if(skip_learning):
        game.set_window_visible(True)
        game.set_mode(Mode.ASYNC_PLAYER)
    else:
        game.set_window_visible(False)
        game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    # Enables labeling of in game objects labeling.
    game.set_labels_buffer_enabled(False)

    game.init()
    print("Doom initialized.")
    return game


if __name__ == '__main__':

    # Create Doom instance
    game = initialize_vizdoom(config_file_path)

    # Action = which buttons are pressed
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Create replay memory which will store the transitions
    memory = ReplayMemory(capacity=replay_memory_size)

    session = tf.Session()
    learn, get_q_values, get_best_action = create_network(session, len(actions))
    saver = tf.train.Saver()
    if load_model:
        print("Loading model from: ", model_savefile)
        saver.restore(session, model_savefile)
    else:
        init = tf.global_variables_initializer()
        session.run(init)
    print("Starting the training!")

    time_start = time()
    if not skip_learning:
        for epoch in range(epochs):
            epoch = epoch + startEpoch
            print("\nEpoch %d\n-------" % (epoch + 1))
            train_episodes_finished = 0
            train_scores = []

            print("Training...")
            game.new_episode()
            for learning_step in trange(learning_steps_per_epoch, leave=False):
                perform_learning_step(epoch)
                if game.is_episode_finished():
                    score = game.get_total_reward()
                    train_scores.append(score)
                    game.new_episode()
                    train_episodes_finished += 1

            print("%d training episodes played." % train_episodes_finished)

            train_scores = np.array(train_scores)

            print("Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()), \
                  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())


            print("Saving the network weigths to:", model_savefile)
            saver.save(session, model_savefile)


            print("\nTesting...")
            test_episode = []
            test_scores = []
            for test_episode in trange(test_episodes_per_epoch, leave=False):
                game.new_episode()
                while not game.is_episode_finished():
                    state = preprocess(game.get_state().screen_buffer)
                    best_action_index = get_best_action(state)

                    game.make_action(actions[best_action_index], frame_repeat)
                r = game.get_total_reward()
                test_scores.append(r)

            test_scores = np.array(test_scores)
            print("Results: mean: %.1f±%.1f," % (
                test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
                  "max: %.1f" % test_scores.max())

            

            print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

    game.close()
    print("======================================")
    print("Training finished. It's time to watch!")

    # Reinitialize the game with window visible
    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()

    for _ in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            best_action_index = get_best_action(state)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)
