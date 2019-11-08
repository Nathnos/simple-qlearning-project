#! /usr/bin/env python3
# coding: utf-8

"""
Function for double q_learning
"""

import random as rd

import numpy as np

def choose_random_action(reward_matrix, current_state):
    """Selects a random action"""
    available_actions = [action for action in range(reward_matrix[0].size)
                         if reward_matrix[current_state, action] != -1]
    return rd.choice(available_actions)

def next_best_action(reward_matrix, q_matrix, current_state):
    """Selects the best action from the q_matrix"""
    available_actions = [action for action in range(q_matrix[0].size)
                         if reward_matrix[current_state, action] != -1]
    max_value = max(q_matrix[current_state, action]
                    for action in available_actions)
    best_actions = [action for action in available_actions
                    if q_matrix[current_state, action] == max_value]
    return rd.choice(best_actions)

def learn_double(reward_matrix, number_of_generations,
                 discount_factor, learning_rate):
    """
    The double q_learning process : similar the as the simple, but
    there are 2 q_matrix, hepling to avoid overfitting
    Works better on noised environments
    """
    q_matrix_a = np.zeros(reward_matrix.shape)
    q_matrix_b = np.zeros(reward_matrix.shape)
    for _ in range(number_of_generations):
        #We select a random initial state
        current_state = rd.randrange(reward_matrix[0].size)
        while True:
            next_action = choose_random_action(reward_matrix, current_state)
            if next_action == current_state:
                break
            old_state_a = q_matrix_a[current_state, next_action]
            old_state_b = q_matrix_b[current_state, next_action]
            reward = reward_matrix[current_state, next_action]
            next_best_value_b = q_matrix_b[next_action, next_best_action(
                reward_matrix, q_matrix_a, next_action)]
            next_best_value_a = q_matrix_a[next_action, next_best_action(
                reward_matrix, q_matrix_b, next_action)]
            q_matrix_a[current_state, next_action] = (
                old_state_a + learning_rate * (
                    reward + discount_factor * next_best_value_b
                        - old_state_a))
            q_matrix_b[current_state, next_action] = (
                old_state_b + learning_rate * (
                    reward + discount_factor * next_best_value_a
                        - old_state_b))
            current_state = next_action
    return q_matrix_a
