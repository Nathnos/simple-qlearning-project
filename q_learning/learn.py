#! /usr/bin/env python3
# coding: utf-8
"""
Module for the learn() function
"""

import random as rd

import numpy as np

def choose_random_action(reward_matrix, current_state):
    """Selects a random action"""
    available_actions = [action for action in range(reward_matrix[0].size)
                         if reward_matrix[current_state, action] != -1]
    return rd.choice(available_actions)

def get_max_q_matrix_value(reward_matrix, q_matrix, current_state):
    """Gets the best value for q_matrix"""
    available_actions = [action for action in range(q_matrix[0].size)
                         if reward_matrix[current_state, action] != -1]
    max_value = max(q_matrix[current_state, action]
                    for action in available_actions)
    return max_value

def learn(reward_matrix, number_of_generations,
          discount_factor, learning_rate):
    """
    The Q-Learning process itself !
    The more episodes, the more accurate outcome
    Agent starts in a random room
    The discount_factor determines the importance of future rewards
    The discount_factor should be 1 for fully deterministic environments
    The discount_factor should be around 0.1 for stochastic problems
    """
    q_matrix = np.zeros(reward_matrix.shape)
    for _ in range(number_of_generations):
        #We select a random initial state
        current_state = rd.randrange(reward_matrix[0].size)
        while True:
            next_action = choose_random_action(reward_matrix, current_state)
            if next_action == current_state:
                break
            old_info = q_matrix[current_state, next_action]
            new_info = (reward_matrix[current_state, next_action] +
                        discount_factor * get_max_q_matrix_value(
                            reward_matrix, q_matrix, next_action))
            q_matrix[current_state, next_action] = (
                learning_rate*new_info + (1-learning_rate)*old_info)
            current_state = next_action
    return q_matrix
