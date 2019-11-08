#! /usr/bin/env python3
# coding: utf-8

"""Module containing the show_best_path() function"""

import random as rd

import numpy as np

def next_best_state(q_matrix, current_state):
    """Returns the next best step, according to the q_matrix"""
    max_value = max(q_matrix[current_state, action]
                    for action in range(q_matrix[0].size))
    best_actions = [action for action in range(q_matrix[0].size)
                    if q_matrix[current_state, action] == max_value]
    return rd.choice(best_actions)

def show_best_path(initial_state, q_matrix):
    """
    Shows to the user the best way to the goal, starting at an initial state
    """
    matrix_size = q_matrix[0].size
    round_arrays = np.vectorize(lambda x: round(x, 3))
    q_matrix = round_arrays(q_matrix)
    current_state = initial_state
    path = []
    while not np.array_equal(q_matrix[current_state],
                             np.zeros((1, matrix_size))[0]):
        path.append(str(current_state))
        current_state = next_best_state(q_matrix, current_state)
    path.append(str(current_state)) #Add last (goal) state
    print("Best path starting at {} :".format(initial_state), "->".join(path))
