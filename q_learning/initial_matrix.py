#! /usr/bin/env python3
# coding: utf-8

"""Module for initial_matrix() function"""

import numpy as np

def initial_matrix():
    #Ajouter restart=False pour enregister et charger l'apprentissage
    """
    Return the matrix of the initial state
    Lines are current state and Columns the actions
    -1 means forbidden ; 0 means no inherent interest ; 100 means goal
    """
    reward_matrix = matrix_bay()
    return reward_matrix


def matrix_bay(matrix_name=None):
    """Many matrixs, for testing or problem solving"""
    #Once goal is reached, no turing back
    if matrix_name is None:
        reward_matrix = np.array([[-1, -1, -1, -1, 0, -1],
                                  [-1, -1, -1, 0, -1, 100],
                                  [-1, -1, -1, 0, -1, -1],
                                  [-1, 0, 0, -1, 0, -1],
                                  [0, -1, -1, 0, -1, 100],
                                  [-1, -1, -1, -1, -1, 100]])
    elif matrix_name == "test2":
        reward_matrix = (
            np.array([[-1, 0, -1, -1, -1, -1, -1, -1, -1, -1],
                      [0, -1, -1, -1, 0, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, 0, -1, -1, -1, 100],
                      [0, -1, -1, -1, 0, -1, 0, -1, -1, -1],
                      [-1, 0, -1, 0, -1, 0, -1, -1, -1, -1],
                      [-1, -1, 0, -1, 0, -1, -1, -1, 0, 100],
                      [-1, -1, -1, 0, -1, -1, -1, 0, -1, 100],
                      [-1, -1, -1, -1, 0, -1, 0, -1, -1, 100],
                      [-1, -1, -1, -1, -1, 0, -1, -1, -1, 100],
                      [-1, -1, -1, -1, -1, -1, -1, -1, -1, 100]]))
    return reward_matrix
