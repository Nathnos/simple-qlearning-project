#! /usr/bin/env python3
# coding: utf-8

"""
Tests for Q-Learning, based on this website :
 http://mnemstudio.org/path-finding-q-learning-tutorial.htm
"""

import logging as lg
import numpy as np

from q_learning.learn import learn
from q_learning.learn_double import learn_double
from q_learning.initial_matrix import initial_matrix
from q_learning.init_arguments import init_arguments
from q_learning.show_best_path import show_best_path

lg.basicConfig(level=lg.DEBUG)

def main():
    """Initialises matrix, and starts the learning process"""
    (number_of_generations, discount_factor,
     learning_rate, initial_state) = init_arguments()
    q_matrix = learn_double(initial_matrix(), number_of_generations,
                            discount_factor, learning_rate)
    np.set_printoptions(precision=1)
    lg.info("Matrix after %d generations :\n%s",
            number_of_generations, q_matrix)
    if initial_state != -1:
        show_best_path(initial_state, q_matrix)

if __name__ == "__main__":
    main()
