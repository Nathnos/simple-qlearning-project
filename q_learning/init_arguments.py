#! /usr/bin/env python3
# coding: utf-8

"""Module for init_arguments() function"""

import logging as lg

from q_learning.parse_arguments import parse_arguments

def init_arguments(number_of_generations=700,
                   discount_factor=0.7, learing_rate=0.1):
    """Initialise the q_learning values : how many generations, or
    episodes, the discount factor and the learing rate
    Take parsed argument, or used default"""
    args = parse_arguments()
    if args.generations is not None:
        try:
            number_of_generations = int(args.generations)
        except ValueError:
            lg.error("You didn't entered a proper number of generations. " \
                     "Using default number (%d)", number_of_generations)
    if args.discount_factor is not None:
        try:
            discount_factor = float(args.discount_factor)
        except ValueError:
            lg.error("You didn't entered a proper discount factor. " \
                     "Using default value (%d)", discount_factor)
    if args.learing_rate is not None:
        try:
            learing_rate = float(args.learing_rate)
        except ValueError:
            lg.error("You didn't entered a proper learing rate. " \
                     "Using default vaule (%d)", learing_rate)
    if args.best_path is not None:
        try:
            show_best_path = int(args.best_path)
        except ValueError:
            lg.error("You didn't entered a proper initial state for " \
                     "the best_path argument.")
            show_best_path = -1
    else:
        show_best_path = -1
    return (number_of_generations, discount_factor,
            learing_rate, show_best_path)
