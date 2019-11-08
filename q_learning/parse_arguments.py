#! /usr/bin/env python3
# coding: utf-8

"""Module for parse_arguments() function"""

import argparse

def parse_arguments():
    """Denfines and returns arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--generations",
                        help="""Defines the number of generations""")
    parser.add_argument("-df", "--discount_factor",
                        help="""Defines the discount_factor""")
    parser.add_argument("-lr", "--learing_rate",
                        help="""Defines the learing_rate""")
    parser.add_argument("-bp", "--best_path",
                        help="""Show the best path for a starting state""")
    return parser.parse_args()
