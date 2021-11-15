#!/usr/bin/env python3

from tiramisu2 import *
from projects import *
from sudoku import *

import argparse

def main():
    parser = argparse.ArgumentParser("Decision Analytics Assignment 1")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(                                                     \
        '-s',                                                               \
        '--sudoku',                                                         \
        help='Solve the sudoku for the given problem',                      \
        action='store_true')
    group.add_argument(                                                     \
        '-t',                                                               \
        '--tiramisu',                                                       \
        help='Solve the tiramisu problem as specified',                     \
        action='store_true')
    group.add_argument(                                                     \
        '-p',                                                               \
        '--projects',                                                       \
        help='Solve the project planning problem, ' +                       \
                'assumes excel file in the same directory',                 \
        action='store_true')
    args = parser.parse_args()
    if args.sudoku:
        sudoku_main()
    elif args.projects:
        projects_main()
    elif args.tiramisu:
        tiramisu_main()
    else:
        assert(False)

if "__main__" == __name__:
    main()
