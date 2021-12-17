#!/usr/bin/env python3
import argparse
import sys

from t1 import *
from t2 import *
from t3 import *

def main():
    parser = argparse.ArgumentParser("Assignment 2: Decision Analytics")
    parser.add_argument(\
                "-t",
                "--task",
                type=int,
                choices=[1, 2, 3])
    args = parser.parse_args()

    if args.task == 1:
        Task1("./Assignment_DA_2_Task_1_data.xlsx").main()
    elif args.task == 2:
        Task2("./Assignment_DA_2_Task_2_data.xlsx", "Cork").main()
    elif args.task == 3:
        TrainCapacity('Assignment_DA_2_Task_3_data.xlsx').main()
    else:
        print(f"{args.task} is not a valid value for task")

if "__main__" == __name__:
    main()
