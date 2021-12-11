#!/usr/bin/env python3

import ortools
import pandas as pd
from ortools.linear_solver import pywraplp
from functools import lru_cache

def get_element(df:pd.DataFrame, rowname:str, colname:str):
    selector = df['Unnamed: 0'] == rowname
    df = df[selector]
    return df[colname].to_numpy()[0]

def get_sum_of_defined_vars(df):
    return df.sum(axis=1, skipna=True).sum(skipna=True)

class TrainBase:
    def __init__(self, excel_file_name):
        self.excel_file_name = excel_file_name

        self.stop_df = self.read_csv("Stops")
        self.distance_df = self.read_csv("Distances")
        self.passenger_df = self.read_csv("Passengers")
        self.train_capacity_df = self.read_csv("Trains")


        self.line_names = [str(l) for l in self.stop_df.columns][1:]
        self.stop_names = [str(s) for s in self.distance_df.columns][1:]


    def read_csv(self, sheet_name:str)->pd.DataFrame:
        df = pd.read_excel(self.excel_file_name, sheet_name=sheet_name)
        return df

    def replace_nans(self, df:pd.DataFrame, newValue:int):
        df.replace(float('nan'), float(newValue), inplace=True)

class ShortestPath(TrainBase):
    def __init__(self, excel_file_name, source, destination):
        super().__init__(excel_file_name)

        self.solver = pywraplp.Solver(\
                        'LPWrapper',\
                        pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

        self.create_edges()


    def create_edges(self):
        """ Decision variables if an edge from x to y is taken
            Two additional housekeeping things are done using constraints here:
            - An edge from a node to itself is always zero
            - If an edge between two nodes doesn't exist, it is always zero
        """
        outer = {}
        for st1 in self.stop_names:
            inner = {}
            for st2 in self.stop_names:
                name = f"edgetaken({st1},{st2})"
                inner[st2] = self.solver.IntVar(0, 1, name)
            outer[st1] = inner

        # Create a constraint that an edge cannot be taken from a stop
        # to itself
        for st in self.stop_names:
            var = outer[st][st]
            constraint = self.solver.Constraint(0, 0)
            constraint.SetCoefficient(var, 1)

        for st1 in self.stop_names:
            for st2 in self.stop_names:
                if float('nan') == get_element(self.distance_df, st1, st2)
                    var = outer[st1][st2]
                    constraint = self.solver.Constraint(0, 0)
                    constraint.SetCoefficient(var, 1)

        return outer


if "__main__" == __name__:
    #Task3("Assignment_DA_2_Task_3_data.xlsx").main()
    ShortestPath("Assignment_DA_2_Task_3_data.xlsx", None, None)

    pass
