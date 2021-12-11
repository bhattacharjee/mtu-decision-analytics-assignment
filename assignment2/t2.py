#!/usr/bin/env python3

import pandas as pd
import ortools
from ortools.linear_solver import pywraplp
from functools import lru_cache

def get_element(df:pd.DataFrame, rowname:str, colname:str):
    selector = df['Unnamed: 0'] == rowname
    df = df[selector]
    return df[colname].to_numpy()[0]

class Task2:
    def __init__(self, excel_file_name):
        self.excel_file_name = excel_file_name

        self.distances_df = self.read_excel("Distances")

        self.solver = pywraplp.Solver(\
                        'LPWrapper',\
                        pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

        # Get all city names
        self.city_names = [str(x) for x in self.distances_df.columns][1:]

        # Map from city to its position in the dataframe
        self.city_numbers = {c: n for n, c in enumerate(self.city_names)}

        # Create a 2-D array of variables. 
        # if x[i][j] = True then there is an ege from city[i] to city[j]
        self.var_edges = self.create_edges()

        # We use additional variables, each having values from 1 to N
        # if city[k] is the i'th city to be visited, then city[k] = i
        # This will help us to avoid loops
        self.var_order = self.create_order_variables()

        self.objective = self.solver.Objective()

        self.optimal_distance = None

        self.objective.SetMinimization()

        self.set_objective_coefficients()

        self.create_constraint_all_towns_visited_only_once()

        self.create_constraints_no_loops_to_same_city()


    def read_excel(self, sheet_name:str):
        df = pd.read_excel(self.excel_file_name, sheet_name=sheet_name)
        return df

    def create_edges(self):
        outer = {}
        for c1 in self.city_names:
            inner = {}
            for c2 in self.city_names:
                varname = f"{c1:>10s} ---> {c2:10s}"
                inner[c2] = self.solver.IntVar(0, 1, varname)
            outer[c1] = inner
        return outer

    def create_order_variables(self):
        pass


    def create_constraint_all_towns_visited_only_once(self):
        for c1 in self.city_names:
            cons1 = self.solver.Constraint(1, 1)
            cons2 = self.solver.Constraint(1, 1)
            for c2 in self.city_names:
                var1 = self.var_edges[c1][c2]
                cons1.SetCoefficient(var1, 1)
                var2 = self.var_edges[c2][c1]
                cons2.SetCoefficient(var2, 1)

    def create_constraints_no_loops_to_same_city(self):
        # No zero length loops from same city to itself
        for c in self.city_names:
            var = self.var_edges[c][c]
            constraint = self.solver.Constraint(0, 0)
            constraint.SetCoefficient(var, 1)

    def print_route(self):
        def get_next_city(start):
            for c2 in self.city_names:
                if 1 == self.var_edges[start][c2].SolutionValue():
                    dist = get_element(self.distances_df, start, c2)
                    return c2, dist
            assert(False) # Should not reach here

        next_city = None
        current_city = "Cork"

        while "Cork" != next_city:
            if next_city != None:
                current_city = next_city
            next_city, dist = get_next_city(current_city)
            print(f"{current_city:>10s} ---> {next_city:10s} -- {dist:>10d}")

    def set_objective_coefficients(self):
        for c1 in self.city_names:
            for c2 in self.city_names:
                var = self.var_edges[c1][c2]
                dist = get_element(self.distances_df, c1, c2)
                self.objective.SetCoefficient(var, float(dist))

    def solve(self):
        self.solver.Solve()
        self.optimal_distance = self.objective.Value()


    def main(self):
        print(self.distances_df)
        self.solve()
        self.print_route()
        print(f"Optimal distance: {self.optimal_distance}")


if "__main__" == __name__:
    Task2("./Assignment_DA_2_Task_2_data.xlsx").main()
