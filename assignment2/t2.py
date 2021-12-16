#!/usr/bin/env python3

import pandas as pd
import ortools
from ortools.linear_solver import pywraplp
from functools import lru_cache

FULL_TSP = False

def get_element(df:pd.DataFrame, rowname:str, colname:str):
    selector = df['Unnamed: 0'] == rowname
    df = df[selector]
    return df[colname].to_numpy()[0]

class Task2:
    def __init__(self, excel_file_name, start_city_name):
        self.excel_file_name = excel_file_name
        self.start_city_name = start_city_name

        self.distances_df = self.read_excel("Distances")

        self.cities_must_visit = \
            ["Dublin", "Limerick", "Waterford", "Galway", "Wexford", \
            "Belfast", "Athlone", "Rosslare", "Wicklow"]

        if start_city_name not in self.cities_must_visit:
            self.cities_must_visit.append(start_city_name)

        self.solver = pywraplp.Solver(\
                        'LPWrapper',\
                        pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

        # Get all city names
        self.city_names = [str(x) for x in self.distances_df.columns][1:]

        # Map from city to its position in the dataframe
        self.city_numbers = {c: n for n, c in enumerate(self.city_names)}

        self.num_cities = len(self.city_names)

        # Index of the starting city, this will be helpful later
        self.start_city_ind = self.city_numbers[self.start_city_name]

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

        if FULL_TSP:
            self.create_constraint_all_towns_visited_only_once()
        else:
            self.create_constraint_all_towns_visited_max_once()

        self.create_constraints_no_loops_to_same_city()

        self.create_constraint_no_complete_subroutes()

        if not FULL_TSP:
            self.create_constraint_must_visit_cities()


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

    # We use additional variables, each having values from 1 to N
    # if city[k] is the i'th city to be visited, then city[k] = i
    # This will help us to avoid loops
    def create_order_variables(self):
        return [self.solver.IntVar(1, len(self.city_names), f"{n}-{c}") \
            for n, c in enumerate(self.city_names)]

    def create_constraint_all_towns_visited_only_once(self):
        # This function is unused by default, it is useful 
        # in the full TSP version of the problem (off by default)
        for c1 in self.city_names:
            cons1 = self.solver.Constraint(1, 1)
            cons2 = self.solver.Constraint(1, 1)
            for c2 in self.city_names:
                var1 = self.var_edges[c1][c2]
                cons1.SetCoefficient(var1, 1)
                var2 = self.var_edges[c2][c1]
                cons2.SetCoefficient(var2, 1)

    def create_constraint_all_towns_visited_max_once(self):
        for c1 in self.city_names:
            cons1 = self.solver.Constraint(0, 1)
            cons2 = self.solver.Constraint(0, 1)
            for c2 in self.city_names:
                var1 = self.var_edges[c1][c2]
                cons1.SetCoefficient(var1, 1)
                var2 = self.var_edges[c2][c1]
                cons2.SetCoefficient(var2, 1)

        # A town entered must also be exited
        for c1 in self.city_names:
            cons = self.solver.Constraint(0, 0)
            for c2 in self.city_names:
                var1 = self.var_edges[c1][c2]
                cons.SetCoefficient(var1, -1)
                var2 = self.var_edges[c2][c1]
                cons.SetCoefficient(var2, 1)

    def create_constraint_must_visit_cities(self):
        # Every city in the must-visit list must be visited
        for city in self.cities_must_visit:
            cons1 = self.solver.Constraint(1, self.solver.infinity())
            cons2 = self.solver.Constraint(1, self.solver.infinity())
            for c2 in self.city_names:
                if c2 != city:
                    var1 = self.var_edges[city][c2]
                    cons1.SetCoefficient(var1, 1)
                    var2 = self.var_edges[c2][city]
                    cons2.SetCoefficient(var2, 1)

    def create_constraints_no_loops_to_same_city(self):
        # No zero length loops from same city to itself
        for c in self.city_names:
            var = self.var_edges[c][c]
            constraint = self.solver.Constraint(0, 0)
            constraint.SetCoefficient(var, 1)

    def create_constraint_no_complete_subroutes(self):
        # There should be no complete sub-routes.
        # This will be achieved by self.var_order
        # Each city is given a number in the order in which it is visited
        # There cannot be an edge from a city later in the route to a
        # city earlier in the route, except for the first and last city
        
        # order of first city is 1
        constraint = self.solver.Constraint(1, 1)
        constraint.SetCoefficient(self.var_order[self.start_city_ind], 1)

        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != self.start_city_ind and j != self.start_city_ind:
                    cons = self.solver.Constraint(\
                        2 - self.num_cities, self.solver.infinity())
                    c1_name = self.city_names[i]
                    c2_name = self.city_names[j]
                    var_edge = self.var_edges[c1_name][c2_name]
                    cons.SetCoefficient(var_edge, 1 - self.num_cities)
                    cons.SetCoefficient(self.var_order[i], -1)
                    cons.SetCoefficient(self.var_order[j], 1)

    def get_next_city(self, start):
        for c2 in self.city_names:
            if 1 == self.var_edges[start][c2].SolutionValue():
                dist = get_element(self.distances_df, start, c2)
                return c2, dist
        assert(False) # Should not reach here

    def print_route(self):
        next_city = None
        current_city = self.start_city_name
        
        n = 0

        while self.start_city_name != next_city:
            n += 1
            if next_city != None:
                current_city = next_city
            next_city, dist = self.get_next_city(current_city)
            print(f"{n:>3d}. {current_city:>10s} ---> "\
                + f"{next_city:10s} -- {dist:>10d}")

    def set_objective_coefficients(self):
        for c1 in self.city_names:
            for c2 in self.city_names:
                var = self.var_edges[c1][c2]
                dist = get_element(self.distances_df, c1, c2)
                self.objective.SetCoefficient(var, float(dist))

    def solve(self):
        self.solver.Solve()
        self.optimal_distance = self.objective.Value()

    def validate_solution_full_tsp(self):
        visited = [False for city in self.city_names]
        visited[self.start_city_ind] = True
        next_city = None
        current_city = self.start_city_name
        dist_accumulator = 0.0

        while self.start_city_name != next_city:
            if next_city != None:
                current_city = next_city
            next_city, dist = self.get_next_city(current_city)
            dist_accumulator += dist
            next_city_index = self.city_numbers[next_city]
            visited[next_city_index] = True

        assert(all(visited))
        assert(dist_accumulator == self.optimal_distance)

    def validate_solution(self):
        visited = []
        visited.append(self.start_city_name)
        next_city = None
        current_city = self.start_city_name
        dist_accumulator = 0.0

        while self.start_city_name != next_city:
            if next_city != None:
                current_city = next_city
            next_city, dist = self.get_next_city(current_city)
            dist_accumulator += dist
            next_city_index = self.city_numbers[next_city]
            if next_city != self.start_city_name:
                visited.append(next_city)

        assert(len(visited) == len(self.cities_must_visit))
        assert(dist_accumulator == self.optimal_distance)
        for city in self.cities_must_visit:
            assert(city in visited)

    def print_all_variables(self):
        for c1 in self.city_names:
            for c2 in self.city_names:
                val = self.var_edges[c1][c2].SolutionValue() 
                if val != 0:
                    print(f"edge({c1}, {c2}) = {val}")

    def main(self):
        self.solve()
        # self.print_all_variables()
        if FULL_TSP:
            self.validate_solution_full_tsp()
        else:
            self.validate_solution()
        self.print_route()
        print()
        print(f"Optimal distance: {self.optimal_distance}")


if "__main__" == __name__:
    Task2("./Assignment_DA_2_Task_2_data.xlsx", "Cork").main()
