#!/usr/bin/env python3

import ortools
import pandas as pd
from ortools.linear_solver import pywraplp
from functools import lru_cache
import math
from tqdm import tqdm

def get_element(df:pd.DataFrame, rowname:str, colname:str):
    selector = df['Unnamed: 0'] == rowname
    df = df[selector]
    return df[colname].to_numpy()[0]

def get_sum_of_defined_vars(df):
    return df.sum(axis=1, skipna=True).sum(skipna=True)

def pair_array(arr:list)->list:
    # Given an array [1, 2, 3, 4]
    # Return an array of pairs [(1,2), (2, 3), (3,4)]
    return [(arr[i], arr[i+1],) for i in range(len(arr) - 1)]

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

        self.source = source
        self.destination = destination

        self.solver = pywraplp.Solver(\
                        'LPWrapper',\
                        pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

        self.edge = self.create_edges()

        self.set_source_destination_true()

        self.cannot_retrace_path()

        self.add_condition_for_intermediate_stops()

        # Objective is to minimize the distance between two points
        self.objective = self.solver.Objective()
        self.set_objective_coefficients()
        self.objective.SetMinimization()




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
                if float('nan') == get_element(self.distance_df, st1, st2):
                    var = outer[st1][st2]
                    constraint = self.solver.Constraint(0, 0)
                    constraint.SetCoefficient(var, 1)

        return outer

    def cannot_retrace_path(self):
        # If we take A->B then we cannot take B->A
        # This also takes care of dead ends
        for st1 in self.stop_names:
            for st2 in self.stop_names:
                if st1 != st2:
                    constraint = self.solver.Constraint(0, 1)
                    var_front = self.edge[st1][st2]
                    var_back = self.edge[st2][st1]
                    constraint.SetCoefficient(var_front, 1)
                    constraint.SetCoefficient(var_back, 1)

    def set_source_destination_true(self):
        """ Set the source and destination station as taken """

        # source
        # It should be connected to exactly one station
        constraint = self.solver.Constraint(1, self.solver.infinity())
        for st in self.stop_names:
            if st != self.source:
                var = self.edge[self.source][st]
                constraint.SetCoefficient(var, 1)

        # source
        # It cannot have incoming nodes
        constraint = self.solver.Constraint(0, 0)
        for st in self.stop_names:
            if st != self.source:
                var = self.edge[st][self.source]
                constraint.SetCoefficient(var, 1)

        # destination
        # It should be connected to exactly one station
        constraint = self.solver.Constraint(1, self.solver.infinity())
        for st in self.stop_names:
            if st != self.destination:
                var = self.edge[st][self.destination]
                constraint.SetCoefficient(var, 1)

        # destination
        # It cannot have outgoing nodes
        constraint = self.solver.Constraint(0, 0)
        for st in self.stop_names:
            if st != self.destination:
                var = self.edge[self.destination][st]
                constraint.SetCoefficient(var, 1)


    def set_objective_coefficients(self):
        # Set the coefficients of the objective. For every path that is taken
        # the weight is the same as the distance taken in the excel sheet
        distdf = self.distance_df.copy()
        maxdistance = get_sum_of_defined_vars(distdf)
        maxdistance *= maxdistance
        maxdistance *= maxdistance
        self.replace_nans(distdf, maxdistance)
        for st1 in self.stop_names:
            for st2 in self.stop_names:
                dist = get_element(distdf, st1, st2)
                var = self.edge[st1][st2]
                self.objective.SetCoefficient(var, dist)

    def add_condition_for_intermediate_stops(self):
        # For intermediate stops, add a condition that if there is an
        # outgoing edge, there must be an incoming edge

        def constrain(stopname):
            if stopname in [self.source, self.destination]:
                return

            constraint = self.solver.Constraint(0, 0)
            # Incoming into that node
            for st in self.stop_names:
                if st != stopname:
                    var = self.edge[st][stopname]
                    constraint.SetCoefficient(var, 1)
                    var = self.edge[stopname][st]
                    constraint.SetCoefficient(var, -1)
        
        for stopname in self.stop_names:
            constrain(stopname)

    def get_next_node(self, node):
        # Given a node, return the next node in the path
        for n in self.stop_names:
            if self.edge[node][n].SolutionValue() == 1:
                return n
        assert(False) # Should never reach here

    def get_path(self, source, destination):
        # Returns [nodestart, node1, ..., nodeend]
        current_node = source
        next_node = None
        ret = []

        while next_node != destination:
            if next_node == None:
                ret.append(current_node)
            else:
                current_node = next_node
            next_node = self.get_next_node(current_node)
            ret.append(next_node)

        return ret
            
    def get_shortest_path(self):
        # Returns distance, [nodestart, node1, node2, ..., nodeend]
        self.solver.Solve()
        return self.objective.Value(), \
            self.get_path(self.source, self.destination)

class TrainCapacity(TrainBase):
    def __init__(self, excel_file_name):
        super().__init__(excel_file_name)

        # This is a 2D array which specifies how many people travel
        # between every pair of directly connected stations
        # If stations are not directly connected, this should be zero
        self.capacity_required = self.initialize_capacity_matrix()

        # We will treat upstream and downstream capacity separately
        # This will give some added flexibility rather than just 
        # assuming that trains upstream and downstream are the same
        self.line_capacity_per_train_up = {}
        for line in self.line_names:
            self.line_capacity_per_train_up[line] = \
                self.initialize_capacity_matrix()

        self.line_capacity_per_train_down = {}
        for line in self.line_names:
            self.line_capacity_per_train_down[line] = \
                self.initialize_capacity_matrix()

        for line in self.line_names:
            self.fill_line_capacity_per_train(line)

        #self.add_all_requirements()



    def initialize_capacity_matrix(self):
        outer = {}
        for s1 in self.stop_names:
            inner = {s: 0.0 for s in self.stop_names}
            outer[s1] = inner
        return outer

    def get_line_stations(self, line):
        # Get all the stations in a line in order, as a list
        stations = []
        for s in self.stop_names:
            order = get_element(self.stop_df, s, line)
            if not math.isnan(order):
                stations.append((order, s,))
        stations = sorted(stations)
        stations = [y for (x, y) in stations]
        return stations

    def station_pairs(self, line:str, downstream=False)->list:
        # Given a line with a list of stations [A, B, C, D]
        # Return a set of pairs [(A, B), (B, C), (C, D)]
        stations = self.get_line_stations(line)
        if downstream:
            stations = stations[::-1]
        return pair_array(stations)
    
    def add_requirements(self, source, destination):
        # For a source and destination, find the number of passengers
        # traveling. Calculate the shortest path, and add the same
        # number of passengers to each leg of the route
        if source == destination:
            return

        req = get_element(self.passenger_df, source, destination)
        dist, route = ShortestPath(self.excel_file_name, source, destination)\
                        .get_shortest_path()
        route = pair_array(route)
        for x, y in route:
            self.capacity_required[x][y] += req

    def add_all_requirements(self):
        # Add the requirements for each pair of adjacent stations
        description = "Calculating shortest distance between pairs of stops"
        pairs = [(s1, s2,) for s1 in self.stop_names for s2 in self.stop_names]
        for s1, s2 in tqdm(pairs, desc=description):
            self.add_requirements(s1, s2)

    def fill_line_capacity_per_train(self, line:str):
        capacity = get_element(self.train_capacity_df, line, 'Capacity')
        
        for x, y in self.station_pairs(line):
            self.line_capacity_per_train_up[line][x][y] = capacity
        for x, y in self.station_pairs(line, downstream=True):
            self.line_capacity_per_train_down[line][x][y] = capacity

if "__main__" == __name__:
    #Task3("Assignment_DA_2_Task_3_data.xlsx").main()
    dist, path = ShortestPath("Assignment_DA_2_Task_3_data.xlsx", 'A', 'P')\
        .get_shortest_path()
    t = TrainCapacity('Assignment_DA_2_Task_3_data.xlsx')
    print(stations := t.station_pairs('L1'))

