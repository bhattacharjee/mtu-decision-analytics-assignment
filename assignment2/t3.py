#!/usr/bin/env python3

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import ortools
import pandas as pd
from ortools.linear_solver import pywraplp
from functools import lru_cache
import math
try:
    from tqdm import tqdm
except:
    tqdm = lambda x, desc: x

def get_element(df:pd.DataFrame, rowname:str, colname:str):
    selector = df['Unnamed: 0'] == rowname
    df = df[selector]
    return df[colname].to_numpy()[0]

def get_sum_of_defined_vars(df):
    return df.sum(axis=1, skipna=True).sum(skipna=True)

def pair_array(arr:list, is_circular=False)->list:
    # Given an array [1, 2, 3, 4]
    # Return an array of pairs [(1,2), (2, 3), (3,4)]
    ret = [(arr[i], arr[i+1],) for i in range(len(arr) - 1)]
    if is_circular and len(arr) >= 2:
        ret.append((arr[-1], arr[0],))
    return ret

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

    @lru_cache(maxsize=128)
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

    @lru_cache(maxsize=128)
    def is_line_circular(self, line):
        route = self.get_line_stations(line)
        first, last = route[0], route[-1]
        d1 = get_element(self.distance_df, first, last)
        d2 = get_element(self.distance_df, last, first)
        if not math.isnan(d1) and not math.isnan(d2):
            return True
        else:
            return False

    @lru_cache(maxsize=128)
    def station_pairs(self, line:str, downstream=False)->list:
        # Given a line with a list of stations [A, B, C, D]
        # Return a set of pairs [(A, B), (B, C), (C, D)]
        stations = self.get_line_stations(line)
        if downstream:
            stations = stations[::-1]
        return pair_array(stations, self.is_line_circular(line))

    @lru_cache(maxsize=128)
    def get_total_line_distance(self, line:str)->list:
        stations = self.get_line_stations(line)
        stations = pair_array(stations, self.is_line_circular(line))
        total = 0
        for x, y in stations:
            total += get_element(self.distance_df, x, y)
        return total
    

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

    @lru_cache(maxsize=128)
    def get_max_size(self):
        return get_sum_of_defined_vars(self.distance_df)

    def set_objective_coefficients(self):
        # Set the coefficients of the objective. For every path that is taken
        # the weight is the same as the distance taken in the excel sheet
        maxdistance = self.get_max_size()
        maxdistance *= maxdistance
        for st1 in self.stop_names:
            for st2 in self.stop_names:
                dist = get_element(self.distance_df, st1, st2)
                if math.isnan(dist):
                    dist = maxdistance
                var = self.edge[st1][st2]
                self.objective.SetCoefficient(var, int(dist))

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

        self.solver = pywraplp.Solver(\
                        'LPWrapper',\
                        pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)


        # Variable: Number of trains for each line, upstream and downstream
        self.var_upstream_trains_per_hour = {}
        self.var_downstream_trains_per_hour = {}
        for line in self.line_names:
            self.var_upstream_trains_per_hour[line] = self.solver.IntVar(\
                            0, self.solver.infinity(), f"n_trains_up({line})")
            self.var_downstream_trains_per_hour[line] = self.solver.IntVar(\
                            0, self.solver.infinity(), f"n_trains_dn({line})")

        self.objective = self.solver.Objective()
        for line in self.line_names:
            self.objective.SetCoefficient(\
                self.var_upstream_trains_per_hour[line], 1)
            self.objective.SetCoefficient(\
                self.var_downstream_trains_per_hour[line], 1)
        self.objective.SetMinimization()

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

        self.add_all_requirements()

        # Add a constraint that each leg has enough capacity
        # to handle its traffic
        self.add_constraints_no_leg_overloads()

        self.constrain_upstream_downstream_same()



    def initialize_capacity_matrix(self):
        outer = {}
        for s1 in self.stop_names:
            outer[s1] = {s: 0 for s in self.stop_names}
        return outer

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
        description = "Finding shortest paths"
        pairs = [(s1, s2,) for s1 in self.stop_names for s2 in self.stop_names]
        for s1, s2 in tqdm(pairs, desc=description):
            self.add_requirements(s1, s2)

    def fill_line_capacity_per_train(self, line:str):
        capacity = get_element(self.train_capacity_df, line, 'Capacity')
        for x, y in self.station_pairs(line):
            self.line_capacity_per_train_up[line][x][y] = capacity
        for x, y in self.station_pairs(line, downstream=True):
            self.line_capacity_per_train_down[line][x][y] = capacity

    def add_constraints_no_leg_overloads(self):
        def constrain(source, dest):
            req = self.capacity_required[source][dest]
            cons = self.solver.Constraint(int(req), self.solver.infinity())
            # debugstr = f"constrain({source},{dest})  : {req} <= 0 "
            # acc = 0.0
            for line in self.line_names:
                capacity = self.line_capacity_per_train_up[line][source][dest]
                var = self.var_upstream_trains_per_hour[line]
                cons.SetCoefficient(var, int(capacity))
                # debugstr = debugstr + f"+ ({capacity} * {var})"
                # acc += capacity
                capacity = self.line_capacity_per_train_down[line][source][dest]
                var = self.var_downstream_trains_per_hour[line]
                cons.SetCoefficient(var, int(capacity))
                # debugstr = debugstr + f"+ ({capacity} * {var})"
                # acc += capacity
            # debugstr = debugstr + f") <= infinity"
            # if req > 0 and acc == 0.0:
            #    print(debugstr)

        for s1 in self.stop_names:
            for s2 in self.stop_names:
                constrain(s1, s2)

    def solve(self):
        self.solver.Solve()

    def constrain_upstream_downstream_same(self):
        # For non-circular routes, upstream requirement is the same as
        # downstream requirement since the same trains shuttle back and forth
        print()
        print("****")
        print("For non-circular lines, forcing same number of upstream and "\
            + " downstream trains")
        for line in self.line_names:
            if not self.is_line_circular(line):
                cons = self.solver.Constraint(0, 0)
                down = self.var_downstream_trains_per_hour[line]
                up = self.var_upstream_trains_per_hour[line]
                cons.SetCoefficient(up, 1)
                cons.SetCoefficient(down, -1)

    def print_solution(self):
        print()
        print('-' * 80)
        print()
        print("If upstream and downstream may have different number of trains:")
        print(f"    Number of trains required (per hour) = " \
            + f": {self.objective.Value()}")
        print()

        for line in self.line_names:
            up = self.var_upstream_trains_per_hour[line].SolutionValue()
            down = self.var_downstream_trains_per_hour[line].SolutionValue()
            print(f"Line {line}: Upstream: {int(up):>5d} " + \
                        f"Downstream: {int(down):>5d}")
        ntrains = 0
        for line in self.line_names:
            up = self.var_upstream_trains_per_hour[line].SolutionValue()
            down = self.var_downstream_trains_per_hour[line].SolutionValue()
            if not self.is_line_circular(line):
                up = max(up, down)
                down = up
            ntrains += up
            ntrains += down

        print('-' * 80)

        print()
        print("Circular routes can have different number of rains clockwise ")
        print("and anticlockwise")
        print("If upstream and downstream have the same number of trains:")
        print(f"    Number of trains required (per hour) = {ntrains}")
        print()

        for line in self.line_names:
            up = self.var_upstream_trains_per_hour[line].SolutionValue()
            down = self.var_downstream_trains_per_hour[line].SolutionValue()
            if not self.is_line_circular(line):
                up = max(up, down)
                down= up
            print(f"Line {line}: Upstream: {int(up):>5d} " + \
                        f"Downstream: {int(down):>5d}")

        print('-' * 80)

        print()
        print("The above didn't take into consideration that trains can return")
        print("after they have reached the destination. So total number of ")
        print("trains required will be smaller. Next we take this into ")
        print("consideration, and the total end-to-end distance of each line.")

        print()
        for l in self.line_names:
            print(f"Length of line {l} is {self.get_total_line_distance(l)} "\
                + " minutes")
        print()
        ntrains = 0
        for line in self.line_names:
            up = self.var_upstream_trains_per_hour[line].SolutionValue()
            down = self.var_downstream_trains_per_hour[line].SolutionValue()
            if not self.is_line_circular(line):
                up = max(up, down)
                down = up
                round_trip_time = self.get_total_line_distance(line) * 2
                req = math.ceil(up * round_trip_time / 60.0)
                print(f"Line {line} -          Trains required: {req}")
                ntrains += req
            else:
                round_trip_time = self.get_total_line_distance(line)
                upreq = math.ceil(up * round_trip_time / 60.0)
                downreq = math.ceil(down * round_trip_time / 60.0)
                print(f"Line {line} - Upstream Trains Required: {upreq} " +\
                    f"        Downstream Trains required: {downreq}")
                ntrains += upreq
                ntrains += downreq
        print()
        print(f"Total number of trains required: {ntrains}")


            

    def main(self):
        self.solve()
        self.print_solution()


if "__main__" == __name__:
    # dist, path = ShortestPath("Assignment_DA_2_Task_3_data.xlsx", 'A', 'P')\
    #    .get_shortest_path()
    # x = ShortestPath("Assignment_DA_2_Task_3_data.xlsx", "A", "B")
    # for i in ['L1', 'L2', 'L3', 'L4']:
    #    print(i, x.get_total_line_distance(i))
    TrainCapacity('Assignment_DA_2_Task_3_data.xlsx').main()

