#!/usr/bin/env python3

from ortools.sat.python import cp_model

EXPLICIT_CONSTRAINTS = \
{
    0: {7: 3},
    1: {0: 7, 2: 5, 4: 2},
    2: {1: 9, 6: 4},
    3: {5: 4, 8: 2},
    4: {1: 5, 2: 9, 3: 6, 8: 8},
    5: {0: 3, 4: 1, 7: 5},
    6: {0: 5, 1: 7, 4: 6, 6: 1},
    7: {3: 3},
    8: {0: 6, 3: 4, 8: 5}
}

def numbers() -> list:
    return [x for x in range(1,10)]


def row(r:int) -> list:
    return [(r, i) for i in range(9)]

def column(c:int) -> list:
    return [(i, c) for i in range(9)]

def square(ind: tuple) -> list:
    r, c = ind
    retval = []
    for i in range(3):
        for j in range(3):
            retval.append((r+i, c+j,))
    return retval

def square_starts() -> list:
    retval = []
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            retval.append((i, j,))
    return retval


class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(\
            self,
            sudoku:dict):
        super().__init__()
        self.sudoku = sudoku
        self.solutions = 0

    def validate(self, sudoku:dict):

        # Each cell should have exactly one variable set
        for i in range(9):
            for j in range(9):
                count = 1
                for k in numbers():
                    if self.Value(sudoku[i][j][k]): count = count + 1
                if (1 != count):
                    print(f"sudoku[{i},{j}] has {count} values")
                assert(count == 1)

        # Each column should have exactly one variable for each number set
        for i in range(9):
            s = set()
            for j in range(9):
                for k in numbers():
                    if self.Value(sudoku[i][j][k]):
                        s.add(k)
            for n in numbers():
                if n not in s:
                    print(f"Number {n} not present in row {i}")
                assert(n in s)
                if len(s) != len(numbers()):
                    print(f"Not all numbers present in row {i}")
                assert(len(s) == len(numbers()))



    def OnSolutionCallback(self):
        self.solutions = self.solutions + 1
        print(f"Solution #{self.solutions:06d}")
        print("=======+===+===+===+===+===+===+===+===+===+")
        #self.validate(self.sudoku)

        print("       | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |")
        print("=======+===+===+===+===+===+===+===+===+===+")
        for i in range(9):
            output_line = f"   {i}  "
            for j in range(9):
                for k in numbers():
                    if self.Value(self.sudoku[i][j][k]):
                        output_line = output_line + f" | {k}"
                        break
            output_line = output_line + f" |"
            print(output_line)
            print("-------+---+---+---+---+---+---+---+---+---+")
        print()
        print()
        print()
        print()


def create_variables(model):
    ret_dict = {}
    for i in range(9):
        dictionary = {}
        for j in range(9):
            inner_dictionary = {}
            for k in numbers():
                inner_dictionary[k] = model.NewBoolVar(f"--[{i},{j}]->{k}--")
            dictionary[j] = inner_dictionary
        ret_dict[i] = dictionary

    return ret_dict


def constraint_one_number_per_cell(model, sudoku:dict):
    for r in range(9):
        for c in range(9):
            variables = []
            for n in numbers():
                variables.append(sudoku[r][c][n])
            model.AddBoolOr(variables)


def constraint_no_duplicates_generic(model, sudoku:dict, indices):
    # Ensure that there are no duplicates in the set of indices passed
    for ii in range(len(indices)):
        for jj in range(ii + 1, len(indices)):
            for n in numbers():
                r1, c1 = indices[ii]
                r2, c2 = indices[jj]
                model.AddBoolOr(                                            \
                        [                                                   \
                            sudoku[r1][c1][n].Not(),                        \
                            sudoku[r2][c2][n].Not(),                        \
                        ])


def constraint_all_numbers_present_generic(model, sudoku:dict, indices):
    for n in numbers():
        variables = []
        for r, c in indices:
            variables.append(sudoku[r][c][n])
        model.AddBoolOr(variables)

def main():
    model = cp_model.CpModel()

    sudoku = create_variables(model)

    constraint_one_number_per_cell(model, sudoku)

    # No duplicates in each row and each column
    for i in range(9):
        constraint_no_duplicates_generic(model, sudoku, row(i))
        constraint_no_duplicates_generic(model, sudoku, column(i))

    # No duplicates in each sub-square
    for sqs in square_starts():
        constraint_no_duplicates_generic(model, sudoku, square(sqs))

    # Every number in each row and each column
    for i in range(9):
        constraint_all_numbers_present_generic(model, sudoku, row(i))
        constraint_all_numbers_present_generic(model, sudoku, column(i))

    # Every number in each sub-square
    for sqs in square_starts():
        constraint_all_numbers_present_generic(model, sudoku, square(sqs))

    # Explicit Constraints
    for r, val in EXPLICIT_CONSTRAINTS.items():
        for c, n in val.items():
            model.AddBoolAnd([sudoku[r][c][n]])

    solver = cp_model.CpSolver()
    solution_printer = SolutionPrinter(sudoku)
    status = solver.SearchForAllSolutions(model, solution_printer)
    print(solver.StatusName(status))


if "__main__" == __name__:
    main()
