#!/usr/bin/env python3

from ortools.sat.python import cp_model

def numbers() -> list:
    return [x for x in range(1,10)]


def row(r:int) -> list:
    return [(r, i) for i in range(9)]

def column(c:int) -> list:
    return [(i, c) for i in range(9)]

def square(ind: tuple) -> list:
    r, c = ind
    return [(r+i,c+j) for i in range(3) for j in range(3)]

def square_starts() -> list:
    return [(i, j) for i in range(0,9,3) for j in range(0,9,3)]

class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(\
            self,
            sudoku:dict):
        super().__init__()
        self.sudoku = sudoku
        self.solutions = 0


    def validate_all_numbers_present(self, indices:dict):
        s = set()
        count = 0

        def update(i, j, k):
            nonlocal count
            nonlocal s
            if self.Value(self.sudoku[i][j][k]):
                count = count + 1
                s.add(k)

        [update(i, j, k) for i, j in indices for k in numbers()]

        if (len(s) != 9 or count != 9):
            print("Either all numbers not present, or some are repeated" +  \
                    "in squares ", indices)
        assert(9 == len(s) and 9 == count)


    def validate_solution(self):

        def validate_cell(i, j):
            count = 0
            for k in numbers():
                if self.Value(self.sudoku[i][j][k]): count = count + 1
            if (1 != count):
                print(f"sudoku[{i},{j}] has {count} values")
            assert(count == 1)

        [validate_cell(i, j) for i in range(9) for j in range(9)]

        [self.validate_all_numbers_present(row(i)) for i in range(9)]

        [self.validate_all_numbers_present(column(i)) for i in range(9)]

        [self.validate_all_numbers_present(square(sqs)) for \
                sqs in square_starts()]



    def OnSolutionCallback(self):
        self.validate_solution()
        self.solutions = self.solutions + 1

        print(f"Solution # {self.solutions}")
        print("=======+===+===+===+===+===+===+===+===+===+")
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


def create_variables(model):
    def get_inner_dict(i, j):
        return {k: model.NewBoolVar(f"--[{i},{j}]->{k}--") for k in numbers()}

    def get_outer_dict(i):
        return {j: get_inner_dict(i, j) for j in range(9)}

    return {i: get_outer_dict(i) for i in range(9)}


def set_constraint_one_number_per_cell(model, sudoku:dict):
    for r in range(9):
        for c in range(9):
            variables = []
            for n in numbers():
                variables.append(sudoku[r][c][n])
            model.AddBoolOr(variables)


def set_constraint_no_duplicates(model, sudoku:dict, indices):
    def update(model, sudoku, i, j, n):
        r1, c1 = indices[i]
        r2, c2 = indices[j]
        model.AddBoolOr(                                                    \
                [                                                           \
                    sudoku[r1][c1][n].Not(),                                \
                    sudoku[r2][c2][n].Not(),                                \
                ])

    for i in range(len(indices)):
        for j in range(i+1, len(indices)):
            for n in numbers():
                update(model, sudoku, i, j, n)



def set_constraint_all_numbers_present(model, sudoku:dict, indices):
    for n in numbers():
        variables = []
        for r, c in indices:
            variables.append(sudoku[r][c][n])
        model.AddBoolOr(variables)


def set_explicit_constraints(model, sudoku:dict):
    explicit_constraints = \
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

    for r, val in explicit_constraints.items():
        for c, n in val.items():
            model.AddBoolAnd([sudoku[r][c][n]])


def main():
    model = cp_model.CpModel()

    sudoku = create_variables(model)

    set_constraint_one_number_per_cell(model, sudoku)

    # No duplicates in each row and each column
    for i in range(9):
        set_constraint_no_duplicates(model, sudoku, row(i))
        set_constraint_no_duplicates(model, sudoku, column(i))

    # No duplicates in each sub-square
    for sqs in square_starts():
        set_constraint_no_duplicates(model, sudoku, square(sqs))

    # Every number in each row and each column
    for i in range(9):
        set_constraint_all_numbers_present(model, sudoku, row(i))
        set_constraint_all_numbers_present(model, sudoku, column(i))

    # Every number in each sub-square
    for sqs in square_starts():
        set_constraint_all_numbers_present(model, sudoku, square(sqs))

    set_explicit_constraints(model, sudoku)

    solver = cp_model.CpSolver()
    solution_printer = SolutionPrinter(sudoku)
    status = solver.SearchForAllSolutions(model, solution_printer)
    print(solver.StatusName(status))


if "__main__" == __name__:
    main()
