#!/usr/bin/env python3

from ortools.sat.python import cp_model

def numbers() -> range:
    """This routine just makes it easier to loop
       through the numbers 1..9

    Returns:
        range: [description]
    """
    return range(1,10)

def row(r:int) -> list:
    """Returns the list of tuples that specify the
       indices for all squares of a row
       This routine just makes it easier to iterate a row

    Args:
        r (int): tow number whose indices are to be generated

    Returns:
        list: list of tuples specifying the row indices, eg.
              [(3, 0), (3, 1), (3, 2), ... ]
    """
    return [(r, i) for i in range(9)]

def column(c:int) -> list:
    """Returns the list of tuples that specify the indices of
       all squares for a given column.
       This routine just makes it easier to iterate through
       all squares of a column

    Args:
        c (int): the column whose indices are to be generated

    Returns:
        list: list of tuples specifying the column indices, eg.
              [(0, 4), (1, 4), (2, 4), ... ]
    """
    return [(i, c) for i in range(9)]

def square(ind: tuple) -> list:
    """Returns a list of tuples that specify the indices of all cells
       inside a sub square

    Args:
        ind (tuple): specifies the indices of the top left cell of the
                     sub-square

    Returns:
        list: all cells in the sub-square. eg.
              [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0),
               (2, 1), (2, 2)]
    """
    return [(i + ind[0], j + ind[1]) for i in range(3) for j in range(3)]

def square_starts() -> list:
    """Returns the index of the top left square of each cell

    Returns:
        list: [(0, 0), (0, 3), (0, 6), (3, 0), (3, 3), (3, 6),
                (6, 0), (6, 3), (6, 6)]

    Yields:
        Iterator[list]: the tuple specifying the index
    """
    #for i in range(0,9,3):
    #    for j in range(0,9,3): yield (i, j)
    # Using a more concise way of writing the above lines
    return ((i, j) for i in range(0, 9, 3) for j in range(0, 9, 3))

class SudokuSolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print the solution, also validate that the solution is indeed
       correct and that no numbers are repeated and that all
       numbers are present

    Args:
        cp_model ([type]): [description]
    """
    def __init__(\
            self,
            sudoku:dict):
        super().__init__()
        self.sudoku = sudoku
        self.solutions = 0

    def validate_all_numbers_present(self, indices:dict):
        """validate that all numbers are present in the
           set of indices provided

        Args:
            indices (dict): indices to look for all the numbers,
                            this could be all indices of a row,
                            all indices of a column, or all indices
                            of a sub-square
        """
        s = set()
        count = 0

        def update(i, j, k):
            nonlocal count, s
            if self.Value(self.sudoku[i][j][k]):
                count = count + 1
                s.add(k)

        [update(i, j, k) for i, j in indices for k in numbers()]
        if (len(s) != 9 or count != 9):
            print("Either all numbers not present, or some are repeated" +  \
                    "in squares ", indices)
        assert(9 == len(s) and 9 == count)

    def validate_cell(self, i, j):
        """Validate that each cell should have exactly one number

        Args:
            i ([type]): first axis index of the cell
            j ([type]): second axis index of the cell
        """
        # Each cell should have exactly one number
        count = 0
        for k in numbers():
            if self.Value(self.sudoku[i][j][k]): count = count + 1
        if (1 != count): print(f"sudoku[{i},{j}] has {count} values")
        assert(count == 1)

    def validate_solution(self):
        """Validate few things things
            1. for each cell, only one variable must be true
            2. For each row, all numbers must be present, and only once
            3. For each column, all numbers must be present and only once
            4. For each sub-square, all numbers must be present and only once
        """
        [self.validate_cell(i, j) for i in range(9) for j in range(9)]

        [self.validate_all_numbers_present(row(i)) for i in range(9)]

        [self.validate_all_numbers_present(column(i)) for i in range(9)]

        [self.validate_all_numbers_present(square(sqs)) for \
                sqs in square_starts()]

    def OnSolutionCallback(self):
        self.validate_solution()
        self.solutions = self.solutions + 1

        print(f"Solution # {self.solutions}")
        print("++=======++===+===+===+===+===+===+===+===+===++")
        print("||   #   ||-0-|-1-|-2-|-3-|-4-|-5-|-6-|-7 |-8-||")
        print("++*******++***+***+***+***+***+***+***+***+***++")

        for i in range(9):
            output_line = f"||   {i}  "
            first = True
            for j in range(9):
                for k in numbers():
                    if self.Value(self.sudoku[i][j][k]):
                        if j % 3 == 0:
                            addstr = f" || {k}" if first else f" | {k}"
                        else:
                            addstr = f" || {k}" if first else f" . {k}"
                        first = False
                        output_line = output_line + addstr
                        break
            output_line = output_line + f" ||"
            print(output_line)
            if (i + 1) % 3 == 0:
                print("++*******++***********+***********+***********++")
            else:
                print("++............................................++")

        print()
        print()


def create_variables(model) -> dict:
    """Create the variables, the variables are a 3 D array
       of 9x9x9
       For each row, for each column, there are 9 boolean variables

    Args:
        model ([type]): [description]

    Returns:
        dict: dictionary of variables, can be indexed in a 3D way
    """
    def get_inner_dict(i, j):
        return {k: model.NewBoolVar(f"--[{i},{j}]->{k}--") for k in numbers()}

    def get_outer_dict(i):
        return {j: get_inner_dict(i, j) for j in range(9)}

    return {i: get_outer_dict(i) for i in range(9)}


def set_constraint_one_number_per_cell(model, sudoku:dict):
    """add constraint that for each row, column, only one of the variable
       must be true. That is one cell can contain exactly one number

    Args:
        model ([type]): [description]
        sudoku (dict): [description]
    """
    for r in range(9):
        for c in range(9):
            model.AddBoolOr([sudoku[r][c][n] for n in numbers()])


def set_constraint_no_duplicates(model, sudoku:dict, indices):
    """Given a set of indices for cells (eg. all cells in one row, 
       or one column), ensure that there are no duplicates in those cells.

    Args:
        model ([type]): [description]
        sudoku (dict): [description]
        indices ([type]): indices of the cells in which there should
                          not be any duplicates
    """
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
    """Given a set of indices for cells (eg. all cells in one row,
       or one column), ensure that all the numbers 1..9 are present.

    Args:
        model ([type]): [description]
        sudoku (dict): [description]
        indices ([type]): [description]
    """
    for n in numbers():
        model.AddBoolOr([sudoku[r][c][n] for r, c in indices])


def set_explicit_constraints(model, sudoku:dict):
    """set the explicit constraints according to what is specified
       in the assignment document

    Args:
        model ([type]): [description]
        sudoku (dict): [description]
    """
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

    for i in range(9):
        outstr = ""
        for j in range(9):
            try:
                outstr = outstr + str(explicit_constraints[i][j]) + " "
            except:
                outstr = outstr + '. '
        print(outstr)


def sudoku_main():
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
    solution_printer = SudokuSolutionPrinter(sudoku)
    status = solver.SearchForAllSolutions(model, solution_printer)
    print(solver.StatusName(status))


if "__main__" == __name__:
    sudoku_main()
