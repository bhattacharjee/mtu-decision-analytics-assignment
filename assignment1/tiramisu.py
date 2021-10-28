#!/usr/bin/env python3

from ortools.sat.python import cp_model

PERSON = ["James", "Daniel", "Emily", "Sophie"]
STARTER = ["Prawn_Cocktail", "Onion_Soup", "Mushroom_Tart", "Carpaccio"]
MAINCOURSE = ["Baked_Mackerel", "Fried_Chicken", "Filet_Steak", "Vegan_Pie"]
DRINK = ["Red_Wine", "Beer", "White_Wine", "Coke"]
DESSERT = ["Apple_Crumble", "Ice_Cream", "Chocolate_Cake", "Tiramisu"]


class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(\
            self,
            person:list,
            starter:list,
            maincourse:list,
            drink:list,
            dessert:list,
            person_starter:dict,
            person_maincourse:dict,
            person_drink:dict,
            person_dessert:dict):
        super().__init__()
        self.person = person
        self.starter = starter
        self.maincourse = maincourse
        self.drink = drink
        self.dessert = dessert
        self.person_starter = person_starter
        self.person_maincourse = person_maincourse
        self.person_drink = person_drink
        self.person_dessert = person_dessert
        self.solutions = 0

    def OnSolutionCallback(self):
        self.solutions = self.solutions + 1
        print(f"Solution #{self.solutions:06d}")
        print("----------------")

        for person in self.person:
            print(f"- {person}")
            [print(f" - {starter}") for starter in self.starter\
                    if self.Value(self.person_starter[person][starter])]
            [print(f" - {maincourse}") for maincourse in self.maincourse\
                    if self.Value(self.person_maincourse[person][maincourse])]
            [print(f" - {dessert}") for dessert in self.dessert\
                    if self.Value(self.person_dessert[person][dessert])]
            [print(f" - {drink}") for drink in self.drink\
                    if self.Value(self.person_drink[person][drink])]
        print()
        print()


def create_variables_and_implicit_constraints(
        model,
        var_list1:list,
        var_list2: list) -> dict:

    # Create the variables
    ret_dict = {}
    for var1 in var_list1:
        variables = {}
        for var2 in var_list2:
            variables[var2] = model.NewBoolVar(f"{var1}--{var2}")
        ret_dict[var1] = variables

    # Every item in var_list1 has a different property from var_list2
    for i in range(len(var_list1)):
        for j in range(i+1, len(var_list1)):
            for k in range(len(var_list2)):
                model.AddBoolOr(                                            \
                            [                                               \
                                ret_dict[var_list1[i]][var_list2[k]].Not(), \
                                ret_dict[var_list1[j]][var_list2[k]].Not()  \
                            ]                                               \
                        )

    # At least one item in var_list2 for each item in var_list1
    for v1 in var_list1:
        variables = []
        for v2 in var_list2:
            variables.append(ret_dict[v1][v2])
        model.AddBoolOr(variables)

    # Max one property for every item in var_list1
    for v1 in var_list1:
        for i in range(len(var_list2)):
            for j in range(i+1, len(var_list2)):
                model.AddBoolOr(                                            \
                        [                                                   \
                            ret_dict[v1][var_list2[i]].Not(),               \
                            ret_dict[v1][var_list2[j]].Not()                \
                        ]                                                   \
                    )

    return ret_dict


def main():
    model = cp_model.CpModel()

    person_starter = create_variables_and_implicit_constraints(             \
                            model,                                          \
                            PERSON,                                         \
                            STARTER)

    person_maincourse = create_variables_and_implicit_constraints(          \
                            model,                                          \
                            PERSON,                                         \
                            MAINCOURSE)

    person_drink = create_variables_and_implicit_constraints(               \
                            model,                                          \
                            PERSON,                                         \
                            DRINK)

    person_dessert = create_variables_and_implicit_constraints(             \
                            model,                                          \
                            PERSON,                                         \
                            DESSERT)

    solution_printer = SolutionPrinter(                                     \
                            person=PERSON,                                  \
                            starter=STARTER,                                \
                            maincourse=MAINCOURSE,                          \
                            drink=DRINK,                                    \
                            dessert=DESSERT,                                \
                            person_starter=person_starter,                  \
                            person_maincourse=person_maincourse,            \
                            person_drink=person_drink,                      \
                            person_dessert=person_dessert)

    # Explicit Constraint 1
    model.AddBoolAnd([person_starter["Emily"]["Prawn_Cocktail"].Not()])
    model.AddBoolAnd([person_maincourse["Emily"]["Baked_Mackerel"].Not()])

    # Explicit Constraint 2
    model.AddBoolAnd([person_starter["Daniel"]["Onion_Soup"].Not()])
    model.AddBoolAnd([person_drink["James"]["Beer"].Not()])



    solver = cp_model.CpSolver()
    status = solver.SearchForAllSolutions(model, solution_printer)
    print(solver.StatusName(status))

if "__main__" == __name__:
    main()

