#!/usr/bin/env python3

from ortools.sat.python import cp_model

PERSON = ["James", "Daniel", "Emily", "Sophie"]
STARTER = ["Prawn_Cocktail", "Onion_Soup", "Mushroom_Tart", "Carpaccio"]
MAINCOURSE = ["Baked_Mackerel", "Fried_Chicken", "Filet_Steak", "Vegan_Pie"]
DRINK = ["Red_Wine", "Beer", "White_Wine", "Coke"]
DESSERT = ["Apple_Crumble", "Ice_Cream", "Chocolate_Cake", "Tiramisu"]

CONSTRAINT3_INTERPRETATION_1 = True
CONSTRAINT3_INTERPRETATION_2 = False 
CONSTRAINT3_INTERPRETATION_3 = False

# For constraint 9
# INTERPRETATION 1 is True if both 
# INTERPRETATION 2 and INTERPRETATION 3 are false
CONSTRAINT9_INTERPRETATION_2 = True
CONSTRAINT9_INTERPRETATION_3 = False 


class TiramisuSolutionPrinter(cp_model.CpSolverSolutionCallback):
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

    def validate_matrix(self, matrix:dict, axis1:list, axis2:list):
        """[summary]

        Args:
            matrix (dict): [description]
            axis1 (list): [description]
            axis2 (list): [description]
        """
        for v1 in axis1:
            i = 0
            for v2 in axis2:
                if self.Value(matrix[v1][v2]): i = i + 1
            assert(i == 1)
        for v2 in axis2:
            i = 0
            for v1 in axis1:
                if self.Value(matrix[v1][v2]): i = i + 1
            assert(i == 1)


    def OnSolutionCallback(self):
        self.solutions = self.solutions + 1
        print(f"Solution #{self.solutions:06d}")
        print("----------------")

        self.validate_matrix(self.person_dessert, self.person, self.dessert)
        self.validate_matrix(self.person_drink, self.person, self.drink)
        self.validate_matrix(self.person_maincourse, self.person, self.maincourse)
        self.validate_matrix(self.person_starter, self.person, self.starter)
        

        for person in self.person:
            print(f"- {person}")
            [print(f"     - {dessert}") for dessert in self.dessert\
                    if self.Value(self.person_dessert[person][dessert])]
            [print(f"     - {drink}") for drink in self.drink\
                    if self.Value(self.person_drink[person][drink])]
            [print(f"     - {starter}") for starter in self.starter\
                    if self.Value(self.person_starter[person][starter])]
            [print(f"     - {maincourse}") for maincourse in self.maincourse\
                    if self.Value(self.person_maincourse[person][maincourse])]
                    
        for person in self.person:
            if self.Value(self.person_dessert[person]['Tiramisu']):
                print(f"\n\n{person} has the Tiramisu")
                break
        
        print()
        print()


def create_variables_and_implicit_constraints(
        model,
        var_list1: list,
        var_list2: list) -> dict:
    """Create a 2D variable array given the two axes
       For example, given Person and Drink, create a 2D array
       for each person and drink
       Also create the implicit constraints that
       1. each person must have a drink
       2. each person can have exactly one drink
       3. No two persons have the same drink

    Args:
        model ([type]): the CP SAT model
        var_list1 (list): list of items in first axes (eg. person names)
        var_list2 (list): list of items in second axis (eg. drink names)

    Returns:
        dict: [description]
    """

    # Create the variables
    ret_dict = {}
    for var1 in var_list1:
        ret_dict[var1] =                                                    \
            {var2: model.NewBoolVar(f"{var1}--{var2}") for var2 in var_list2}

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
        model.AddBoolOr([ret_dict[v1][v2] for v2 in var_list2])

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


def tiramisu_main():
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

    solution_printer = TiramisuSolutionPrinter(                             \
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
    # ---------------------
    # Emily does not like prawn cocktail as starter,
    # nor does she want baked mackerel as main course
    model.AddBoolAnd([person_starter["Emily"]["Prawn_Cocktail"].Not()])
    model.AddBoolAnd([person_maincourse["Emily"]["Baked_Mackerel"].Not()])
    # ---------------------

    # Explicit Constraint 2
    # ---------------------
    # Daniel does not want the onion soup as starter and
    # James does not drink beer
    model.AddBoolAnd([person_starter["Daniel"]["Prawn_Cocktail"].Not()])
    model.AddBoolAnd([person_drink["James"]["Beer"].Not()])
    # ---------------------

    # Explicit Constraint 3
    # ---------------------
    # Sophie will only have fried chicken as main course
    # if she does not have to take the prawn cocktail as starter
    #
    # Interpretation 1:
    # Or in other words Fried Chicken implies No Prawn Cocktail, and vice versa
    #
    # Interpretation 2:
    # Another way to interpret this condition is to say that Sophie has
    # either Prawn Cocktail or Fried Chicken, a xor condition.
    #
    # Interpretation 3:
    # A third way to interpret this condition is to say that
    # if she does not have prawn cocktail, she will definitely have fried
    # chicken
    # Or in other words, Not Prawn Cocktail implies Fried Chicken
    # 
    # 
    if CONSTRAINT3_INTERPRETATION_1:
        model.AddBoolOr(                                                    \
                    [                                                       \
                        person_starter["Sophie"]["Prawn_Cocktail"].Not(),   \
                        person_maincourse["Sophie"]["Fried_Chicken"].Not()  \
                    ]                                                       \
                )
    elif CONSTRAINT3_INTERPRETATION_2:
        model.AddBoolXOr(                                                   \
                    [                                                       \
                        person_starter["Sophie"]["Prawn_Cocktail"],         \
                        person_maincourse["Sophie"]["Fried_Chicken"]        \
                    ]                                                       \
                )
    elif CONSTRAINT3_INTERPRETATION_3:
        model.AddBoolAnd(                                                   \
                    [                                                       \
                        person_maincourse["Sophie"]["Fried_Chicken"]        \
                    ]                                                       \
                ).OnlyEnforceIf(person_starter["Sophie"]["Prawn_Cocktail"].Not())
    else:
        raise Exception('At least one interpretation of constraint 3 must hold')
    # ---------------------

    # Explicit constraint 4
    # ---------------------
    # The filet steak main course should be combined with the
    # onion soup as starter and with the apple crumble for dessert
    for person in PERSON:
        model.AddBoolOr(                                                    \
                [                                                           \
                    person_maincourse[person]["Filet_Steak"].Not(),         \
                    person_starter[person]["Onion_Soup"]                    \
                ])
        model.AddBoolOr(                                                    \
                [                                                           \
                    person_starter[person]["Onion_Soup"].Not(),             \
                    person_maincourse[person]["Filet_Steak"]                \
                ])
        model.AddBoolOr(                                                    \
                [                                                           \
                    person_maincourse[person]["Filet_Steak"].Not(),         \
                    person_dessert[person]["Apple_Crumble"]                 \
                ])
        model.AddBoolOr(                                                    \
                [                                                           \
                    person_dessert[person]["Apple_Crumble"].Not(),          \
                    person_maincourse[person]["Filet_Steak"]                \
                ])
    # ---------------------


    # Explicit Constraint 5
    # ---------------------
    # The person who orders the mushroom tart as starter
    # also orders the red wine
    for person in PERSON:
        model.AddBoolOr(                                                    \
                [                                                           \
                    person_starter[person]["Mushroom_Tart"].Not(),          \
                    person_drink[person]["Red_Wine"]                        \
                ])
        model.AddBoolOr(                                                    \
                [                                                           \
                    person_starter[person]["Mushroom_Tart"],                \
                    person_drink[person]["Red_Wine"].Not()                  \
                ])
    # ---------------------


    # Explicit Constraint 6
    # ---------------------
    # The baked mackerel should not be combined with ice cream for dessert,
    # nor should the vegan pie be ordered as main together with
    # prawn cocktail or carpaccio as starter
    for person in PERSON:
        model.AddBoolOr(                                                    \
                [                                                           \
                    person_maincourse[person]["Baked_Mackerel"].Not(),      \
                    person_dessert[person]["Ice_Cream"].Not()               \
                ])
        model.AddBoolOr(                                                    \
                [                                                           \
                    person_maincourse[person]["Vegan_Pie"].Not(),           \
                    person_starter[person]["Prawn_Cocktail"].Not()          \
                ])
        model.AddBoolOr(                                                    \
                [                                                           \
                    person_maincourse[person]["Vegan_Pie"].Not(),           \
                    person_starter[person]["Carpaccio"].Not()               \
                ])


    # Explicit Constraint 7
    # ---------------------
    # The filet steak should be eaten with either beer or coke for drinks
    for person in PERSON:
        model.AddBoolOr(                                                    \
                [                                                           \
                    person_maincourse[person]["Filet_Steak"].Not(),         \
                    person_drink[person]["Beer"],                           \
                    person_drink[person]["Coke"]                            \
                ])



    # Explicit Constraint 8
    # ---------------------
    # One of the women drinks white wine, while the other
    # prefers red wine for drinks
    model.AddBoolOr(                                                        \
            [                                                               \
                person_drink["Emily"]["White_Wine"],                        \
                person_drink["Emily"]["Red_Wine"]                           \
            ])
    model.AddBoolOr(                                                        \
            [                                                               \
                person_drink["Sophie"]["White_Wine"],                       \
                person_drink["Sophie"]["Red_Wine"]                          \
            ])


    # Explicit Constraint 9
    # ---------------------
    # One of the men has chocolate cake for dessert while the other
    # prefers not to have ice cream or coke but
    # will accept one of the two if necessary
    model.AddBoolXOr(                                                       \
            [                                                               \
                person_dessert["James"]["Chocolate_Cake"],                  \
                person_dessert["Daniel"]["Chocolate_Cake"]                  \
            ])
    model.AddBoolOr(                                                        \
            [                                                               \
                person_dessert["James"]["Ice_Cream"].Not(),                 \
                person_drink["James"]["Coke"].Not()                         \
            ]).OnlyEnforceIf(person_dessert["Daniel"]["Chocolate_Cake"])
    model.AddBoolOr(                                                        \
            [                                                               \
                person_dessert["Daniel"]["Ice_Cream"].Not(),                \
                person_drink["Daniel"]["Coke"].Not()                        \
            ]).OnlyEnforceIf(person_dessert["James"]["Chocolate_Cake"])

    # The problem statement doesn't say so, but probably the two conditions
    # below are implicit. If the two conditions below are added,
    # then we get only 1 solution. 
    #
    # If they are discarded, we get multiple
    # solutions, which satisfy all other criteria, except that the same man
    # has both chocolate cake and coke.
    #
    # The man who has the chocolate cake doesn't have ice cream or coke
    # Since there is already one condition that someone cannot have two
    # desserts, we only need to cover for coke
    if CONSTRAINT9_INTERPRETATION_2:
        model.AddBoolAnd(                                                   \
                [                                                           \
                    person_drink["James"]["Coke"].Not()                     \
                ]).OnlyEnforceIf(person_dessert["James"]["Chocolate_Cake"])
        model.AddBoolAnd(                                                   \
                [                                                           \
                    person_drink["Daniel"]["Coke"].Not()                    \
                ]).OnlyEnforceIf(person_dessert["Daniel"]["Chocolate_Cake"])
    # Another way to arrive at a single solution (which incidentally is the
    # same, is to assume that the 'Not' is misplaced, and assume that
    # one man has chocolate Cake, and the other prefers to have Ice Cream
    # Or Coke but cannot have both
    # Since we've already added conditions that the men cannot have
    # Ice cream and Coke  both, we only need to add a condition that they
    # have either of them when the other man has chocolate cake.
    # Again, I'm not sure which of the three assumptions is correct
    elif CONSTRAINT9_INTERPRETATION_3:
        model.AddBoolOr(                                                    \
                [                                                           \
                    person_dessert["James"]["Ice_Cream"],                   \
                    person_drink["James"]["Coke"]                           \
                ]).OnlyEnforceIf(person_dessert["Daniel"]["Chocolate_Cake"])
        model.AddBoolOr(                                                    \
                [                                                           \
                    person_dessert["Daniel"]["Ice_Cream"],                  \
                    person_drink["Daniel"]["Coke"]                          \
                ]).OnlyEnforceIf(person_dessert["James"]["Chocolate_Cake"])


    solver = cp_model.CpSolver()
    status = solver.SearchForAllSolutions(model, solution_printer)
    print(solver.StatusName(status))

if "__main__" == __name__:
    tiramisu_main()
