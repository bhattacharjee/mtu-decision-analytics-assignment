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
            person,
            starter,
            maincourse,
            drink,
            dessert,
            person_starter,
            person_maincourse,
            person_drink,
            perrson_dessert):
        super().__init__(self)
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
        print(f"Solution #{self.solutions}")
        print("---------------------------------------------------------------")

        for person in self.person:
            print(f"- {person}")
            [print(f" - {color}") for color in self.color\
                    if self.Value(self.person_color[person][color])]
            [print(f" - {starter}") for starter in self.starter\
                    if self.Value(self.person_starter[person][starter])]
            [print(f" - {dessert}") for dessert in self.dessert\
                    if self.Value(self.person_dessert[person][dessert])]
            [print(f" - {drink}") for drink in self.drink\
                    if self.Value(self.person_drink[person][drink])]
            print()


