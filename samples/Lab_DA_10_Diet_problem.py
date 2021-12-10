import pandas as pd
from ortools.linear_solver import pywraplp

def diet():
    nutrition_contents = pd.DataFrame([[2,5,4,9],[4,3,1,7]], index=["Feed A","Feed B"], columns=["Energy","Protein","Calcium","Cost"])    
    print(nutrition_contents)
    print()
    
    nutrition_requirements = pd.Series([12,15,8], index=["Energy","Protein","Calcium"])    
    print(nutrition_requirements)
    print()

    feeds = set(nutrition_contents.index)
    print(feeds)

    requirements = set(nutrition_requirements.index)
    print(requirements)
    print()
    
    feeding_amount = {}
    
    MIP = False
    if MIP:
        solver = pywraplp.Solver('LPWrapper',
                                 pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        for feed in feeds:
            feeding_amount[feed] = solver.IntVar(0, solver.infinity(), feed)
    else:
        solver = pywraplp.Solver('LPWrapper', 
                             pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
        for feed in feeds:
            feeding_amount[feed] = solver.NumVar(0, solver.infinity(), feed)

        
    for requirement in requirements:
        c = solver.Constraint(float(nutrition_requirements[requirement]), solver.infinity())
        for feed in feeds:
            c.SetCoefficient(feeding_amount[feed], float(nutrition_contents[requirement][feed]))

    cost = solver.Objective()    
    for feed in feeds:
        cost.SetCoefficient(feeding_amount[feed], float(nutrition_contents["Cost"][feed]))
    cost.SetMinimization()
    solver.Solve()
    
    total_cost = 0
    for feed in feeds:
        print(feed, " -> ", feeding_amount[feed].solution_value())
        total_cost += feeding_amount[feed].solution_value()*nutrition_contents["Cost"][feed]
    print()                
    print("Total cost:", total_cost)
        

def main():
    diet()

main()
