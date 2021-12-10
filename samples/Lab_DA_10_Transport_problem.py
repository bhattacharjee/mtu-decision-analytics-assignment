import pandas as pd
from ortools.linear_solver import pywraplp

def transport():
    transmission_costs = pd.DataFrame([[5,5,3],[6,4,1]],index=["Supplier A","Supplier B"], columns=["Consumer A","Consumer B", "Consumer C"])        
    print(transmission_costs)
    print()

    supply = pd.Series([6,9],index=["Supplier A","Supplier B"])
    print(supply)
    print()

    demand = pd.Series([8,5,2],index=["Consumer A","Consumer B", "Consumer C"])
    print(demand)
    print()

        
    suppliers = set(transmission_costs.index)
    consumers = set(transmission_costs.columns)
    
    print(suppliers)
    print(consumers)
    print()
    
    solver = pywraplp.Solver('LPWrapper', 
                             pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

    delivery = {}    
    for supplier in suppliers:
        for consumer in consumers:
            delivery[(supplier,consumer)] = solver.NumVar(0,solver.infinity(),supplier+"_" +consumer)


    # each supplier needs to supply all their energy
    for supplier in suppliers:
        c = solver.Constraint(float(supply[supplier]), solver.infinity())
        for consumer in consumers:
            c.SetCoefficient(delivery[(supplier,consumer)], 1)
            

    # each each consumer needs to have their demand met
    for consumer in consumers:
        c = solver.Constraint(float(demand[consumer]), solver.infinity())
        for supplier in suppliers:
            c.SetCoefficient(delivery[(supplier,consumer)], 1)

    cost = solver.Objective()
    for supplier in suppliers:
        for consumer in consumers:
            cost.SetCoefficient(delivery[(supplier,consumer)], float(transmission_costs[consumer][supplier]))
    cost.SetMinimization()
    solver.Solve()

    total_cost = 0
    for supplier in suppliers:
        for consumer in consumers:
            if delivery[(supplier,consumer)].solution_value()>0:
                print("Delivery from",supplier,"to",consumer,"is",delivery[(supplier,consumer)].solution_value())
                total_cost += delivery[(supplier,consumer)].solution_value()*transmission_costs[consumer][supplier]
    print()
    print("Total cost:",total_cost)


def main():
    transport()
    
main()
