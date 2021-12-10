#!/usr/bin/env python3

import ortools
import pandas as pd
from ortools.linear_solver import pywraplp

EPSILON = 0.000001

def get_element(df:pd.DataFrame, rowname:str, colname:str):
    selector = df['Unnamed: 0'] == rowname
    df = df[selector]
    return df[colname].to_numpy()[0]

class Task1():
    def __init__(self, excel_file_name:str):
        self.excel_file_name = excel_file_name

        self.supplier_stock_df = self.read_csv("Supplier stock")
        self.replace_nans(self.supplier_stock_df, 0)

        self.raw_material_cost_df = self.read_csv("Raw material costs")
        self.replace_nans(self.raw_material_cost_df, 'inf')

        self.raw_material_shipping_df = self.read_csv("Raw material shipping")
        self.replace_nans(self.raw_material_shipping_df, 'inf')

        self.product_requirements_df = self.read_csv("Product requirements")
        self.replace_nans(self.product_requirements_df, 0)

        self.production_capacity_df = self.read_csv("Production capacity")
        self.replace_nans(self.production_capacity_df, 0)

        self.production_cost_df = self.read_csv("Production cost")
        self.replace_nans(self.production_cost_df, 'inf')

        self.customer_demand_df = self.read_csv("Customer demand")
        self.replace_nans(self.customer_demand_df, 0)

        self.shipping_cost_df = self.read_csv("Shipping costs")
        self.replace_nans(self.shipping_cost_df, 'inf')
        
        self.product_names = [str(x) for x in \
            self.product_requirements_df['Unnamed: 0']]

        self.supplier_names = [str(x) for x in \
            self.supplier_stock_df['Unnamed: 0']]

        self.factory_names = [str(c) for c in \
            self.production_capacity_df.columns][1:]

        self.material_names = [str(c) for c in \
            self.product_requirements_df.columns][1:]

        self.customer_names = [str(c) for c in \
            self.customer_demand_df][1:]

        self.solver = pywraplp.Solver(\
                        'LPWrapper',\
                        pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

        self.cost_objective = self.solver.Objective()

        self.optimal_cost = float('nan')


        # 3D matrix, how many units of product p is supplied by factory f
        # to customer c
        # Indexing: [factory][customer][product]
        self.var_fcp = self.create_factory_customer_product_variables()

        # 3D matrix. How many units of material m is supplied by supplier s to
        # factory f
        # Indexing: [supplier][factory][material]
        self.var_sfm = self.create_supplier_factory_material_variables()

        # Sheet 1: Supplier Stock
        self.create_supplier_stock_constraints()

        # Sheet 2: Raw Materials Cost
        self.accumulate_raw_materials_cost()

        # Sheet 3: Raw Metrials Shipping
        self.accumulate_raw_materials_shipping_cost()

        # Sheet 4
        # Implicit, each factory should have all raw materials
        # to make all the products it makes
        # This involves a join between var_sfm and var_fcp
        # For each factory and material
        # Incoming >= Outgoing
        # OR
        # 0 <= Incoming - Outgoing <= INF
        self.create_constraint_meet_factory_requirements()

        # Sheet 5: production capacity
        self.create_production_capacity_constraints()

        # Sheet 6: Production Cost
        self.accumulate_production_cost()

        # Sheet 7: Customer demand
        # Create a constraint that all customer demands are met
        self.create_constraint_meet_customer_demands()

        # Sheet 8: Shipping Cost
        self.accumulate_shipping_cost()


    def replace_nans(self, df:pd.DataFrame, newValue:int):
        df.replace(float('nan'), float(newValue), inplace=True)

    def read_csv(self, sheet_name:str)->pd.DataFrame:
        df = pd.read_excel(self.excel_file_name, sheet_name=sheet_name)
        return df

    def create_factory_customer_product_variables(self):
        ret = {}
        for factory in self.factory_names:
            outer = {}
            for customer in self.customer_names:
                inner = {}
                for product in self.product_names:
                    varname = f"factory:{factory}-customer:{customer}" +\
                        f"-product:{product}"
                    variable = self.solver.NumVar(\
                        0, self.solver.infinity(), varname)
                    inner[product] = variable
                outer[customer] = inner
            ret[factory] = outer 
        return ret

    def create_supplier_factory_material_variables(self):
        ret = {}
        for supplier in self.supplier_names:
            outer = {}
            for factory in self.factory_names:
                inner = {}
                for material in self.material_names:
                    varname = f"supplier:{supplier}-factory:{factory}" +\
                        f"-material:{material}"
                    variable = self.solver.NumVar(\
                        0, self.solver.infinity(), varname)
                    inner[material] = variable
                outer[factory] = inner
            ret[supplier] = outer
        return ret

    # Sheet 1
    def create_supplier_stock_constraints(self):
        """ Create constraints for stocks each supplier has """
        
        def set_supplier_zero(supplier:str, material:str):
            for factory in self.factory_names:
                var = self.var_sfm[supplier][factory][material]
                constraint = self.solver.Constraint(0, 0)
                constraint.SetCoefficient(var, 1)

        def set_supplier_capacity(supplier:str, material:str, capacity:int):
            constraint = self.solver.Constraint(0, capacity)
            for factory in self.factory_names:
                var = self.var_sfm[supplier][factory][material]
                constraint.SetCoefficient(var, 1)

        for supplier in self.supplier_names:
            for material in self.material_names:
                capacity = get_element(\
                    self.supplier_stock_df, supplier, material)
                if 0 == capacity:
                    set_supplier_zero(supplier, material)
                else:
                    set_supplier_capacity(supplier, material, capacity)

    # Sheet 5
    def create_production_capacity_constraints(self):
        """ Create constraints for each product and  """
        def set_zero(factory:str, product:str):
            for customer in self.customer_names:
                var = self.var_fcp[factory][customer][product]
                constraint = self.solver.Constraint(0, 0)
                constraint.SetCoefficient(var, 1)

        def set_capacity(factory:str, product:str, capacity:int):
            constraint = self.solver.Constraint(0, capacity)
            for customer in self.customer_names:
                var = self.var_fcp[factory][customer][product]
                constraint.SetCoefficient(var, 1)

        for factory in self.factory_names:
            for product in self.product_names:
                capacity = get_element(\
                    self.production_capacity_df, product, factory)
                if 0 == capacity:
                    set_zero(factory, product)
                else:
                    set_capacity(factory, product, capacity)


    # Sheet 7
    def create_constraint_meet_customer_demands(self):
        """Ensure all customer demands are met"""

        def set_zero(customer:str, product:str):
            for factory in self.factory_names:
                var = self.var_fcp[factory][customer][product]
                constraint = self.solver.Constraint(0, self.solver.infinity())
                constraint.SetCoefficient(var, 1)

        def set_demand(customer:str, product:str, demand):
            constraint = self.solver.Constraint(demand, self.solver.infinity())
            for factory in self.factory_names:
                var = self.var_fcp[factory][customer][product]
                constraint.SetCoefficient(var, 1)

        for product in self.product_names:
            for customer in self.customer_names:
                demand = get_element(self.customer_demand_df, product, customer)
                if 0 == demand:
                    set_zero(customer, product)
                else:
                    set_demand(customer, product, demand)

    # Sheet 6
    def accumulate_production_cost(self):
        for factory in self.factory_names:
            for product in self.product_names:
                cost_per_unit = get_element(\
                    self.production_cost_df, product, factory)
                for customer in self.customer_names:
                    var = self.var_fcp[factory][customer][product]
                    self.cost_objective.SetCoefficient(\
                        var, float(cost_per_unit))

    # Sheet 8
    def accumulate_shipping_cost(self):
        for factory in self.factory_names:
            for customer in self.customer_names:
                shipping_cost_per_unit = get_element(\
                    self.shipping_cost_df, factory, customer)
                for product in self.product_names:
                    var = self.var_fcp[factory][customer][product]
                    self.cost_objective.SetCoefficient(\
                        var, float(shipping_cost_per_unit))

    # Sheet 2: Raw Materials Cost
    def accumulate_raw_materials_cost(self):
        for supplier in self.supplier_names:
            for material in self.material_names:
                material_cost = get_element(\
                    self.raw_material_cost_df, supplier, material)
                for factory in self.factory_names:
                    var = self.var_sfm[supplier][factory][material]
                    self.cost_objective.SetCoefficient(\
                        var, float(material_cost))

    # Sheet 3: Raw Metrials Shipping
    def accumulate_raw_materials_shipping_cost(self):
        for supplier in self.supplier_names:
            for factory in self.factory_names:
                shipping_cost = get_element(\
                    self.raw_material_shipping_df, supplier, factory)
                for material in self.material_names:
                    var = self.var_sfm[supplier][factory][material]
                    self.cost_objective.SetCoefficient(\
                        var, float(shipping_cost))

    # Sheet 4
    # For each factory and material
    # Incoming >= Outgoing
    # OR
    # 0 <= Incoming - Outgoing <= INF
    def create_constraint_meet_factory_requirements(self):
        for factory in self.factory_names:
            for material in self.material_names:
                constraint = self.solver.Constraint(0, self.solver.infinity())

                # Incoming constraints
                for supplier in self.supplier_names:
                    var = self.var_sfm[supplier][factory][material]
                    constraint.SetCoefficient(var, 1)

                for customer in self.customer_names:
                    # Outgoing constraints
                    for product in self.product_names:
                        material_per_unit = \
                            get_element(self.product_requirements_df,\
                                        product,\
                                        material)
                        material_per_unit = float(-1 * material_per_unit)
                        var = self.var_fcp[factory][customer][product]
                        constraint.SetCoefficient(var, material_per_unit)
                    
    def solve(self):
        self.solver.Solve()
        self.optimal_cost = self.cost_objective.Value()
        
    # Verify that the stock has not been exceeded for any supplier
    def verify_supplier_stock_not_exceeded(self):
        for supplier in self.supplier_names:
            for materal in self.material_names:
                accumulated = 0
                for factory in self.factory_names:
                    accumulated = accumulated +\
                        self.var_sfm[supplier][factory][materal].SolutionValue()
                max_value = get_element(self.supplier_stock_df,\
                                supplier, materal)
                assert(accumulated <= max_value + EPSILON)

    # Verify that all customer demands have been met
    def verify_customer_demands_met(self):
        for customer in self.customer_names:
            for product in self.product_names:

                accumulated = 0
                for factory in self.factory_names:
                    accumulated = accumulated +\
                        self.var_fcp[factory][customer][product].SolutionValue()
                demand = get_element(self.customer_demand_df, product, customer)
                assert(accumulated - demand > (-1 * EPSILON))

    # Verify that the production capacity has not been exceeded
    # for any factory
    def verify_production_capacity_not_exceeded(self):
        for factory in self.factory_names:
            for prod in self.product_names:
                accumulator = 0.0
                for customer in self.customer_names:
                    accumulator = accumulator +\
                        self.var_fcp[factory][customer][prod].SolutionValue()
                capacity = get_element(\
                            self.production_capacity_df, prod, factory)
                assert(accumulator <= capacity + EPSILON)


    # Verify that material requirements are satisfied for each
    # factory/material  combination based on its deliveries to its customers
    def verify_material_requirements_satisfied(self):
        for fact in self.factory_names:
            for mat in self.material_names:

                # calculate requirements
                required = 0
                for prod in self.product_names:
                    for cust in self.customer_names:
                        n_prods = self.var_fcp[fact][cust][prod].SolutionValue()
                        per_unit_requirement = get_element(\
                                                self.product_requirements_df,\
                                                prod, mat)
                        required = required + (n_prods * per_unit_requirement)

                # Calculate actual amount procured
                procured = 0
                for supp in self.supplier_names:
                    procured = procured +\
                        self.var_sfm[supp][fact][mat].SolutionValue()

                assert(required <= procured + EPSILON)

    def verify_solution(self):
        self.verify_supplier_stock_not_exceeded()
        self.verify_customer_demands_met()
        self.verify_production_capacity_not_exceeded()
        self.verify_material_requirements_satisfied()

    def print_supplier_factory_material(self):
        print()
        print("Printing Supplier Factory Orders")
        print('*' * len("Printing Supplier Factory Orders"))
        print()
        for fact in self.factory_names:
            print(fact)
            print('-' * len(fact))
            for supp in self.supplier_names:
                out_str = f"    {supp} - "
                for mat in self.material_names:
                    value = self.var_sfm[supp][fact][mat].SolutionValue()
                    if (0.0 != value):
                        out_str = out_str + "    "
                        temp = f"{mat:.15s}: {round(value,2):05.2f}"
                        out_str = out_str + f"{temp:23s}"
                print(out_str)
            print()

    def print_supplier_bill_for_each_factory(self):
        print()
        print("Printing supplier bill for factories")
        print('*' * len("Printing supplier bill for factories"))
        print()
        for fact in self.factory_names:
            print(fact)
            print('-' * len(fact))
            for supp in self.supplier_names:
                cost = 0.0
                for mat in self.material_names:
                    qty = self.var_sfm[supp][fact][mat].SolutionValue()
                    mat_cost = get_element(self.raw_material_cost_df, supp, mat)
                    shp_cost = get_element(self.raw_material_shipping_df,\
                                                supp, fact)
                    if qty >= EPSILON:
                        cost = cost + (mat_cost + shp_cost) * qty
                print(f"    - {supp:20s} : {round(cost, 2):10.2f}")
            print()

    def print_units_and_cost_per_factory(self):
        print()
        print("Printing production and cost for each factory")
        print('*' * len("Printing production and cost for each factory"))
        print()
        for fact in self.factory_names:
            print(fact)
            print('-' * len(fact))
            tot_cost = 0.0
            for prod in self.product_names:
                prod_qty = 0.0

                for cust in self.customer_names:
                    prod_qty = prod_qty +\
                        self.var_fcp[fact][cust][prod].SolutionValue()
                print(f"        {prod:10s}: {round(prod_qty,2):15.2f}")
                prod_cost = get_element(self.production_cost_df, prod, fact)

                if (prod_cost == float('inf') and prod_qty != 0):
                    # If a factory cannot produce an item, it's production
                    # quantity should actually be zero
                    assert(False)
                # Now that we have verified that factories only produce
                # items they can, we simplify the code by setting the
                # production cost to 0
                if (prod_cost == float('inf')): prod_cost = 0
                tot_cost = tot_cost + prod_qty * prod_cost
            print(f"    Total Manufacturing Cost = {round(tot_cost,2):05.2f}")
            print()

    def print_customer_factory_units_ship_cost(self):
        # For each customer, determine how many units are being shipped
        # from each factory, also the total shipping cost per customer
        print()
        print("Printing shipments for each customer")
        print('*' * len("Printing shipments for each customer"))
        print()
        for cust in self.customer_names:
            ship_cost = 0.0
            print(cust)
            print('-' * len(cust))
            for prod in self.product_names:
                out_str = ""
                out_str = out_str + f"Product {prod:15s} "
                for fact in self.factory_names:
                    qty = self.var_fcp[fact][cust][prod].SolutionValue()
                    if qty >= EPSILON:
                        ship_cost_unit = get_element(self.shipping_cost_df,\
                                                            fact, cust)
                        ship_cost = ship_cost + (qty * ship_cost_unit)
                        out_str = out_str + f"    {fact:.15s} : "
                        out_str = out_str +f"{round(qty,2):5.2f}    "
                print(out_str if out_str != "" else "\n")
            print(f"Total Shipping Cost: {round(ship_cost,2):5.2f}")
            print()

    def print_solution(self):
        print(f"Best cost found: {self.optimal_cost:.2f}")
        self.print_supplier_factory_material()
        self.print_supplier_bill_for_each_factory()
        self.print_units_and_cost_per_factory()
        self.print_customer_factory_units_ship_cost()

        # TODO: Start from Step N


def t1_main()->None:
    t1 = Task1("./Assignment_DA_2_Task_1_data.xlsx")
    t1.solve()
    t1.verify_solution()
    t1.print_solution()

    
if "__main__" == __name__:
    t1_main()
