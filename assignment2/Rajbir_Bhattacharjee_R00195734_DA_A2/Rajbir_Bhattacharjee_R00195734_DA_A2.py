#!/usr/bin/env python3

import ortools
import pandas as pd
from ortools.linear_solver import pywraplp
from functools import lru_cache

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
        self.cost_objective.SetMinimization()

        self.optimal_cost = float('nan')


        # 3D matrix, how many units of product p is supplied by factory f
        # to customer c
        # Indexing: [factory][customer][product]
        self.var_fcp = self.create_factory_customer_product_variables()

        # Setting the coefficients is done in two steps , hence the coefficients
        # must be accumulated before assigning to the variables
        # We use another 3D matrix of floats to accumulate the coefficients
        # before assigning them
        self.coeff_fcp = self.create_factory_customer_product_coefficients()

        # 3D matrix. How many units of material m is supplied by supplier s to
        # factory f
        # Indexing: [supplier][factory][material]
        self.var_sfm = self.create_supplier_factory_material_variables()

        # Setting the coefficients is done in two steps , hence the coefficients
        # must be accumulated before assigning to the variables
        # We use another 3D matrix of floats to accumulate the coefficients
        # before assigning them
        self.coeff_sfm = self.create_supplier_factory_material_coefficients()

        # Sheet 1: Supplier Stock
        self.create_supplier_stock_constraints()

        # Sheet 2: Raw Materials Cost
        self.accumulate_raw_materials_cost()

        # Sheet 3: Raw Metrials Shipping
        self.accumulate_raw_materials_shipping_cost()

        # Sheet 8: Shipping Cost
        self.accumulate_shipping_cost()

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


        self.set_objective_coefficients()

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

    def create_factory_customer_product_coefficients(self):
        ret = {}
        for factory in self.factory_names:
            outer = {}
            for customer in self.customer_names:
                inner = {}
                for product in self.product_names:
                    inner[product] = 0.0
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

    def create_supplier_factory_material_coefficients(self):
        ret = {}
        for supplier in self.supplier_names:
            outer = {}
            for factory in self.factory_names:
                inner = {}
                for material in self.material_names:
                    inner[material] = 0.0
                outer[factory] = inner
            ret[supplier] = outer
        return ret

    # Sheet 1
    def create_supplier_stock_constraints(self):
        # Create constraints for stocks each supplier has
        # (fixed values as per excel)
        
        def set_supplier_zero(supplier:str, material:str):
            # If a supplier doesn't have a material as per the excel
            # force it to be zero
            for factory in self.factory_names:
                var = self.var_sfm[supplier][factory][material]
                constraint = self.solver.Constraint(0, 0)
                constraint.SetCoefficient(var, 1.0)

        def set_supplier_capacity(supplier:str, material:str, capacity:int):
            # if a supplier has a x amount of a material as per the excel
            # force it to be that value
            constraint = self.solver.Constraint(0, capacity)
            for factory in self.factory_names:
                var = self.var_sfm[supplier][factory][material]
                constraint.SetCoefficient(var, 1.0)

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
        # Create constraints for production capacity for each factory
        # (as per excel sheet, fixed values)

        def set_zero(factory:str, product:str):
            for customer in self.customer_names:
                var = self.var_fcp[factory][customer][product]
                constraint = self.solver.Constraint(0, 0)
                constraint.SetCoefficient(var, 1.0)

        def set_capacity(factory:str, product:str, capacity:int):
            constraint = self.solver.Constraint(0, capacity)
            for customer in self.customer_names:
                var = self.var_fcp[factory][customer][product]
                constraint.SetCoefficient(var, 1.0)

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
        # Get the demand for each customer from the excel sheet
        # Then loop over each factory, and ensure that the total
        # from all factories to that customer meets the demand

        def set_zero(customer:str, product:str):
            for factory in self.factory_names:
                var = self.var_fcp[factory][customer][product]
                constraint = self.solver.Constraint(0, self.solver.infinity())
                constraint.SetCoefficient(var, 1.0)

        def set_demand(customer:str, product:str, demand):
            constraint = self.solver.Constraint(demand, self.solver.infinity())
            for factory in self.factory_names:
                var = self.var_fcp[factory][customer][product]
                constraint.SetCoefficient(var, 1.0)

        for product in self.product_names:
            for customer in self.customer_names:
                demand = get_element(self.customer_demand_df, product, customer)
                if 0 == demand:
                    set_zero(customer, product)
                else:
                    set_demand(customer, product, demand)

    # Sheet 6
    def accumulate_production_cost(self):
        # coeff_fcp is an accumulator of production cost, shipping cost,
        # material cost, etc.
        # In this function, add the production cost for each
        # factory, customer, product
        for factory in self.factory_names:
            for product in self.product_names:
                cost_per_unit = get_element(\
                    self.production_cost_df, product, factory)
                cost_per_unit = float(cost_per_unit)
                if cost_per_unit == float('inf'): cost_per_unit = 0.0
                for customer in self.customer_names:
                    self.coeff_fcp[factory][customer][product] += cost_per_unit

    # Sheet 8
    def accumulate_shipping_cost(self):
        # coeff_fcp is an accumulator of production cost and shipping cost
        # In this function, add the shipping cost for each
        # factory, customer, product
        for factory in self.factory_names:
            for customer in self.customer_names:
                shipping_cost_per_unit = get_element(\
                    self.shipping_cost_df, factory, customer)
                shipping_cost_per_unit = float(shipping_cost_per_unit)
                if shipping_cost_per_unit == float('inf'):
                    shipping_cost_per_unit = 0.0
                for product in self.product_names:
                    self.coeff_fcp[factory][customer][product] += \
                        shipping_cost_per_unit

    # Sheet 2: Raw Materials Cost
    def accumulate_raw_materials_cost(self):
        # coeff_sfm is an accumulator for raw materials cost, and raw materials
        # shipping cost.
        # In this function accumulate the raw materials cost for each
        # supplier, material and factory
        for supplier in self.supplier_names:
            for material in self.material_names:
                material_cost = get_element(\
                    self.raw_material_cost_df, supplier, material)
                material_cost = float(material_cost)
                if material_cost == float('inf'): material_cost = 0.0
                for factory in self.factory_names:
                    self.coeff_sfm[supplier][factory][material] += \
                        material_cost

    # Sheet 3: Raw Metrials Shipping
    def accumulate_raw_materials_shipping_cost(self):
        # coeff_sfm is an accumulator for raw materials cost, and raw materials
        # shipping cost.
        # In this function accumulate the raw materials shipping cost for each
        # supplier, material and factory
        for supplier in self.supplier_names:
            for factory in self.factory_names:
                shipping_cost = get_element(\
                    self.raw_material_shipping_df, supplier, factory)
                shipping_cost = float(shipping_cost)
                if shipping_cost == float('inf'): shipping_cost = 0.0
                for material in self.material_names:
                    self.coeff_sfm[supplier][factory][material] += shipping_cost

    def set_objective_coefficients(self):
        # Total cost is a sum of
        # 1. production cost
        # 2. shipping cost
        # 3. raw materials cost
        # 4. Raw materials shipping cost

        # This loop adds up production cost and shipping cost
        # The two have already been added up and stored in coeff_fcp
        for prod in self.product_names:
            for fact in self.factory_names:
                for cust in self.customer_names:
                    var = self.var_fcp[fact][cust][prod]
                    val = self.coeff_fcp[fact][cust][prod]
                    self.cost_objective.SetCoefficient(var, val)

        # This loop adds up raw materials cost and raw materials shipping cost
        # The two have already been added up and stored in coeff_sfm
        for fact in self.factory_names:
            for supp in self.supplier_names:
                for mat in self.material_names:
                    var = self.var_sfm[supp][fact][mat]
                    val = self.coeff_sfm[supp][fact][mat]
                    self.cost_objective.SetCoefficient(var, val)
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
                    constraint.SetCoefficient(var, 1.0)

                for customer in self.customer_names:
                    # Outgoing constraints
                    for product in self.product_names:
                        material_per_unit = \
                            get_element(self.product_requirements_df,\
                                        product,\
                                        material)
                        material_per_unit = float(-1 * material_per_unit)
                        var = self.var_fcp[factory][customer][product]
                        constraint.SetCoefficient(var, float(material_per_unit))
                    
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
            print()
            print(f"Total Shipping Cost: {round(ship_cost,2):5.2f}")
            print()
        print()

    @lru_cache(maxsize=128)
    def average_material_cost_for_factory(self, fact, mat):
        """ What is the average price of a material for a factory, including
            material shipping proce"""
        tot_cost = 0.0
        tot_qty = 0.0
        for supp in self.supplier_names:
            qty = self.var_sfm[supp][fact][mat].SolutionValue()
            if qty >= EPSILON:
                tot_qty += qty
                mat_price = \
                    get_element(self.raw_material_cost_df, supp, mat)
                tot_cost += (mat_price * qty)
                ship_price = \
                    get_element(self.raw_material_shipping_df, supp, fact)
                tot_cost += (ship_price * qty)
        retval = tot_cost / tot_qty if tot_qty >= 0.0 else 0.0
        #print(f"{fact}     {mat}    Avg Price: {retval}")
        return retval

    @lru_cache(maxsize=128)
    def average_product_cost_for_factory(self, fact, prod):
        """ Average production cost for a product for each factory,
            does not include shipping cost for the finished product"""
        cost = 0.0
        capacity = get_element(self.production_capacity_df, prod, fact)
        if (capacity == 0):
            print(f"{fact} cannot produce {prod}")
            return 0

        # Get the material cost for making this product (including shipping)
        for mat in self.material_names:
            qty = get_element(self.product_requirements_df, prod, mat)
            if qty >= 0.0:
                cost += (qty * \
                    self.average_material_cost_for_factory(fact, mat))

        # Add the production cost
        production_cost = get_element(self.production_cost_df, prod, fact)
        if production_cost != float('inf') and production_cost != float('nan'):
            cost += production_cost

        return cost

    @lru_cache(maxsize=128)
    def shipping_cost_factory_customer(self, fact, cust):
        return get_element(self.shipping_cost_df, fact, cust)

    def print_unit_product_cost_per_customer(self):

        print()
        print("Printing product unit cost per customer")
        print('*' * len("Printing product unit cost per customer"))
        print()

        all_costs = 0.0
        all_qty = 0.0

        for cust in self.customer_names:
            # 2D map for materials used
            # mat_used[factory][material] = <qty used for this customer>
            print(cust)
            print('-' * len(cust))

            for prod in self.product_names:
                qty_acc = 0.0
                cost_acc = 0.0

                for fact in self.factory_names:
                    qty = self.var_fcp[fact][cust][prod].SolutionValue()
                    if qty >= EPSILON:
                        cost = self.average_product_cost_for_factory(fact, prod)
                        ship = self.shipping_cost_factory_customer(fact, cust)
                        cost_acc += (cost * qty)
                        cost_acc += (ship * qty)
                        qty_acc += qty

                avg_prod_cost = cost_acc / qty_acc if qty_acc > 0.0 else 0.0
                if qty_acc == 0.0: assert(cost_acc == 0.0)
                if qty_acc != 0.0:
                    print(f"        {prod} : AVG COST: {avg_prod_cost:8.2f}")

                all_costs += cost_acc
                all_qty += qty_acc

        print()
        print("Total_cost ", all_costs, all_qty)


    def get_factory_customer_material_fraction(self, fact, cust, mat):
        # Determine for each customer the fraction of each material each
        # factory has to order for manufacturing products delivered to that
        # particular customer
        
        def get_total_factory_material():
            total = 0.0
            for supp in self.supplier_names:
                value = self.var_sfm[supp][fact][mat].SolutionValue()
                if value > 0.0 and value != float('inf') \
                    and value != float('nan'):
                    total += float(value)
            return total

        def get_product_material_requirements(prod):
            value = get_element(self.product_requirements_df, prod, mat)
            value = float(value)
            if value == float('inf') or value == float('nan'):
                value = 0.0
            return value

        def get_total_customer_material():
            total = 0.0
            for prod in self.product_names:
                req_pu = get_product_material_requirements(prod)
                req_pu = float(req_pu)
                if req_pu == float('inf') or req_pu == float('nan'):
                    req_pu = 0.0
                qty = self.var_fcp[fact][cust][prod].SolutionValue()
                qty = float(qty)
                if qty == float('inf') or qty == float('nan'):
                    qty = 0.0
                total += (qty * req_pu)
            return total
                
        total_mat = get_total_factory_material()
        cust_mat = get_total_customer_material()
        if total_mat == 0.0: assert(cust_mat == 0.0)
        return cust_mat / total_mat if cust_mat > 0.0 else 0.0

    def print_factory_customer_material_fraction(self):
        print()
        print("Printing what fraction of material is used for each customer")
        print('*' * \
            len("Printing what fraction of material is used for each customer"))
        print()
        for fact in self.factory_names:
            print(f'{fact}')
            print('-' * len(f'{fact}'))
            for cust in self.customer_names:
                outstr = "    -- "
                outstr = outstr + f"{cust} :    "
                for mat in self.material_names:
                    frac = self.get_factory_customer_material_fraction(\
                                fact, cust, mat)
                    outstr = outstr + f"{mat} : {frac:5.2f}    "
                print(outstr)

    def print_solution(self):
        print(f"Best cost found: {self.optimal_cost:.2f}")
        self.print_supplier_factory_material()
        self.print_supplier_bill_for_each_factory()
        self.print_units_and_cost_per_factory()
        self.print_customer_factory_units_ship_cost()
        self.print_unit_product_cost_per_customer()
        self.print_factory_customer_material_fraction()
        print()
        print()
        print(f"TOTAL COST = {self.optimal_cost:12.2f}")


    def main(self):
        self.solve()
        self.verify_solution()
        self.print_solution()

    
import pandas as pd
import ortools
from ortools.linear_solver import pywraplp
from functools import lru_cache

FULL_TSP = False

def get_element(df:pd.DataFrame, rowname:str, colname:str):
    selector = df['Unnamed: 0'] == rowname
    df = df[selector]
    return df[colname].to_numpy()[0]

class Task2:
    def __init__(self, excel_file_name, start_city_name):
        self.excel_file_name = excel_file_name
        self.start_city_name = start_city_name

        self.distances_df = self.read_excel("Distances")

        self.cities_must_visit = \
            ["Dublin", "Limerick", "Waterford", "Galway", "Wexford", \
            "Belfast", "Athlone", "Rosslare", "Wicklow"]

        if start_city_name not in self.cities_must_visit:
            self.cities_must_visit.append(start_city_name)

        self.solver = pywraplp.Solver(\
                        'LPWrapper',\
                        pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

        # Get all city names
        self.city_names = [str(x) for x in self.distances_df.columns][1:]

        # Map from city to its position in the dataframe
        self.city_numbers = {c: n for n, c in enumerate(self.city_names)}

        self.num_cities = len(self.city_names)

        # Index of the starting city, this will be helpful later
        self.start_city_ind = self.city_numbers[self.start_city_name]

        # Create a 2-D array of variables. 
        # if x[i][j] = True then there is an ege from city[i] to city[j]
        self.var_edges = self.create_edges()

        # We use additional variables, each having values from 1 to N
        # if city[k] is the i'th city to be visited, then city[k] = i
        # This will help us to avoid loops
        self.var_order = self.create_order_variables()

        self.objective = self.solver.Objective()

        self.optimal_distance = None

        self.objective.SetMinimization()

        self.set_objective_coefficients()

        if FULL_TSP:
            self.create_constraint_all_towns_visited_only_once()
        else:
            self.create_constraint_all_towns_visited_max_once()

        self.create_constraints_no_loops_to_same_city()

        self.create_constraint_no_complete_subroutes()

        if not FULL_TSP:
            self.create_constraint_must_visit_cities()


    def read_excel(self, sheet_name:str):
        df = pd.read_excel(self.excel_file_name, sheet_name=sheet_name)
        return df

    def create_edges(self):
        outer = {}
        for c1 in self.city_names:
            inner = {}
            for c2 in self.city_names:
                varname = f"{c1:>10s} ---> {c2:10s}"
                inner[c2] = self.solver.IntVar(0, 1, varname)
            outer[c1] = inner
        return outer

    # We use additional variables, each having values from 1 to N
    # if city[k] is the i'th city to be visited, then city[k] = i
    # This will help us to avoid loops
    def create_order_variables(self):
        return [self.solver.IntVar(1, len(self.city_names), f"{n}-{c}") \
            for n, c in enumerate(self.city_names)]

    def create_constraint_all_towns_visited_only_once(self):
        # This function is unused by default, it is useful 
        # in the full TSP version of the problem (off by default)
        for c1 in self.city_names:
            cons1 = self.solver.Constraint(1, 1)
            cons2 = self.solver.Constraint(1, 1)
            for c2 in self.city_names:
                var1 = self.var_edges[c1][c2]
                cons1.SetCoefficient(var1, 1)
                var2 = self.var_edges[c2][c1]
                cons2.SetCoefficient(var2, 1)

    def create_constraint_all_towns_visited_max_once(self):
        for c1 in self.city_names:
            cons1 = self.solver.Constraint(0, 1)
            cons2 = self.solver.Constraint(0, 1)
            for c2 in self.city_names:
                var1 = self.var_edges[c1][c2]
                cons1.SetCoefficient(var1, 1)
                var2 = self.var_edges[c2][c1]
                cons2.SetCoefficient(var2, 1)

        # A town entered must also be exited
        for c1 in self.city_names:
            cons = self.solver.Constraint(0, 0)
            for c2 in self.city_names:
                var1 = self.var_edges[c1][c2]
                cons.SetCoefficient(var1, -1)
                var2 = self.var_edges[c2][c1]
                cons.SetCoefficient(var2, 1)

    def create_constraint_must_visit_cities(self):
        # Every city in the must-visit list must be visited
        for city in self.cities_must_visit:
            cons1 = self.solver.Constraint(1, self.solver.infinity())
            cons2 = self.solver.Constraint(1, self.solver.infinity())
            for c2 in self.city_names:
                if c2 != city:
                    var1 = self.var_edges[city][c2]
                    cons1.SetCoefficient(var1, 1)
                    var2 = self.var_edges[c2][city]
                    cons2.SetCoefficient(var2, 1)

    def create_constraints_no_loops_to_same_city(self):
        # No zero length loops from same city to itself
        for c in self.city_names:
            var = self.var_edges[c][c]
            constraint = self.solver.Constraint(0, 0)
            constraint.SetCoefficient(var, 1)

    def create_constraint_no_complete_subroutes(self):
        # There should be no complete sub-routes.
        # This will be achieved by self.var_order
        # Each city is given a number in the order in which it is visited
        # There cannot be an edge from a city later in the route to a
        # city earlier in the route, except for the first and last city
        
        # order of first city is 1
        constraint = self.solver.Constraint(1, 1)
        constraint.SetCoefficient(self.var_order[self.start_city_ind], 1)

        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != self.start_city_ind and j != self.start_city_ind:
                    cons = self.solver.Constraint(\
                        2 - self.num_cities, self.solver.infinity())
                    c1_name = self.city_names[i]
                    c2_name = self.city_names[j]
                    var_edge = self.var_edges[c1_name][c2_name]
                    cons.SetCoefficient(var_edge, 1 - self.num_cities)
                    cons.SetCoefficient(self.var_order[i], -1)
                    cons.SetCoefficient(self.var_order[j], 1)

    def get_next_city(self, start):
        for c2 in self.city_names:
            if 1 == self.var_edges[start][c2].SolutionValue():
                dist = get_element(self.distances_df, start, c2)
                return c2, dist
        assert(False) # Should not reach here

    def print_route(self):
        next_city = None
        current_city = self.start_city_name
        
        n = 0

        while self.start_city_name != next_city:
            n += 1
            if next_city != None:
                current_city = next_city
            next_city, dist = self.get_next_city(current_city)
            print(f"{n:>3d}. {current_city:>10s} ---> "\
                + f"{next_city:10s} -- {dist:>10d}")

    def set_objective_coefficients(self):
        for c1 in self.city_names:
            for c2 in self.city_names:
                var = self.var_edges[c1][c2]
                dist = get_element(self.distances_df, c1, c2)
                self.objective.SetCoefficient(var, float(dist))

    def solve(self):
        self.solver.Solve()
        self.optimal_distance = self.objective.Value()

    def validate_solution_full_tsp(self):
        visited = [False for city in self.city_names]
        visited[self.start_city_ind] = True
        next_city = None
        current_city = self.start_city_name
        dist_accumulator = 0.0

        while self.start_city_name != next_city:
            if next_city != None:
                current_city = next_city
            next_city, dist = self.get_next_city(current_city)
            dist_accumulator += dist
            next_city_index = self.city_numbers[next_city]
            visited[next_city_index] = True

        assert(all(visited))
        assert(dist_accumulator == self.optimal_distance)

    def validate_solution(self):
        visited = []
        visited.append(self.start_city_name)
        next_city = None
        current_city = self.start_city_name
        dist_accumulator = 0.0

        while self.start_city_name != next_city:
            if next_city != None:
                current_city = next_city
            next_city, dist = self.get_next_city(current_city)
            dist_accumulator += dist
            next_city_index = self.city_numbers[next_city]
            if next_city != self.start_city_name:
                visited.append(next_city)

        assert(len(visited) == len(self.cities_must_visit))
        assert(dist_accumulator == self.optimal_distance)
        for city in self.cities_must_visit:
            assert(city in visited)

    def print_all_variables(self):
        for c1 in self.city_names:
            for c2 in self.city_names:
                val = self.var_edges[c1][c2].SolutionValue() 
                if val != 0:
                    print(f"edge({c1}, {c2}) = {val}")

    def main(self):
        self.solve()
        # self.print_all_variables()
        if FULL_TSP:
            self.validate_solution_full_tsp()
        else:
            self.validate_solution()
        self.print_route()
        print()
        print(f"Optimal distance: {self.optimal_distance}")


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import ortools
import pandas as pd
from ortools.linear_solver import pywraplp
from functools import lru_cache
import math
import numpy as np
import random

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
        # if line is circular return True, else False
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
        # Get the total distance between the first and the last station
        # of any line. For circular lines, this is the round trip time
        # to the same station
        stations = self.get_line_stations(line)
        stations = pair_array(stations, self.is_line_circular(line))
        total = 0
        for x, y in stations:
            total += get_element(self.distance_df, x, y)
        # print(f"Line Length: {line} --> {total}")
        return total

    @lru_cache(maxsize=8)
    def get_round_trip_time(self, line:str)->float:
        # For any line, return the round-trip time for the train
        dist = self.get_total_line_distance(line)
        dist = float(dist) if self.is_line_circular(line) else float(dist * 2)
        # print(f"RTT: {line} --> {dist}")
        return dist

    @lru_cache(maxsize=512)
    def get_line_for_leg(self, stop1:str, stop2:str)->list:
        # Given two adjacent stations, it returns the list of the lines
        # lines connecting the stations
        line_arr = []
        for line in self.line_names:
            stations = self.get_line_stations(line)
            legs = pair_array(stations, self.is_line_circular(line))
            for x, y in legs:
                if (x == stop1 and y == stop2) or (x == stop2 and y == stop1):
                    line_arr.append(line)
        return line_arr

    

class ShortestPath(TrainBase):
    def __init__(self, excel_file_name, source, destination):
        super().__init__(excel_file_name)

        self.solver_invoked = False

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
        # Decision variables if an edge from x to y is taken
        #    Two additional housekeeping things are done using constraints here:
        #    - An edge from a node to itself is always zero
        #    - If an edge between two nodes doesn't exist, it is always zero
        #
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

    def solve(self):
        if not self.solver_invoked:
            self.solver.Solve()
        self.solver_invoked = True
            
    def get_shortest_path(self):
        # Returns distance, [nodestart, node1, node2, ..., nodeend]
        self.solve()
        shortest_path = self.get_path(self.source, self.destination)
        stations_in_leg = {}
        for x, y in pair_array(shortest_path):
            stations_in_leg[(x,y,)] = self.get_line_for_leg(x, y)
        return self.objective.Value(), shortest_path, stations_in_leg

class TrainCapacity(TrainBase):
    def __init__(self, excel_file_name):
        super().__init__(excel_file_name)

        self.solver_invoked = False

        # This is a 2D array which specifies how many people travel
        # between every pair of directly connected stations
        # If stations are not directly connected, this should be zero
        self.capacity_required = self.initialize_capacity_matrix()

        self.solver = pywraplp.Solver(\
                        'LPWrapper',\
                        pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

        self.shortest_path_samples = list()

        # We will treat upstream and downstream capacity separately
        # This will give some added flexibility rather than just 
        # assuming that trains upstream and downstream are the same
        #
        # This also helps in keeping the code common
        #
        # For non-circular lines, these values for upstream and downstream
        # will be the same. We will force them by our variables and constraints
        # rather than changing the numbers here
        #
        # --------------------------------------------------------------------
        # VARIABLE: frequency of trains required on each line (per hour)
        # --------------------------------------------------------------------
        self.var_upstream_trains_per_hour = {}
        self.var_downstream_trains_per_hour = {}
        for line in self.line_names:
            self.var_upstream_trains_per_hour[line] = self.solver.IntVar(\
                            0, self.solver.infinity(), f"n_trains_up({line})")
            self.var_downstream_trains_per_hour[line] = self.solver.IntVar(\
                            0, self.solver.infinity(), f"n_trains_dn({line})")

        # We will treat upstream and downstream capacity separately
        # This will give some added flexibility rather than just 
        # assuming that trains upstream and downstream are the same
        #
        # This also helps in keeping the code common
        #
        # For non-circular lines, these values for upstream and downstream
        # will be the same. We will force them by our variables and constraints
        # rather than changing the numbers here
        #
        # Variables, number of trains required on each line
        # For non-circular lines, upstream and downstream trains have
        # the same number as the same trains run back and forth
        # This is specified in the last condition in 
        # constrain_link_trains_per_hour_and_total_trains
        #
        # These variables are simply the following equation:
        #
        # n_trains >= frequency * round_trip_time
        #
        # --------------------------------------------------------------------
        # VARIABLE: Total number of trains on a line 
        # --------------------------------------------------------------------
        self.var_trains_on_line_up = {}
        self.var_trains_on_line_down = {}
        for line in self.line_names:
            self.var_trains_on_line_up[line] = self.solver.IntVar(\
                0, self.solver.infinity(), f"uptrains-on-line({line})")
            self.var_trains_on_line_down[line] = self.solver.IntVar(\
                0, self.solver.infinity(), f"downtrains-on-line({line})")

        self.objective = self.solver.Objective()

        # We will treat upstream and downstream capacity separately
        # This will give some added flexibility rather than just 
        # assuming that trains upstream and downstream are the same
        #
        # This also helps in keeping the code common
        #
        # For non-circular lines, these values for upstream and downstream
        # will be the same. We will force them by our variables and constraints
        # rather than changing the numbers here
        #
        # This matrix denotes the capacity for each line, that is
        # how many passengers can a single train carry from A to B in line L
        # where A and B are adjacent stations in line L
        #
        # For non-circular lines, one of upstream or downstream from A to B
        #
        # Note: these are not variables
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

        self.constrain_link_trains_per_hour_and_total_trains()

        self.set_objective_coefficients()

        self.add_all_requirements()

        # Add a constraint that each leg has enough capacity
        # to handle its traffic
        self.add_constraints_no_leg_overloads()


    def constrain_link_trains_per_hour_and_total_trains(self):
        for line in self.line_names:
            rtt = self.get_round_trip_time(line) / 60.0

            # var_upstream_trains_per_hour is the frequency of trains required
            # Therefore:
            # Number of trains required >= frequency * round_trip_time
            constrain = self.solver.Constraint(0, self.solver.infinity())
            up_per_hour_var = self.var_upstream_trains_per_hour[line]
            var_n_trains_up = self.var_trains_on_line_up[line]
            constrain.SetCoefficient(var_n_trains_up, 1)
            constrain.SetCoefficient(up_per_hour_var, -rtt)

            # var_downstream_trains_per_hour is the frequency of trains required
            # Therefore:
            # Number of trains required >= frequency * round_trip_time
            constrain2 = self.solver.Constraint(0, self.solver.infinity())
            down_per_hour_var = self.var_downstream_trains_per_hour[line]
            var_n_trains_down = self.var_trains_on_line_down[line]
            constrain2.SetCoefficient(var_n_trains_down, 1)
            constrain2.SetCoefficient(down_per_hour_var, -rtt)

            # For non-circular routes, the same train travels back and forth
            # hence the same number of trains upstream and downstream
            if not self.is_line_circular(line):
                constrain3 = self.solver.Constraint(0, 0)
                var = self.var_trains_on_line_up[line]
                var2 = self.var_trains_on_line_down[line]
                constrain3.SetCoefficient(var, 1)
                constrain3.SetCoefficient(var2, -1)

    def set_objective_coefficients(self):
        for line in self.line_names:
            var = self.var_trains_on_line_up[line]
            self.objective.SetCoefficient(var, 1)

            # Only add upstream and downstream to the objective for non-circular
            # routes. For circular routes, the same train runs upstream
            # and downstream, so we need to add only one
            if self.is_line_circular(line):
                var = self.var_trains_on_line_down[line]
                self.objective.SetCoefficient(var, 1)
        self.objective.SetMinimization()


    def initialize_capacity_matrix(self):
        outer = {}
        for s1 in self.stop_names:
            outer[s1] = {s: 0 for s in self.stop_names}
        return outer

    def save_shortest_path(self, source, destination, dist, route, lines_in_leg):
        # Save a sample of the shortest paths calculated for printing
        if random.random() <= 0.9:
            return
        t = (source, destination, dist, route, lines_in_leg,)
        self.shortest_path_samples.append(t)

    def print_shortest_paths(self):
        for source, destination, dist, route, lines_in_leg \
            in self.shortest_path_samples:
            print(f"DISTANCE({source} --> {destination}) = {int(dist):>3d}" +\
                    f"           {route}")
            print(f"                                  {lines_in_leg}")
            print()

    def add_requirements(self, source, destination):
        # For a source and destination, find the number of passengers
        # traveling. Calculate the shortest path, and add the same
        # number of passengers to each leg of the route
        # Basically finds the shortest path, and then adds the traffic between
        # the source and destination to each leg of the path
        if source == destination:
            return
        req = int(get_element(self.passenger_df, source, destination))
        dist, route, lines_in_leg = \
            ShortestPath(self.excel_file_name, source, destination)\
                .get_shortest_path()
        self.save_shortest_path(source, destination, dist, route, lines_in_leg)
        route = pair_array(route)
        for x, y in route:
            self.capacity_required[x][y] += req

    def add_all_requirements(self):
        # Add the requirements for each pair of adjacent stations
        # Basically finds the shortest path, and then adds the traffic between
        # the source and destination to each leg of the path
        description = "Finding shortest paths"
        pairs = [(s1, s2,) for s1 in self.stop_names for s2 in self.stop_names]
        for s1, s2 in tqdm(pairs, desc=description):
            self.add_requirements(s1, s2)

    def fill_line_capacity_per_train(self, line:str):
        # This matrix denotes the capacity for each line, that is
        # how many passengers can a single train carry from A to B in line L
        # where A and B are adjacent stations in line L
        #
        # For non-circular lines, one of upstream or downstream from A to B
        # will be 0
        capacity = get_element(self.train_capacity_df, line, 'Capacity')
        if self.is_line_circular(line):
            for x, y in self.station_pairs(line):
                self.line_capacity_per_train_up[line][x][y] = capacity
            for x, y in self.station_pairs(line, downstream=True):
                self.line_capacity_per_train_down[line][x][y] = capacity
        else:
            upstream_stations = self.get_line_stations(line)
            upstream_stations = pair_array(upstream_stations)
            downstream_stations = self.get_line_stations(line)[::-1]
            downstream_stations = pair_array(downstream_stations)
            for x, y in upstream_stations:
                self.line_capacity_per_train_up[line][x][y] = capacity
            for x, y in downstream_stations:
                self.line_capacity_per_train_down[line][x][y] = capacity

    def add_constraints_no_leg_overloads(self):
        # Ensures that between each pair of connected stations,
        # there are enough trains to carry people to meet the demand
        #
        # The demand has already been accumulated
        # if A -> B -> C, and we are considering B -> C,
        # both B -> C and A -> C have already been added
        #
        # Also A -> A = 0, we could have added an if condition to remove this
        # constraint, but its just easier to let it be
        def constrain(source, dest):
            # Requirement(A to B) <= sum_over_lines(line capacity from A to B)
            req = self.capacity_required[source][dest]
            cons = self.solver.Constraint(int(req), self.solver.infinity())
            for line in self.line_names:
                capacity = self.line_capacity_per_train_up[line][source][dest]
                var = self.var_upstream_trains_per_hour[line]
                cons.SetCoefficient(var, int(capacity))

                capacity = self.line_capacity_per_train_down[line][source][dest]
                var = self.var_downstream_trains_per_hour[line]
                cons.SetCoefficient(var, int(capacity))

        for s1 in self.stop_names:
            for s2 in self.stop_names:
                constrain(s1, s2)

    def solve(self):
        if not self.solver_invoked:
            self.solver.Solve()
        self.solver_invoked = True

    def print_solution(self):
        self.solve()

        for line in self.line_names:
            uptrains = self.var_trains_on_line_up[line].SolutionValue()
            downtrains = self.var_trains_on_line_down[line].SolutionValue()
            if self.is_line_circular(line):
                print(f"{line}:    UP: {uptrains}    DOWN: {downtrains}")
            else:
                print(f"{line}:    UP: {uptrains}")
        print()
        print(f"Total number of trains required: {self.objective.Value()}")
        print()

    def main(self):
        self.solve()
        print()
        print("Printing a sample of shortest paths calculated between stations")
        print("---------------------------------------------------------------")
        print()
        self.print_shortest_paths()
        print()
        print("Printing number of trains needed")
        print("--------------------------------")
        print()
        self.print_solution()
        print()


#!/usr/bin/env python3
import argparse
import sys

from t1 import *
from t2 import *
from t3 import *

def main():
    parser = argparse.ArgumentParser("Assignment 2: Decision Analytics")
    parser.add_argument(\
                "-t",
                "--task",
                type=int,
                choices=[1, 2, 3])
    args = parser.parse_args()

    if args.task == 1:
        Task1("./Assignment_DA_2_Task_1_data.xlsx").main()
    elif args.task == 2:
        Task2("./Assignment_DA_2_Task_2_data.xlsx", "Cork").main()
    elif args.task == 3:
        TrainCapacity('Assignment_DA_2_Task_3_data.xlsx').main()
    else:
        print(f"{args.task} is not a valid value for task")

if "__main__" == __name__:
    main()
