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
        self.var_fcp_a = self.create_factory_customer_product_coefficients()

        # 3D matrix. How many units of material m is supplied by supplier s to
        # factory f
        # Indexing: [supplier][factory][material]
        self.var_sfm = self.create_supplier_factory_material_variables()

        # Setting the coefficients is done in two steps , hence the coefficients
        # must be accumulated before assigning to the variables
        # We use another 3D matrix of floats to accumulate the coefficients
        # before assigning them
        self.var_sfm_a = self.create_supplier_factory_material_coefficients()

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
        """ Create constraints for stocks each supplier has """
        
        def set_supplier_zero(supplier:str, material:str):
            for factory in self.factory_names:
                var = self.var_sfm[supplier][factory][material]
                constraint = self.solver.Constraint(0, 0)
                constraint.SetCoefficient(var, 1.0)

        def set_supplier_capacity(supplier:str, material:str, capacity:int):
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
        """ Create constraints for each product and  """
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
        """Ensure all customer demands are met"""

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
        for factory in self.factory_names:
            for product in self.product_names:
                cost_per_unit = get_element(\
                    self.production_cost_df, product, factory)
                cost_per_unit = float(cost_per_unit)
                if cost_per_unit == float('inf'): cost_per_unit = 0.0
                for customer in self.customer_names:
                    self.var_fcp_a[factory][customer][product] += cost_per_unit

    # Sheet 8
    def accumulate_shipping_cost(self):
        for factory in self.factory_names:
            for customer in self.customer_names:
                shipping_cost_per_unit = get_element(\
                    self.shipping_cost_df, factory, customer)
                shipping_cost_per_unit = float(shipping_cost_per_unit)
                if shipping_cost_per_unit == float('inf'):
                    shipping_cost_per_unit = 0.0
                for product in self.product_names:
                    self.var_fcp_a[factory][customer][product] += \
                        shipping_cost_per_unit

    # Sheet 2: Raw Materials Cost
    def accumulate_raw_materials_cost(self):
        for supplier in self.supplier_names:
            for material in self.material_names:
                material_cost = get_element(\
                    self.raw_material_cost_df, supplier, material)
                material_cost = float(material_cost)
                if material_cost == float('inf'): material_cost = 0.0
                for factory in self.factory_names:
                    self.var_sfm_a[supplier][factory][material] += \
                        material_cost

    # Sheet 3: Raw Metrials Shipping
    def accumulate_raw_materials_shipping_cost(self):
        for supplier in self.supplier_names:
            for factory in self.factory_names:
                shipping_cost = get_element(\
                    self.raw_material_shipping_df, supplier, factory)
                shipping_cost = float(shipping_cost)
                if shipping_cost == float('inf'): shipping_cost = 0.0
                for material in self.material_names:
                    self.var_sfm_a[supplier][factory][material] += shipping_cost

    def set_objective_coefficients(self):
        for prod in self.product_names:
            for fact in self.factory_names:
                for cust in self.customer_names:
                    var = self.var_fcp[fact][cust][prod]
                    val = self.var_fcp_a[fact][cust][prod]
                    self.cost_objective.SetCoefficient(var, val)

        for fact in self.factory_names:
            for supp in self.supplier_names:
                for mat in self.material_names:
                    var = self.var_sfm[supp][fact][mat]
                    val = self.var_sfm_a[supp][fact][mat]
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

        # TODO: Start from Step N

    def main(self):
        self.solve()
        self.verify_solution()
        self.print_solution()

    
if "__main__" == __name__:
    Task1("./Assignment_DA_2_Task_1_data.xlsx").main()
