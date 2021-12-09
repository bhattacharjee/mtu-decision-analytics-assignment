#!/usr/bin/env python3

import ortools
import pandas as pd
from ortools.linear_solver import pywraplp

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


        # 3D matrix, how many units of product p is supplied by factory f
        # to customer c
        self.var_fcp = self.create_factory_customer_product_variables()

        # 3D matrix. How many units of material m is supplied by supplier s to
        # factory f
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
        print(f"Best cost found: {self.cost_objective.Value()}")
        


def t1_main()->None:
    t1 = Task1("./Assignment_DA_2_Task_1_data.xlsx")
    t1.solve()
    
if "__main__" == __name__:
    t1_main()
