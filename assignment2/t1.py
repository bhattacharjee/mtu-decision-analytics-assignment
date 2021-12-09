#!/usr/bin/env python3

import ortools
import pandas as pd
from ortools.linear_solver import pywraplp

class Task1():
    def __init__(self, excel_file_name:str):
        self.excel_file_name = excel_file_name

        self.supplier_stock_df = self.read_csv("Supplier stock")
        self.replace_nans(self.supplier_stock_df, 0)

        self.raw_materials_stock_df = self.read_csv("Raw material costs")
        self.replace_nans(self.raw_materials_stock_df, 'inf')

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


        self.factory_customer = self.create_factory_customer_variables()

        self.supplier_factory = self.create_supplier_factory_variables()

        print(self.supplier_factory)


    def replace_nans(self, df:pd.DataFrame, newValue:int):
        df.replace(float('nan'), float(newValue), inplace=True)
        print(df)


    def read_csv(self, sheet_name:str)->pd.DataFrame:
        df = pd.read_excel(self.excel_file_name, sheet_name=sheet_name)
        print()
        print(sheet_name)
        print(df)
        print()
        print()
        return df

    def create_factory_customer_variables(self):
        ret = {}
        for factory in self.factory_names:
            inner = {}
            for customer in self.customer_names:
                varname = f"factory:{factory}-customer:{customer}"
                variable = self.solver.NumVar(\
                    0, self.solver.infinity(), varname)
                inner[customer] = variable
            ret[factory] = inner
        return ret

    def create_supplier_factory_variables(self):
        ret = {}
        for supplier in self.supplier_names:
            inner = {}
            for factory in self.factory_names:
                varname = f"supplier:{supplier}-factory:{factory}"
                variable = self.solver.NumVar(\
                    0, self.solver.infinity(), varname)
                inner[factory] = variable
            ret[supplier] = inner
        return ret




def t1_main()->None:
    t1 = Task1("./Assignment_DA_2_Task_1_data.xlsx")
    

if "__main__" == __name__:
    t1_main()
