#!/usr/bin/env python3

import ortools
import pandas as pd
from ortools.linear_solver import pywraplp

class Task1():
    def __init__(self, excel_file_name:str):
        self.excel_file_name = excel_file_name
        self.supplier_stock_df = self.read_csv("Supplier stock")
        self.raw_materials_stock_df = self.read_csv("Raw material costs")
        self.raw_material_shipping_df = self.read_csv("Raw material shipping")
        self.product_requirements_df = self.read_csv("Product requirements")
        self.production_capacity_df = self.read_csv("Production capacity")
        self.production_cost_df = self.read_csv("Production cost")
        self.customer_demand_df = self.read_csv("Customer demand")
        self.shipping_cost_df = self.read_csv("Shipping costs")

        self.solver = pywraplp.Solver(\
                        'LPWrapper',\
                        pywraplp.Solver.GLPK_LINEAR_PROGRAMMING)


    def read_csv(self, sheet_name:str)->pd.DataFrame:
        return pd.read_excel(self.excel_file_name, sheet_name=sheet_name)


def t1_main()->None:
    t1 = Task1("./Assignment_DA_2_Task_1_data.xlsx")
    

if "__main__" == __name__:
    t1_main()
