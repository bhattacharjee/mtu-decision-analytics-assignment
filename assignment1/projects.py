#!/usr/bin/env python3

import pandas as pd
from ortools.sat.python import cp_model

class Project:

    def __init__(self, excel_file:str):
        self.projects_df = None
        self.quotes_df = None
        self.depend_df = None
        self.value_df = None
        self.project_names = None
        self.job_names = None
        self.contractor_names = None
        self.month_names = None
        self.excel_file = excel_file
        self.model = cp_model.CpModel()

        # One variable per project to indicate whether it is picked up or not
        self.project_vars = {}


        self.read_excel(self.excel_file)
        self.create_project_variables_and_constraints()

    def read_excel(self, excelfile:str) -> None:
        self.projects_df = pd.read_excel(excelfile, sheet_name='Projects')
        self.quotes_df = pd.read_excel(excelfile, sheet_name='Quotes')
        self.depend_df = pd.read_excel(excelfile, sheet_name='Dependencies')
        self.value_df = pd.read_excel(excelfile, sheet_name='Value')

        self.projects_df.rename(columns={'Unnamed: 0':'Project'}, inplace=True)
        self.quotes_df.rename(columns={'Unnamed: 0':'Contractor'}, inplace=True)
        self.depend_df.rename(columns={'Unnamed: 0':'Project'}, inplace=True)
        self.value_df.rename(columns={'Unnamed: 0':'Project'}, inplace=True)

        self.month_names = self.projects_df.columns[1:].tolist()
        self.job_names = self.quotes_df.columns[1:].tolist()
        self.project_names = self.projects_df['Project'].tolist()
        self.contractor_names = self.quotes_df['Contractor'].tolist()


        print(f"Project Names   : {self.project_names}")
        print(f"Month Names     : {self.month_names}")
        print(f"Job Names       : {self.job_names}")
        print(f"Contractor Names: {self.contractor_names}")

        print(self.depend_df)

    def create_project_variables_and_constraints(self):
        # Create a single variable for each project
        # Also lookup the dependencies DF and add constraints accordingly
        for p in self.project_names:
            self.project_vars[p] = self.model.NewBoolVar(f"{p}")

        def add_required_dependency(p1:str, p2:str)->None:
            # p1 implies p2
            self.model.AddBoolOr(                                           \
                    [                                                       \
                        self.project_vars[p1].Not(),                        \
                        self.project_vars[p2]                               \
                    ])

        def add_conflict_dependency(p1:str, p2:str)->None:
            self.model.AddBoolOr(                                           \
                    [                                                       \
                        self.project_vars[p1].Not(),                        \
                        self.project_vars[p2].Not()                         \
                    ])

        for p1 in self.project_names:
            for p2 in self.project_names:
                row_p1 = self.depend_df[self.depend_df['Project'] == p1]
                e_p1_p2 = row_p1[p2].tolist()[0]
                if (isinstance(e_p1_p2, str)):
                    if ('required' == e_p1_p2.lower()):
                        add_required_dependency(p1, p2)
                    if ('conflict' == e_p1_p2.lower()):
                        add_conflict_dependency(p1, p2)

        print(self.depend_df)


def main():
    prj = Project('Assignment_DA_1_data.xlsx')

main()
