#!/usr/bin/env python3

import pandas as pd
from ortools.sat.python import cp_model
import sys

class ProjectSolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self):
        super().__init__()
        self.solutions = 0

    def OnSolutionCallback(self):
        self.solutions = self.solutions + 1
        sys.stdout.write('.')
        sys.stdout.flush()

class Project:

    def __init__(self, excel_file:str):
        self.project_df = None
        self.quote_df = None
        self.depend_df = None
        self.value_df = None
        self.project_names = None
        self.job_names = None
        self.contractor_names = None
        self.month_names = None
        self.excel_file = excel_file
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()

        # One variable per project to indicate whether it is picked up or not
        self.varproject = {}

        # 4-D dict of variables: Project, Month, Job, Contractor
        self.varmatrix = {}


        self.read_excel(self.excel_file)
        self.create_project_variables_and_constraints()

        self.create_matrix_variables()

    def solve(self):
        solution_printer = ProjectSolutionPrinter()
        status = self.solver.SearchForAllSolutions(self.model, solution_printer)
        print(self.solver.StatusName(status))
        return self.solver, solution_printer.solutions

    def read_excel(self, excelfile:str) -> None:
        self.project_df = pd.read_excel(excelfile, sheet_name='Projects')
        self.quote_df = pd.read_excel(excelfile, sheet_name='Quotes')
        self.depend_df = pd.read_excel(excelfile, sheet_name='Dependencies')
        self.value_df = pd.read_excel(excelfile, sheet_name='Value')

        self.project_df.rename(columns={'Unnamed: 0':'Project'}, inplace=True)
        self.quote_df.rename(columns={'Unnamed: 0':'Contractor'}, inplace=True)
        self.depend_df.rename(columns={'Unnamed: 0':'Project'}, inplace=True)
        self.value_df.rename(columns={'Unnamed: 0':'Project'}, inplace=True)

        self.month_names = self.project_df.columns[1:].tolist()
        self.job_names = self.quote_df.columns[1:].tolist()
        self.project_names = self.project_df['Project'].tolist()
        self.contractor_names = self.quote_df['Contractor'].tolist()


        print(f"Project Names   : {self.project_names}")
        print(f"Month Names     : {self.month_names}")
        print(f"Job Names       : {self.job_names}")
        print(f"Contractor Names: {self.contractor_names}")

    def create_project_variables_and_constraints(self):
        # Create a single variable for each project
        # Also lookup the dependencies DF and add constraints accordingly
        for p in self.project_names:
            self.varproject[p] = self.model.NewBoolVar(f"{p}")

        def add_required_dependency(p1:str, p2:str)->None:
            # p1 implies p2
            self.model.AddBoolOr(                                           \
                    [                                                       \
                        self.varproject[p1].Not(),                        \
                        self.varproject[p2]                               \
                    ])

        def add_conflict_dependency(p1:str, p2:str)->None:
            self.model.AddBoolOr(                                           \
                    [                                                       \
                        self.varproject[p1].Not(),                        \
                        self.varproject[p2].Not()                         \
                    ])

        for p1 in self.project_names:
            for p2 in self.project_names:
                row_p1 = self.depend_df[self.depend_df['Project'] == p1]
                e_p1_p2 = row_p1[p2].tolist()[0]
                if (isinstance(e_p1_p2, str)):
                    if ('required' == e_p1_p2.lower()):
                        add_required_dependency(p1, p2)
                    if ('conflict' == e_p1_p2.lower()):


    def create_matrix_variables(self):
        # 4-D array of variables: Project, Month, Job, Contractor
        for project in self.project_names:
            prj_variables = {}
            for month in self.month_names:
                mnth_variables = {}
                for job in self.job_names:
                    job_variables = {}
                    for contractor in self.contractor_names:
                        job_variables[contractor] = self.model.NewBoolVar(  \
                                f"{project}-{month}-{job}-{contractor}")
                    mnth_variables[job] = job_variables
                prj_variables[month] = mnth_variables
            self.varproject[project] = prj_variables

    def get_project_job_month_relationships(self)->dict:
        # return a hash: project -> [(month, job) ....]
        ret = {}
        for prjname in self.project_names:
            jobmonthlist = []
            prjrow = self.project_df[self.project_df['Project'] == prjname]
            for mnth in self.month_names:
                element = prjrow[mnth].tolist()[0]
                if (isinstance(element, str)):
                    jobmonthlist.append((mnth, element,))
            ret[prjname] = jobmonthlist 
        print(ret)
        return ret



def main():
    prj = Project('Assignment_DA_1_data.xlsx')
    prj.get_project_job_month_relationships()

    #solver, num_solutions = prj.solve()
    #print(f"{num_solutions} solutions")

main()
