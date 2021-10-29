#!/usr/bin/env python3

import pandas as pd
from ortools.sat.python import cp_model
import sys
import math

class ProjectSolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, project):
        super().__init__()
        self.solutions = 0
        self.project = project

    def get_contractors(self, p, m, j):
        pmjc = self.project.var_pmjc
        cnames = self.project.contractor_names
        variables = []
        for c in cnames:
            if self.Value(pmjc[p][m][j][c]):
                variables.append(c)
        return variables

    def OnSolutionCallback(self):
        self.solutions = self.solutions + 1
        print(f"Solution # {self.solutions:02d}")
        print("---------------------------")
        chosen_projects = [p for p in self.project.project_names if         \
                                self.Value(self.project.var_p[p])]

        solution = {}
        for p in chosen_projects:
            solution[p] = []
            rltn = self.project.get_project_job_month_relationships()
            for m, j in rltn[p]:
                c = self.get_contractors(p, m, j)
                if 1 != len(c):
                    print(f"problem in OnSolutionCallback, #cont = {len(c)}")
                    assert(False)
                solution[p].append((m, j, c[0]))
        for p in sorted(solution.keys()):
            mjclist = solution[p]
            print(p, mjclist)
        profit = self.project.validate_and_get_profit(solution)
        print('-------------')
        print(f"Profit = {profit}")
        print()
        print()




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
        self.var_p = {}

        # 4-D dict of variables: Project, Month, Job, Contractor
        self.var_pmjc = {}

        # 3-D dict of variables: Project, Month, Contractor

        self.read_excel(self.excel_file)

        self.create_p_variables()
        self.create_pmjc_variables()

        self.create_constraints_between_projects()

        self.create_pmjc_p_constraints()
        
        self.constraint_contractor_single_simultaneous_project()

        # TODO: This makes it infeasible for some reason
        self.create_constraint_one_contractor_per_job()

        self.constraint_complete_all_jobs_for_project_on_time()

        self.constraint_project_not_selected()

        self.add_job_contractor_constraints()

        self.add_profit_margin_constraint(2160)
    

    def validate_solution(self, soln:dict)->None:
        # solution is a dictionary
        # project -> [(m, j, c), ...]
        projects = set(soln.keys())
        contractors = set()
        months = set()
        jobs = set()

        for p, mjclist in soln.items():
            for m, j, c in mjclist:
                contractors.add(c)
                months.add(m)
                jobs.add(j)
                
                # The contractor should be capable of doing the job
                cjcost = self.get_contractor_job_cost(c, j)
                if math.isnan(cjcost):
                    print("infeasible {cjcost}: {c} cannot do {j}")
                    
        for p, mjclist in soln.items():
            for mm in months:
                for cc in contractors:
                    count = 0
                    for m, j, c in mjclist:
                        if m == mm and c == cc:
                            count = count + 1
                    if (count > 1):
                        print("Simultaneous work: {mm} {cc} {count}")
                        assert(False)

        # Every job in every project must be done in time, and no extra
        # jobs should be undertaken
        for p, mjclist in soln.items():
            mjlist = self.get_project_job_month_relationships()[p]
            if len(mjlist) != len(mjclist):
                print(f"Number of jobs for {p} don't match: {mjlist} {mjclist}")
                assert(False)
                temp = sorted([(m, j,) for m, j, c in mjclist])
                temp2 = sorted(mjlist)
                for i, j in zip(temp, temp2):
                    if i != j:
                        print(f"Mismatch in {p}, {temp}, {temp2}")
                        assert(False)
        
        # No contractor should have jobs in parallel
        cmcount = {(c, m,):0 for c in contractors for m in months}
        for p, mjclist in soln.items():
            for m, j, c in mjclist:
                try:
                    cmcount[(c,m,)] = cmcount[(c,m,)] + 1
                except Exception as e:
                    print("Error adding to cmcount")
                    print(e)
        for k, count in cmcount.items():
            if (count > 1):
                print(f"count = {count} for {k}")
                assert(False)

    def get_profit(self, soln:dict)->int:
        # solution is a dictionary
        # project -> [(m, j, c), ...]
        profit = 0
        for p in soln.keys():
            profit = profit + self.get_project_value(p)
            for (m, j, c) in soln[p]:
                profit = profit - self.get_contractor_job_cost(c, j)
        return profit

    def validate_and_get_profit(self, soln:dict)->int:
        self.validate_solution(soln)
        return self.get_profit(soln)


    def solve(self):
        solution_printer = ProjectSolutionPrinter(self)
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

    def print_all_names(self):
        print(f"Project Names   : {self.project_names}")
        print(f"Month Names     : {self.month_names}")
        print(f"Job Names       : {self.job_names}")
        print(f"Contractor Names: {self.contractor_names}")
        print(f"Project Names   : {len(self.project_names)}")
        print(f"Month Names     : {len(self.month_names)}")
        print(f"Job Names       : {len(self.job_names)}")
        print(f"Contractor Names: {len(self.contractor_names)}")

    def create_p_variables(self):
        # Create a single variable for each project
        # Also lookup the dependencies DF and add constraints accordingly
        for p in self.project_names:
            self.var_p[p] = self.model.NewBoolVar(f"{p}")

    def create_constraints_between_projects(self):
        def add_required_dependency(p1:str, p2:str)->None:
            # p1 implies p2
            self.model.AddBoolOr(                                           \
                    [                                                       \
                        self.var_p[p1].Not(),                               \
                        self.var_p[p2]                                      \
                    ])

        def add_conflict_dependency(p1:str, p2:str)->None:
            self.model.AddBoolOr(                                           \
                    [                                                       \
                        self.var_p[p1].Not(),                               \
                        self.var_p[p2].Not()                                \
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

    def create_pmjc_variables(self):
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
            self.var_pmjc[project] = prj_variables

    def create_pmjc_p_constraints(self):
        # if an entry in pmjc is True then the corresponding entry in p must
        # also be true
        for p in self.project_names:
            for m in self.month_names:
                for c in self.contractor_names:
                    for j in self.job_names:
                        self.model.AddBoolOr(                               \
                                [                                           \
                                    self.var_pmjc[p][m][j][c].Not(),        \
                                    self.var_p[p],                          \
                                ])

    def constraint_contractor_single_simultaneous_project(self):
        # This constraint can be simplified, since a contractor can only do
        # one project at a time, that implies he can only do one job
        # at a time
        # For each month, for each contractor -> count of jobs = 1
        for m in self.month_names:
            for c in self.contractor_names:
                variables = []
                for p in self.project_names:
                    for j in self.job_names:
                        variables.append(self.var_pmjc[p][m][j][c])
                self.model.Add(sum(variables) <= 1)

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
        return ret

    def constraint_complete_all_jobs_for_project_on_time(self):
        # If a project is selected, then each job for the project must be
        # done in the month specified
        def add_constraint(p, j, m):
            cvars = [self.var_pmjc[p][m][j][c] for c in self.contractor_names]
            self.model.Add(sum(cvars) == 1).OnlyEnforceIf(self.var_p[p])

        rltn = self.get_project_job_month_relationships()
        for p, monthjoblist in rltn.items():
            for monthjob in monthjoblist:
                m, j = monthjob
                add_constraint(p, j, m)

    def constraint_project_not_selected(self):
        # If a project is not selected none of its jobs should
        # be done
        for p in self.project_names:
            variables = []
            for m in self.month_names:
                for j in self.job_names:
                    for c in self.contractor_names:
                        variables.append(self.var_pmjc[p][m][j][c])
            self.model.Add(sum(variables) == 0)                             \
                    .OnlyEnforceIf(self.var_p[p].Not())

    def get_contractor_job_cost(self, c:str, j:str):
        row = self.quote_df[self.quote_df['Contractor'] == c]
        value = row[j].tolist()[0]
        return value

    def get_project_value(self, p:str):
        row = self.value_df[self.value_df['Project'] == p]
        value = row['Value'].tolist()[0]
        return value

    def add_job_contractor_constraints(self):
        # Not all contractors can do all jobs

        def add_constraint(c:str, j:str):
            # Given c, j set to false All p, m
            variables = []
            for p in self.project_names:
                for m in self.month_names:
                    variables.append(self.var_pmjc[p][m][j][c].Not())
            self.model.AddBoolAnd(variables)

        for c in self.contractor_names:
            for j in self.job_names:
                cost = self.get_contractor_job_cost(c, j)
                cannotdo = math.isnan(cost)
                if cannotdo:
                    add_constraint(c, j)

    def add_profit_margin_constraint(self, margin:int):
        revenue = []
        for p in self.project_names:
            revenue.append(self.get_project_value(p) * self.var_p[p])
        expenses = []
        for p in self.project_names:
            for m in self.month_names:
                for j in self.job_names:
                    for c in self.contractor_names:
                        var = self.var_pmjc[p][m][j][c]
                        cost = self.get_contractor_job_cost(c, j)
                        cost = 0 if math.isnan(cost) else int(cost)
                        expenses.append(cost * var)
        self.model.Add(sum(revenue) >= sum(expenses) + margin)

    def create_constraint_one_contractor_per_job(self):
        # Only one contractor per job

        def add_constraint(p:str, j:str):
            # Only one contractor per job for every project, in a given month
            for m in self.month_names:
                variables = []
                for c in self.contractor_names:
                    variables.append(self.var_pmjc[p][m][j][c])
                self.model.Add(sum(variables) <= 1).OnlyEnforceIf(self.var_p[p])

        rltn = self.get_project_job_month_relationships()
        for p, mjlist in rltn.items():
            for m, j in mjlist:
                add_constraint(p, j)


def main():
    prj = Project('Assignment_DA_1_data.xlsx')
    prj.get_project_job_month_relationships()

    solver, num_solutions = prj.solve()
    print(f"{num_solutions} solutions")

main()


























"""
    def constraint_contractor_single_simultaneous_project(self):
        def get_job_vars(prj:str, cntr:str, mnth:str)->list:
            return [self.var_pmjc[prj][mnth][j][cntr] for j in self.job_names]

        def add_constraints(var1_list:list, var2_list):
            # If any in var1 is true, then nothing in var2 should be true
            # and vice versa
            for v1 in var1_list:
                for v2 in var2_list:
                    self.model.AddBoolOr(                                   \
                            [                                               \
                                v1.Not(),                                   \
                                v2.Not(),                                   \
                            ])

        for cntr in self.contractor_names:
            for mnth in self.month_names:
                for i in range(len(self.project_names)):
                    for j in range(i+1, len(self.project_names)):
                        prj1 = self.project_names[i]
                        prj2 = self.project_names[j]
                        vars1 = get_job_vars(prj1, cntr, mnth)
                        vars2 = get_job_vars(prj2, cntr, mnth)
                        add_constraints(vars1, vars2)
"""
