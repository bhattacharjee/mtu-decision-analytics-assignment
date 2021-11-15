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
        """[summary]

        Args:
            p ([type]): [description]
            m ([type]): [description]
            j ([type]): [description]

        Returns:
            [type]: [description]
        """
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
    """[summary]
    """

    def __init__(self, excel_file:str):
        """[summary]

        Args:
            excel_file (str): [description]
        """
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

        # PART A: READ THE EXCEL
        self.read_excel(self.excel_file)

        # PART B: Create teh variables
        # For each project picked, have a T/F variable
        self.crtvars_p()

        # PART B: Create teh variables
        # Have a 4-D array of T/F variables, the dimensions specify the
        # following:
        # PROJECTS, MONTHS, JOBS, CONTRACTORS
        # if a contractor picks up a job in a month for a project, then the
        # corresponding variable is set to True
        self.crtvars_pmjc()

        # PART C: Contractors cannot work on two projects simultaneously
        self.crtcons_contractor_single_simult_project()

        # PART D-1: Only one contractor should be assined to a job
        # (for a project and month)
        self.crtcons_one_contractor_per_job()

        # PART E: If a project is not selected, no one should work on it
        self.crtcons_project_not_selected()

        # PART F: Add constraints for dependencies between projects
        self.crtcons_project_dependencies_conflicts()

        # PART G: Add constraints so that difference between the value of
        # projects delivered and the cost of all contractors is at least 2160
        self.crtcons_profit_margin(2160)

        # Implicit constraint, if any contractor picks up any job
        # for any project in any month in the 4D array, then the project
        # must have been picked up in the 1D projects array
        self.crtcons_pmjc_p()

        # Implicit constraint, if a project is selected, then
        # all jobs for the project must be done
        self.crtcons_complete_all_jobs_for_project()

        self.crtcons_job_contractor()
    

    def validate_solution(self, soln:dict)->None:
        """[summary]

        Args:
            soln (dict): [description]
        """
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
        """[summary]

        Args:
            soln (dict): [description]

        Returns:
            int: [description]
        """
        # solution is a dictionary
        # project -> [(m, j, c), ...]
        profit = 0
        for p in soln.keys():
            profit = profit + self.get_project_value(p)
            for (m, j, c) in soln[p]:
                profit = profit - self.get_contractor_job_cost(c, j)
        return profit

    def validate_and_get_profit(self, soln:dict)->int:
        """[summary]

        Args:
            soln (dict): [description]

        Returns:
            int: [description]
        """
        self.validate_solution(soln)
        return self.get_profit(soln)


    def solve(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        solution_printer = ProjectSolutionPrinter(self)
        status = self.solver.SearchForAllSolutions(self.model, solution_printer)
        print(self.solver.StatusName(status))
        return self.solver, solution_printer.solutions

    # PART A: Read the Excel
    def read_excel(self, excelfile:str) -> None:
        """[summary]

        Args:
            excelfile (str): [description]
        """
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
        """[summary]
        """
        print(f"Project Names   : {self.project_names}")
        print(f"Month Names     : {self.month_names}")
        print(f"Job Names       : {self.job_names}")
        print(f"Contractor Names: {self.contractor_names}")
        print(f"Project Names   : {len(self.project_names)}")
        print(f"Month Names     : {len(self.month_names)}")
        print(f"Job Names       : {len(self.job_names)}")
        print(f"Contractor Names: {len(self.contractor_names)}")

    # PART B: Create teh variables
    # For each project picked, have a T/F variable
    def crtvars_p(self):
        """[summary]
        """
        # Create a single variable for each project
        # Also lookup the dependencies DF and add constraints accordingly
        for p in self.project_names:
            self.var_p[p] = self.model.NewBoolVar(f"{p}")

    def crtcons_project_dependencies_conflicts(self):
        """Add constraints for dependencies between projects.
           We have already stored the dependencies of projects in the
           dataframe self.depend_df
        """
        def add_required_dependency(p1:str, p2:str)->None:
            # p1 implies p2
            self.model.AddBoolOr(                                           \
                    [                                                       \
                        self.var_p[p1].Not(),                               \
                        self.var_p[p2]                                      \
                    ])

        def add_conflict_dependency(p1:str, p2:str)->None:
            # P1 is incompatible with P2, so both of them cannot be TRUE
            # NOT(A AND B) is written here as NOT A OR NOT B
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

    # PART B: Create teh variables
    # Have a 4-D array of T/F variables, the dimensions specify the
    # following:
    # PROJECTS, MONTHS, JOBS, CONTRACTORS
    # if a contractor picks up a job in a month for a project, then the
    # corresponding variable is set to True
    def crtvars_pmjc(self):
        """[summary]
        """
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

    # Implicit constraint, if any contractor picks up any job
    # for any project in any month in the 4D array, then the project
    # must have been picked up in the 1D projects array
    def crtcons_pmjc_p(self):
        """Implicit constraint, if any contractor picks up any job
           for any project in any month in the 4D array, then the project
           must have been picked up in the 1D projects array
        """
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

    # PART C: Contractors cannot work on two projects simultaneously
    def crtcons_contractor_single_simult_project(self):
        """Implement a constraint that a contractor cannot work on two
           projects on the same month.
           For every contractor and month, the sum of jobs in all projects
           must be at most 1
        """
        # This constraint can be simplified, since a contractor can only do
        # one project at a time, that implies he can only do one job
        # at a time, and vice versa
        # So we'll replace this by adding a constrint for one simultaneous job
        # For each month, for each contractor -> count of jobs = 1
        for m in self.month_names:
            for c in self.contractor_names:
                variables = []
                for p in self.project_names:
                    for j in self.job_names:
                        variables.append(self.var_pmjc[p][m][j][c])
                self.model.Add(sum(variables) <= 1)

    def get_project_job_month_relationships(self)->dict:
        """Get the list of (jobs, months) for each project

        Returns:
            dict: keys are projects, values are array of (month, job)
        """
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

    # Implicit constraint, if a project is selected, then
    # all jobs for the project must be done
    def crtcons_complete_all_jobs_for_project(self):
        """Implicit constraint, if a project is selected, then
           all jobs for the project must be done
        """
        # If a project is selected, then each job for the project must be
        # done in the month specified
        def add_constraint(p, j, m):
            cvars = [self.var_pmjc[p][m][j][c] for c in self.contractor_names]
            self.model.Add(sum(cvars) == 1).OnlyEnforceIf(self.var_p[p])

        # rltn is a dictionary of the following format
        # project -> [(month1, job1), (month2, job2), ...]
        rltn = self.get_project_job_month_relationships()
        for p, monthjoblist in rltn.items():
            for monthjob in monthjoblist:
                m, j = monthjob
                add_constraint(p, j, m)

    # PART E: If a project is not selected, no one should work on it
    def crtcons_project_not_selected(self):
        """If a project is not selected, then no one should work on it
           This means that

           Not ProjectX => sum(all months, jobs, contractors for ProjectX) == 0
        """
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

    def get_contractor_job_cost(self, c:str, j:str)->float:
        """[summary]

        Args:
            c (str): [description]
            j (str): [description]

        Returns:
            float: [description]
        """
        row = self.quote_df[self.quote_df['Contractor'] == c]
        value = row[j].tolist()[0]
        return value

    def get_project_value(self, p:str)->int:
        """[summary]

        Args:
            p (str): [description]

        Returns:
            int: [description]
        """
        row = self.value_df[self.value_df['Project'] == p]
        value = row['Value'].tolist()[0]
        return value

    def crtcons_job_contractor(self)->None:
        """[summary]
        """
        # Not all contractors can do all jobs

        def add_constraint(c:str, j:str):
            """[summary]

            Args:
                c (str): [description]
                j (str): [description]
            """
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

    # PART G: Add constraints so that difference between the value of
    # projects delivered and the cost of all contractors is at least 2160
    def crtcons_profit_margin(self, margin:int)->None:
        """Add constraints so that the difference between the value
           of the projects delivered and the cost of all contractors is 
           at least margin.

        Args:
            margin (int): the margin
        """
        revenue = []        # This is the value of the projects delivered
        for p in self.project_names:
            revenue.append(self.get_project_value(p) * self.var_p[p])

        expenses = []       # This is the cost of all contractors
        for p in self.project_names:
            for m in self.month_names:
                for j in self.job_names:
                    for c in self.contractor_names:
                        var = self.var_pmjc[p][m][j][c]
                        cost = self.get_contractor_job_cost(c, j)
                        cost = 0 if math.isnan(cost) else int(cost)
                        expenses.append(cost * var)

        # Add the constraint that revenue is at least expenses + margin
        self.model.Add(sum(revenue) >= sum(expenses) + margin)

    # PART D-1: Only one contractor should be assined to a job
    # (for a project and month)
    def crtcons_one_contractor_per_job(self)->None:
        """Only one contractor per job needs to work on it
        """
        # Only one contractor per job

        def add_constraint(p:str, j:str):
            # Only one contractor per job for every project, in a given month
            for m in self.month_names:
                variables = []
                for c in self.contractor_names:
                    variables.append(self.var_pmjc[p][m][j][c])
                self.model.Add(sum(variables) <= 1).OnlyEnforceIf(self.var_p[p])

        # rltn is a hashmap where every item is
        # Project => [(month1, job1), (month2, job2), ...]
        rltn = self.get_project_job_month_relationships()
        for p, mjlist in rltn.items():
            for m, j in mjlist:
                add_constraint(p, j)


def main():
    prj = Project('Assignment_DA_1_data.xlsx')
    print("Created all variables, calling solver...")

    solver, num_solutions = prj.solve()
    print(f"{num_solutions} solutions")

main()
