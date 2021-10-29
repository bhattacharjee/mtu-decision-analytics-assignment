#!/usr/bin/env python3

import pandas as pd

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

        self.read_excel(self.excel_file)

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


def main():
    prj = Project('Assignment_DA_1_data.xlsx')

main()
