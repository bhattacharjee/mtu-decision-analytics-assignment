#!/usr/bin/env python3

import ortools
import pandas as pd
from ortools.linear_solver import pywraplp
from functools import lru_cache

def get_element(df:pd.DataFrame, rowname:str, colname:str):
    selector = df['Unnamed: 0'] == rowname
    df = df[selector]
    return df[colname].to_numpy()[0]

class Task3:
    def __init__(self, excel_file_name):
        self.excel_file_name = excel_file_name

        self.stop_df = self.read_csv("Stops")
        self.distance_df = self.read_csv("Distances")
        self.passenger_df = self.read_csv("Passengers")
        self.train_capacity_df = self.read_csv("Trains")

        self.replace_nans(self.stop_df, 0)
        self.replace_nans(self.distance_df, 0)
        self.replace_nans(self.passenger_df, 0)

        self.line_names = [str(l) for l in self.stop_df.columns][1:]
        self.stop_names = [str(s) for s in self.distance_df.columns][1:]


    def read_csv(self, sheet_name:str)->pd.DataFrame:
        df = pd.read_excel(self.excel_file_name, sheet_name=sheet_name)
        return df

    def replace_nans(self, df:pd.DataFrame, newValue:int):
        df.replace(float('nan'), float(newValue), inplace=True)

    def main(self):
        # TODO: Fill up 
        pass

if "__main__" == __name__:
    Task3("Assignment_DA_2_Task_3_data.xlsx").main()
