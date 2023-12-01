#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 12:38:52 2023

@author: Chidex
"""

import pandas as pd
import numpy as np


def clean_data(data):
    """
    Removes unwanted columns, renames some columns and represent missing 
    values as nans

    Parameters
    ----------
    data : DataFrame
        The data to clean.

    Returns
    -------
    cleaned_data : DataFrame
        The data with renamed columns and nan as missing values.

    """
    cleaned_data = data.drop(columns=["Series Code", "Country Code"])

    columns = ["Region" if col_name == "Country Name" else col_name
               .split(" [")[0] for col_name in list(cleaned_data.columns)]

    cleaned_data.columns = columns
    cleaned_data.replace('..', np.nan, inplace=True)

    return cleaned_data


def get_dataframes(filename):
    """
    Load data from csv file and return two dataframes, one with the years as
    columns and the second with the regions as columns

    Parameters
    ----------
    filename : str
        The name of the csv file to load with pandas.

    Returns
    -------
    df : DataFrame
        DataFrame with the years as column.
    df_transposed : DataFrame
        Transposed DataFrame with the regions as the first level column.

    """
    df = pd.read_csv(filename)

    df = clean_data(df)

    # Set the region and series name as index, leaving years as the columns
    df = df.set_index(
        ["Region", "Series Name"]).sort_index()

    df_transposed = df.transpose().sort_index()

    return df, df_transposed


df, df_transposed = get_dataframes("worldbank_data.csv")
