#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 12:38:52 2023

@author: Chidex
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import stats


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
    df.rename(index={"Self-employed, total (% of total employment) (modeled " +
                     "ILO estimate)": "Self-employed (% of total employment)"},
                     inplace=True)

    df = df.astype("float64")

    df_transposed = df.transpose().sort_index()

    return df, df_transposed


def plot_heatmap(corr, region_name):
    """
    Plot heatmap to show the correlation between the indicators

    Parameters
    ----------
    corr : DataFrame
        A dataframe containg the correlation coeffiecients of the inindicators.
    region_name : str
        The name of the plotted region. It will be used to name the saved plot

    Returns
    -------
    None.

    """
    
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(corr, interpolation="nearest")
    fig.colorbar(im, orientation='vertical', fraction=0.05)

    columns_length = len(corr.columns)

    # Show all ticks and label them with the column name
    ax.set_xticks(np.arange(columns_length), labels=corr.columns, fontsize=15)
    ax.set_yticks(np.arange(columns_length), labels=corr.columns, fontsize=15)
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=-90)
    
    # The threshold for which to change the text color to ensure visibility
    threshold = im.norm(corr.to_numpy().max())/2

    # Loop over the data and create text annotations
    for i in range(columns_length):
        for j in range(columns_length):
            # Set the text color based on the color of the box
            color = ("white", "black")[
                int(im.norm(corr.to_numpy()[i, j]) > threshold)]

            ax.text(j, i, corr.to_numpy()[i, j], ha="center",
                    va="center", color=color)

    plt.savefig(f"{region_name}_heatmap.png", bbox_inches='tight')

    plt.show()
    
    return


df, df_transposed = get_dataframes("worldbank_data.csv")

# Get some summary statistics for the variables across different regions
df_transposed.describe()
df_transposed.agg([stats.skew, stats.kurtosis])

# Plot heatmap and examine the correlation between the indicators
corr = df_transposed.corr().round(2)
for region in corr.columns.levels[0]:
    plot_heatmap(corr.loc[region, region], region)
