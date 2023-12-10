#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 12:38:52 2023

@author: Chidex
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import stats


def clean_data(data):
    """
    Removes unwanted columns, renames some columns, represent missing 
    values as nans, and set numeric values to float type

    Parameters
    ----------
    data : DataFrame
        The data to clean.

    Returns
    -------
    cleaned_data : DataFrame
        The data with renamed columns, correct index, and values set as float.

    """

    cleaned_data = data.drop(columns=["Series Code", "Country Code"])

    # Change column names like "2002 [YR2002]" to "2002" and "Country Name"
    # to "Region"
    columns = ["Region" if col_name == "Country Name" else col_name
               .split(" [")[0] for col_name in list(cleaned_data.columns)]
    cleaned_data.columns = columns

    cleaned_data.replace('..', np.nan, inplace=True)

    # Set the region and series name as index, leaving years as the columns
    cleaned_data = cleaned_data.set_index(
        ["Region", "Series Name"]).sort_index()

    # Change to a shorter name
    cleaned_data.rename(index={
                        "Self-employed, total (% of total employment)"
                        + " (modeled ILO estimate)": "Self-employed"},
                        inplace=True)

    # Rename index like "Sample (current US$)" to "Sample (US$)"
    cleaned_data.rename(lambda x: x.replace(
        "current ", ""), axis='index', inplace=True)

    # Ensure that the values are currently represented as floats
    cleaned_data = cleaned_data.astype("float64")

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

    df_transposed = df.transpose().sort_index()
    df_transposed.index = pd.to_datetime(df_transposed.index, format="%Y")

    # Define a new column showing the net exports (Balance of Trade)
    imports = df_transposed.xs("Imports of goods and services (US$)",
                               axis=1, level="Series Name")
    exports = df_transposed.xs("Exports of goods and services (US$)",
                               axis=1, level="Series Name")
    net_exports = exports - imports
    net_exports.columns = pd.MultiIndex.from_product(
        [net_exports.columns, ['Net exports (US$)']],
        names=["Region", "Series Name"])

    # Add the new Net exports column to the transposed dataframe and
    # remove the import and export columns
    df_transposed = pd.concat([df_transposed, net_exports], axis=1) \
        .sort_index(level="Region", axis=1) \
        .drop(columns=["Imports of goods and services (US$)",
                       "Exports of goods and services (US$)"],
              level="Series Name")

    return df, df_transposed


def extract_statistics(description_df, skew_kurtosis_df):
    """
    Shows the summary statistics of some indicators for all the regions
    in a styled pandas DataFrame

    Parameters
    ----------
    description_df : DataFrame
        The dataframe containing the result of calling the describe method
        on a dataframe i.e the dataframe showing the mean, min, max e.tc.
    skew_kurtosis_df : DataFrame
        The dataframe containing the skewness and kurtosis of the data.

    Returns
    -------
    None.

    """
    # Define the indicators whose summary statistics would be in the table
    columns_to_show = ["GDP per capita (US$)",
                       "Gross national expenditure (US$)",
                       "Inflation, consumer prices (annual %)"]

    # Select the statistics of only the required indicators
    idx = pd.IndexSlice
    selected_desc = description_df.loc[["mean"], idx[:, columns_to_show]]
    selected_skew_kurt = df_skew_kurtosis.loc[:, idx[:, columns_to_show]]

    combined_stat = pd.concat([selected_desc, selected_skew_kurt])

    # Reshape the combined statistics to the desired shape
    summary_stats_df = combined_stat.T.unstack().reorder_levels(
        [1, 0], axis=1).sort_index(level=0, axis=1, sort_remaining=False)

    # Remove the index and columns names. I'm passing a list containing two
    # null items to rename_axis since the column has two levels
    summary_stats_df.rename_axis([None, None], axis="columns", inplace=True)
    summary_stats_df.index.name = None

    header_styles = ["background-color: darkblue; color: white;" +
                     "border: 1px solid white; text-align: left;"]
    # Apply the following styles to the resulting dataframe:
    # Format the values, add border to the table, set the table title,
    # highlight the min and max mean of each indicators, and add background
    # color to the colum and row headers
    summary_stats_df = summary_stats_df.style.format("{:,.2f}") \
        .set_properties(**{'border': '1px solid black'}) \
        .set_caption("Summary Statistics of Indicators Over the Years") \
        .highlight_max(color='lightgreen', subset=idx[:, idx[:, "mean"]]) \
        .highlight_min(color='cyan', subset=idx[:, idx[:, "mean"]]) \
        .apply_index(lambda v: header_styles * len(v)) \
        .apply_index(lambda v: header_styles * len(v), axis="columns") \
        .set_table_styles([
            {'selector': 'caption',
             'props': [
                 ('font-size', '20px'),
                 ('font-weight', 'bold'),
                 ('padding-bottom', '20px'),
             ]
             }
        ])

    # Save the extracted statistics to an excel file
    summary_stats_df.to_excel(
        'plots/summary_statistics.xlsx', engine='openpyxl')

    return


def plot_boxplot(df, title, ylabel):
    """
    Make a boxplot from a DataFrame

    Parameters
    ----------
    df : DataFrame
        The dataframe to plot.
    title : str
        The title of the box plot
    ylabel: str
        The label of the y-axis

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots()

    # Sort the box plots by their median values
    index = df.median().sort_values(ascending=False).index

    # Add lines showing the means in each box
    df[index].boxplot(ax=ax, showmeans=True, meanline=True, medianprops=dict(
        color="deeppink", linewidth=1.5), meanprops=dict(color="green",
                                                         linewidth=1.5))

    # Set the plot title and shift it away from the plot with the y option
    ax.set_title(title, fontweight="bold", y=1.03)
    ax.set_ylabel(ylabel)

    plt.setp(ax.get_xticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")

    # Show a legend for the mean and median lines
    median_line = Line2D([], [], color="deeppink", label="Median",
                         markersize=18)
    mean_line = Line2D([], [], color="green", label="Mean",
                       markersize=18, linestyle="--")
    ax.legend(handles=[median_line, mean_line],
              loc="lower center", ncol=1, fontsize=12)
    ax.legend_.set_bbox_to_anchor([0.8, 0.75])

    plt.savefig("plots/box_plot.png", bbox_inches='tight')

    plt.show()

    return


def plot_heatmap(corr, region_name):
    """
    Plot heatmap to show the correlation between the indicators

    Parameters
    ----------
    corr : DataFrame
        A dataframe containg the correlation coeffiecients of the inindicators
    region_name : str
        The name of the plotted region. It will be used to name the saved plot
        and for the plot title

    Returns
    -------
    None.

    """

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(corr, interpolation="nearest")

    # Add a colorbar to the plot
    fig.colorbar(im, orientation='vertical', fraction=0.045)

    columns_length = len(corr.columns)

    # Set the plot title
    ax.set_title(
        f"Correlation of Indicators for {region_name}",
        fontweight="bold", fontsize=20, y=1.03)

    # Show all ticks and label them with the column name
    ax.set_xticks(np.arange(columns_length), labels=corr.columns, fontsize=15)
    ax.set_yticks(np.arange(columns_length), labels=corr.columns, fontsize=15)

    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="left")

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

    plt.savefig(f"plots/{region_name} heatmap.png", bbox_inches='tight')

    plt.show()

    return


def plot_line_graphs(df, columns, title, xlabel):
    """
    Plot

    Parameters
    ----------
    df : DataFrame
        The dataframe to plot.
    columns: list or sequence
        The list of columns to include in the line plot
    title : str
        The title of the plot.
    xlabel : str
        The label of the x-axis.

    Returns
    -------
    None.

    """

    fig, axes = plt.subplots(2, 3, figsize=(20, 12), layout='constrained')

    fig.suptitle(title, fontsize=30, fontweight="bold", y=1.10)

    i = 0
    j = 0

    idx = pd.IndexSlice
    selected_df = df.loc[:, idx[:, columns]]

    for series_name in columns:
        df_to_plot = selected_df.xs(series_name, level="Series Name", axis=1)
        ax = axes[i, j]

        df_to_plot.plot(ax=ax, legend=False, grid=True, xlim=("1996", "2020"),
                        linewidth=4, x_compat=True)

        ax.set_ylabel(series_name, fontsize=20)

        # Set the axes to make the next plot on
        if j < 2:
            j += 1
        else:
            i += 1
            j = 0

    fig.supxlabel(xlabel, fontsize=20)

    # Display one legend for all the subplots
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncols=3, fontsize=18,
               bbox_to_anchor=[0.5, 1.07])

    plt.savefig("plots/line_plot.png", bbox_inches='tight')

    plt.show()

    return


# Get the pandas dataframe from the file
df, df_transposed = get_dataframes("worldbank_data.csv")

# Get some summary statistics for the indicators across different regions
df_description = df_transposed.describe().round(2)
df_skew_kurtosis = df_transposed.agg([stats.skew, stats.kurtosis]).round(2)

extract_statistics(df_description, df_skew_kurtosis)

# Plot heatmap and examine the correlation between the indicators
corr = df_transposed.corr().round(2)
for region in corr.columns.levels[0]:
    plot_heatmap(corr.loc[region, region], region)

# Plot line graph showing the trend of indicators across different regions
line_plot_title = "Trend of different indicators in the different regions"
columns_to_plot = [
    "Total natural resources rents (% of GDP)",
    "Inflation, consumer prices (annual %)",
    "GDP per capita (US$)",
    "Net migration",
    "Net exports (US$)",
    "Gross national expenditure (US$)"]
plot_line_graphs(df_transposed, columns_to_plot, line_plot_title, "Years")

# Make boxplot of inflation across different regions for the observed years
plot_boxplot(df_transposed.xs(
    "Inflation, consumer prices (annual %)", axis=1, level="Series Name"),
    "Inflation Percentage for Each Region",
    "Inflation, consumer prices (annual %)")
