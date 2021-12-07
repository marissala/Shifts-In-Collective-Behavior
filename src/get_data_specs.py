"""Write out an overview of the dataset
Author: Maris Sala
Date: 8th Nov 2021
"""
import re
import ast
import glob
import json
import numpy as np
import pandas as pd
from icecream import ic
from re import match
import jsonlines
import pprint
from datetime import datetime

import seaborn as sns; sns.set()
import pyplot_themes as themes
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.cbook.deprecation import MatplotlibDeprecationWarning

##################################################################################
### READ DATA                                                                  ###
##################################################################################

def read_tmp_data(data_path: str):
    """Reads a temporary smaller dataset for testing
    Args:
    data_path: complete path of temporary dataset (.csv)
    
    Returns:
    pandas DataFrame
    """
    df = pd.read_csv(data_path, sep=";")
    ic(df.columns)
    ic(df.head())
    return df

def read_json_data(filename: str):
    """Reads large json file that has problems with newlines
    Args:
    filename: complete path to json file

    Returns:
    pandas DataFrame
    """
    ic("[INFO] Read string")
    with open(filename) as f:
        giant_string = f.read()
    giant_string = giant_string[:-5]
    
    ic("[INFO] Base cleaning")
    giant_string = re.sub('\\n', ' ', giant_string)
    giant_string = f'[{giant_string}]'
    
    ic("[INFO] Json loads")
    data = json.loads(giant_string)
    ic(len(data))
    del giant_string
    df = pd.DataFrame.from_records(data)
    return df

def read_data(filename: str, 
              from_originals: bool):
    """Reads in the original large json file, or a smaller saved sample
    Args:
    filename: complete path to json file
    from_originals: if user wants to load original json or a sample dataset
    
    Returns:
    pandas DataFrame
    """
    if from_originals:
        df = read_json_data(filename) # read_json_data(filename)
    else:
        df = pd.read_csv("ds.csv")
    if "text" not in df.columns:
        df["text"] = df["message"]
    return df

##################################################################################
### DOWNSAMPLE                                                                 ###
##################################################################################

def downsample(df,
            frequency='1T',
            if_list=False):
    """Downsamples based on date, concatenates texts
    Args:
    df: pandas DataFrame with columns "date" and "text"
    frequency: time interval for downsampling, default 1T - creates 1min timebins
    if_list: boolean, when texts should be aggregated as lists, drop rest of columns

    Returns:
    Downsampled pandas DataFrame
    """
    df = df.dropna().reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    ic(len(df["text"]))
    ic(len(df["date"]))
    df["text"] = pd.Series(df["text"])
    df["date"] = pd.Series(df["date"])

    df.index = pd.to_datetime(df["date"])
    df = df.drop("date", axis = 1)
    if if_list:
        df = df.resample(frequency).agg({"text": lambda x: list(x)})
    else:
        df = df.resample(frequency).agg({"text": ' '.join, "id": 'count', "comment_count": 'sum', "likes_count": 'sum', "shares_count": 'sum'})

    df = df.reset_index(col_fill = "date")
    
    if not if_list:
        df["text"] = df["text"].astype(str)
    df["date"] = df["date"].astype(str)
    
    return df

def downsample_per_group(ori_df,
            frequency='1T',
            if_list=False):
    """Downsamples based on date, concatenates texts. Per group.
    Args:
    ori_df: pandas DataFrame with columns "date" and "text"
    frequency: time interval for downsampling, default 1T - creates 1min timebins

    Returns:
    Downsampled pandas DataFrame
    """
    group_ids = list(set(ori_df["group_id"].dropna()))
    ic(len(group_ids))

    resampled = pd.DataFrame()
    complete_groups = []
    incomplete_groups = []
    
    ic("Looping over the groups")
    for group_id in group_ids:
        df = ori_df[ori_df["group_id"] == group_id].reset_index(drop=True)
        df["text"] = df["text"].astype(str)
            
        df["text"] = pd.Series(df["text"])
        df["date"] = pd.Series(df["date"])

        df.index = pd.to_datetime(df["date"])
        df = df.drop("date", axis = 1)
            
        df = df.resample(frequency).agg({"text": ' '.join, 
                                        "id": 'count', 
                                        "user_id": 'nunique',
                                        "comment_count": 'sum', 
                                        "likes_count": 'sum', 
                                        "shares_count": 'sum'})
        df = df.reset_index(col_fill = "date")
            
        df["text"] = df["text"].astype(str)
        df["date"] = df["date"].astype(str)

        df["group_id"] = group_id

        df.index = pd.to_datetime(df["date"]).dt.date
        missing = df[df.id == 0].index
        df = df.drop(df[df.id == 0].index)

        if len(missing) == 0:
            resampled = resampled.append(df)
            complete_groups.append(str(int(group_id)))
        else:
            incomplete_groups.append(str(int(group_id)))
    
    ic(complete_groups)
    ic(incomplete_groups)
    return resampled, complete_groups, incomplete_groups

##################################################################################
### SAVE DATA SPECS                                                            ###
##################################################################################

# Does not work with the current config of the dataset
def save_data_specs(datatype: str,
                    df):
    """Saves specs (word count means etc) of the specific dataset. Saved file is in /out/
    Args:
    datatype: string, "comments" or "posts" for Facebook data
    df: pandas DataFrame
    
    Returns:
    NA
    """
    df_specs = {}
    df_specs['Dataset'] = datatype
    df_specs['Length'] = len(df)
    df_specs['Unique groups'] = len(df["owner_group_id"].unique())
    df_specs['Privacy describe'] = df.privacy.describe()
    df_specs['Privacy counts'] = df.groupby("privacy").count()
    df_specs['Overall word count'] = {"Mean": df.word_count.mean(),
                            'Median': df.word_count.median(),
                            'Min': df.word_count.min(),
                            'Max': df.word_count.max()}
    df_specs['Privacy word count'] = {"Mean": df.groupby("privacy")["word_count"].mean(),
                            'Median': df.groupby("privacy")["word_count"].median(),
                            'Min': df.groupby("privacy")["word_count"].min(),
                            'Max': df.groupby("privacy")["word_count"].max()}
    with open(f'out/{datatype}_specs.json', 'w', encoding='utf-8') as f:
        json.dump(str(df_specs), f, ensure_ascii=False, indent=4)

##################################################################################
### PLOT SETTINGS                                                              ###
##################################################################################

def set_base_plot_settings(fontsize: int, 
                           if_palette: bool):
    """Controls the base settings for all seaborn plots
    Args:
    fontsize: font size for ticks
    if_palette: if True then returns a colorblind friendly color palette,
                else returns nothing
    
    Returns:
    fig, ax1 and palette
    """
    matplotlib.rc('ytick', labelsize=fontsize)
    matplotlib.rc('xtick', labelsize=fontsize)
    themes.theme_minimal(grid=False, ticks=False, fontsize=fontsize)
    a4_dims = (25,15)
    
    if if_palette:
        #          0 black      1 orange  2 L blue   3 green    4 L orange  5 D blue  6 D orange 7 purple
        palette = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
    else:
        palette = 0
    
    fig, (ax1) = plt.subplots(1,1, figsize=a4_dims)
    sns.set(font_scale = 2)

    return fig, ax1, palette

def set_late_plot_settings(fig, 
                           ax1, 
                           if_dates:bool):
    """Configures plot settings after plotting. Removes x and y labels,
       sets ylim, fontsizes for labels
    Args:
    fig: matplotlib seaborn figure
    ax1: matplotlib seaborn axis
    if_dates: whether x axis is a time series

    Returns:
    Updated fig and ax1
    """
    ax1.set(xlabel="", ylabel = "")
    ax1.xaxis.get_label().set_fontsize(40)
    ax1.yaxis.get_label().set_fontsize(40)

    ax1.grid(color='darkgrey', linestyle='-', linewidth=0.5, which= "both")
    if if_dates:
        # Define the date format
        ax1.xaxis_date(tz="UTC")
        date_form = mdates.DateFormatter("%d-%b")
        ax1.xaxis.set_major_formatter(date_form)

    ax1.set(ylim=(0, None))
    return fig, ax1

def set_late_barplot_settings(fig, 
                              ax1):
    """Sets settings for barplots after plotting. 
       Removes labels on x and y axis
    Args:
    fig: matplotlib seaborn figure
    ax1: matplotlib seaborn axis

    Returns:
    Updated fig and ax1
    """
    ax1.set(xlabel="", ylabel = "")
    ax1.xaxis.get_label().set_fontsize(40)
    ax1.yaxis.get_label().set_fontsize(40)
    return fig, ax1

# columns; 'id', 'group_id', 'user_id', 'user_name', 'message', 'name',
#                       'description', 'link', 'application', 'comment_count', 'likes_count',
#                       'shares_count', 'picture', 'story', 'created', 'updated', 'word_count'

##################################################################################
### ALL VISUALS FUNCTIONS                                                      ###
##################################################################################

def basics_lineplot(df, 
                    root_path:str, 
                    datatype:str):
    """Generates a line plot with numeric statistics of the dataset
       over time
    Args:
    df: pandas DataFrame, needs column "date" in Datetime format
    root_path: path of the current directory
    datatype: "posts" or "comments" for the Facebook data

    Returns:
    Saves the figure in the local directory
    """
    ic("Word count")
    df['word_count'] = df.text.str.split().str.len()

    ic("Base plot settings")
    fig, ax1, palette = set_base_plot_settings(fontsize=30, if_palette = True)
    ic("Line plot")
    ax1 = sns.lineplot(x="date", y="word_count",
                      palette = palette[0], 
                      label = "Words",
                        linewidth = 3, data = df)

    ax1 = sns.lineplot(x="date", y="comment_count",
                      palette = palette[1], 
                      label = "Comments",
                        linewidth = 3, data = df)

    ax1 = sns.lineplot(x="date", y="likes_count",
                      palette = palette[2], 
                      label = "Likes",
                        linewidth = 3, data = df)
    
    ax1 = sns.lineplot(x="date", y="shares_count",
                      palette = palette[3], 
                      label = "Shares",
                        linewidth = 3, data = df)
    
    ax1 = sns.lineplot(x="date", y="id",
                      palette = palette[4], 
                      label = "Number of posts",
                        linewidth = 3, data = df)
    
    ic("Late plot settings")
    fig, ax1 = set_late_plot_settings(fig, ax1, if_dates = True)

    ic("Add legend")
    leg = plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', facecolor='white')
    # set the linewidth of each legend object
    for legobj in leg.legendHandles:
        legobj.set_linewidth(10.0)
    
    ax1.xaxis_date(tz="UTC")
    date_form = mdates.DateFormatter("%Y")
    ax1.xaxis.set_major_formatter(date_form)

    ic("Save image")
    plot_name = f"{root_path}out/fig/specs/{datatype}_word_count.png"
    fig.savefig(plot_name, bbox_extra_artists=(leg,), bbox_inches='tight')
    
    ic("Save figure done\n------------------\n")

def jitter(values,j):
    """Creates slightly altered numbers to jitter them in a scatterplot
    Args:
    values: list of values to jitter (column from DataFrame on y-axis)
    j: dimensions

    Returns:
    jittered values
    """
    return values + np.random.normal(j,0.1,values.shape)

def posts_per_day_per_group_scatterplot(ori_df, 
                                        root_path:str, 
                                        datatype:str):
    """Generates a scatter plot with posts per day
    Args:
    ori_df: pandas DataFrame, needs column "date" in Datetime format
    root_path: path of the current directory
    datatype: "posts" or "comments" for the Facebook data

    Returns:
    Saves the figure in the local directory
    """
    df = ori_df.groupby(["date", "group_id"]).agg({"id": 'count'}).reset_index()
    ic(df.head())

    ic("Base plot settings")
    fig, ax1, palette = set_base_plot_settings(fontsize=30, if_palette = True)
    ic("Scatter plot")

    ax1 = sns.scatterplot(x="date", y=jitter(df["id"],2), 
                        hue="group_id", 
                        s = 10,
                        legend=False, data = df)
    
    ic("Late plot settings")
    fig, ax1 = set_late_plot_settings(fig, ax1, if_dates = True)
    
    ax1.xaxis_date(tz="UTC")
    date_form = mdates.DateFormatter("%Y")
    ax1.xaxis.set_major_formatter(date_form)

    ic("Save image")
    plot_name = f"{root_path}out/fig/specs/{datatype}_posts_per_day_per_group_scatter.png"
    fig.savefig(plot_name, bbox_inches='tight')
    
    ic("Save figure done\n------------------\n")

def posts_per_group_barplot(ori_df, 
                            root_path:str, 
                            datatype:str):
    """Generates a bar plot with posts per group
    Args:
    ori_df: pandas DataFrame
    root_path: path of the current directory
    datatype: "posts" or "comments" for the Facebook data

    Returns:
    Saves the figure in the local directory
    """
    df = ori_df.groupby("group_id").agg({"id": 'nunique'}).reset_index()
    n = 50
    ic(df.describe()) # Report this in the paper!
    df = df.sort_values('id', ascending=False)[0:n]

    ic("Base plot settings")
    fig, ax1, palette = set_base_plot_settings(fontsize=30, if_palette = True)
    ic("Bar plot")
    df["group_id"] = df["group_id"].astype(int).astype(str)
    ax1 = sns.barplot(y="group_id", x="id",
                      color = palette[7], 
                      #order = df.sort_values('id', ascending=False).group_id,
                      data = df)

    ic("Late plot settings")
    fig, ax1 = set_late_barplot_settings(fig, ax1)

    ax1.tick_params(labelsize=15)

    ic("Save image")
    plot_name = f"{root_path}out/fig/specs/{datatype}_posts_per_{n}_groups.png"
    fig.savefig(plot_name, bbox_inches='tight')
    ic("Save figure done\n------------------\n")

    df = ori_df.groupby("group_id").agg({"id": 'nunique'}).reset_index()
    ic(df.describe()) # Report this in the paper!
    df = df.sort_values('id', ascending=False)

    ic("Base plot settings")
    fig, ax1, palette = set_base_plot_settings(fontsize=30, if_palette = True)
    ic("Bar plot")
    df["group_id"] = df["group_id"].astype(int).astype(str)
    ax1 = sns.barplot(x="group_id", y="id",
                      color = palette[7], 
                      #order = df.sort_values('id', ascending=False).group_id,
                      data = df)

    ic("Late plot settings")
    fig, ax1 = set_late_barplot_settings(fig, ax1)

    ax1.tick_params(labelsize=15)
    ax1.set_xticklabels(ax1.get_xticklabels(),rotation = 50)

    ic("Save image")
    plot_name = f"{root_path}out/fig/specs/{datatype}_posts_per_group.png"
    fig.savefig(plot_name, bbox_inches='tight')
    ic("Save figure done\n------------------\n")

def unique_users_per_group_barplot(ori_df, 
                                   root_path:str, 
                                   datatype:str):
    """Generates a bar plot with unique users per group
    Args:
    ori_df: pandas DataFrame
    root_path: path of the current directory
    datatype: "posts" or "comments" for the Facebook data

    Returns:
    Saves the figure in the local directory
    """
    df = ori_df.groupby("group_id").agg({"user_id": 'nunique'}).reset_index()
    ic(df.describe()) #Report this in the paper!
    n=50
    df = df.sort_values("user_id", ascending=False)[:n]

    ic("Base plot settings")
    fig, ax1, palette = set_base_plot_settings(fontsize=30, if_palette = True)
    ic("Bar plot")
    df["group_id"] = df["group_id"].astype(int).astype(str)
    ax1 = sns.barplot(y="group_id", x="user_id",
                      color = palette[7], 
                      #order = df.sort_values('user_id', ascending=False).group_id,
                      data = df)

    ic("Late plot settings")
    fig, ax1 = set_late_barplot_settings(fig, ax1)

    ax1.tick_params(labelsize=15)

    ic("Save image")
    plot_name = f"{root_path}out/fig/specs/{datatype}_unique_users_per_{n}_groups.png"
    fig.savefig(plot_name, bbox_inches='tight')
    ic("Save figure done\n------------------\n")

    df = ori_df.groupby("group_id").agg({"user_id": 'nunique'}).reset_index()
    ic(df.describe()) #Report this in the paper!
    df = df.sort_values("user_id", ascending=False)

    ic("Base plot settings")
    fig, ax1, palette = set_base_plot_settings(fontsize=30, if_palette = True)
    ic("Bar plot")
    df["group_id"] = df["group_id"].astype(int).astype(str)
    ax1 = sns.barplot(x="group_id", y="user_id",
                      color = palette[7], 
                      #order = df.sort_values('user_id', ascending=False).group_id,
                      data = df)

    ic("Late plot settings")
    fig, ax1 = set_late_barplot_settings(fig, ax1)

    ax1.tick_params(labelsize=15)
    ax1.set_xticklabels(ax1.get_xticklabels(),rotation = 50)

    ic("Save image")
    plot_name = f"{root_path}out/fig/specs/{datatype}_unique_users_per_group.png"
    fig.savefig(plot_name, bbox_inches='tight')
    ic("Save figure done\n------------------\n")

def posts_users_scatterplot(ori_df, 
                            root_path:str, 
                            datatype:str):
    """Generates a scatter plot with posts vs users
    Args:
    ori_df: pandas DataFrame
    root_path: path of the current directory
    datatype: "posts" or "comments" for the Facebook data

    Returns:
    Saves the figure in the local directory
    """
    users = ori_df.groupby("group_id").agg({"user_id": 'nunique'}).reset_index()
    posts = ori_df.groupby("group_id").agg({"id": 'nunique'}).reset_index()
    
    df = pd.merge(left=users, right=posts, on="group_id")
    ic(df.head())

    ic("Base plot settings")
    fig, ax1, palette = set_base_plot_settings(fontsize=30, if_palette = True)
    ic("Scatter plot")
    
    ax1 = sns.scatterplot(x="id", y="user_id",
                      color = palette[6],
                      s = 15, 
                      #order = df.sort_values('user_id', ascending=False).group_id,
                      data = df)

    ic("Late plot settings")
    fig, ax1 = set_late_plot_settings(fig, ax1, if_dates = False)

    ax1.set_xlabel("Unique posts per group", fontsize = 40)
    ax1.set_ylabel("Unique users per group", fontsize = 40)

    ic("Save image")
    plot_name = f"{root_path}out/fig/{datatype}_posts_vs_unique_users.png"
    fig.savefig(plot_name, bbox_inches='tight')
    ic("Save figure done\n------------------\n")

def unique_users_over_time_lineplot(ori_df, 
                                    root_path:str, 
                                    datatype:str):
    """Generates a line plot with unique users over time
    Args:
    ori_df: pandas DataFrame, needs column "date" in Datetime format
    root_path: path of the current directory
    datatype: "posts" or "comments" for the Facebook data

    Returns:
    Saves the figure in the local directory
    """
    df = ori_df.groupby("date").agg({"user_id": 'count'}).reset_index()
    ic(df.describe())

    ic("Base plot settings")
    fig, ax1, palette = set_base_plot_settings(fontsize=30, if_palette = True)
    ic("Line plot")
    ax1 = sns.lineplot(x="date", y="user_id",
                      palette = palette[0], 
                        linewidth = 3, data = df)
    
    ic("Late plot settings")
    fig, ax1 = set_late_plot_settings(fig, ax1, if_dates = True)
    
    ax1.xaxis_date(tz="UTC")
    date_form = mdates.DateFormatter("%Y")
    ax1.xaxis.set_major_formatter(date_form)

    ic("Save image")
    plot_name = f"{root_path}out/fig/specs/{datatype}_unique_users_over_time.png"
    fig.savefig(plot_name, bbox_inches='tight')
    
    ic("Save figure done\n------------------\n")

def total_lifespan_per_group_pointplot(ori_df, root_path, datatype):
    """Generates a point plot with starting and ending dates for group activity
    Args:
    ori_df: pandas DataFrame, needs column "date" in Datetime format
    root_path: path of the current directory
    datatype: "posts" or "comments" for the Facebook data

    Returns:
    Saves the figure in the local directory
    """
    df = ori_df.groupby("group_id").agg({"date": ['min', 'max']}).reset_index()
    
    frame = {"group_id": df["group_id"], "date": df["date", "min"]}
    mindates = pd.DataFrame(data=frame)
    frame = {"group_id": df["group_id"], "date": df["date", "max"]}
    maxdates = pd.DataFrame(data=frame)
    
    combined = pd.concat([mindates, maxdates])
    combined["date"] = pd.to_datetime(combined["date"])
    combined['days'] = (maxdates['date'] - mindates['date']).dt.days
    combined = combined.sort_values("date").reset_index(drop=True).reset_index()

    ic("Base plot settings")
    fig, ax1, palette = set_base_plot_settings(fontsize=30, if_palette = True)
    ic("Line & Scatter plot")
    ic(len(combined["group_id"]))
    ax1 = sns.lineplot(x="date", y="days", hue="group_id",
                        palette = "tab10",
                        linewidth = 3,
                        data = combined, legend = False)
    ax1 = sns.scatterplot(x="date", y="days", hue="group_id",
                        palette = "tab10",
                        s = 200,
                        data = combined, legend = False)
    
    ic("Late plot settings")
    fig, ax1 = set_late_plot_settings(fig, ax1, if_dates = True)
    
    ax1.xaxis_date(tz="UTC")
    date_form = mdates.DateFormatter("%Y")
    ax1.xaxis.set_major_formatter(date_form)

    #ax1.set(ylim=(min(df["group_id"])-100, max(df["group_id"])+100))

    filename = f"{root_path}res/min_max_dates_per_group.csv"
    combined.to_csv(filename, index=False)

    ic("Save image")
    plot_name = f"{root_path}out/fig/specs/{datatype}_total_lifespan_per_group.png"
    fig.savefig(plot_name, bbox_inches='tight')
    
    ic("Save figure done\n------------------\n")


def output_descriptive_df(df):
    """Output a file that keeps numeric data only needed for visuals
    Args:
    df: pandas DataFrame with the original columns

    Returns:
    Saves a .csv file of numeric data, returns nothing
    """
    df["date"] = pd.to_datetime(df["created"], utc=True)
    df['word_count'] = df.text.str.split().str.len()
    ic("Select necessary columns")
    df = df[['id', 'date', 'group_id', 'user_id', 'comment_count', 'likes_count', 'shares_count', 'word_count']]
    ic("Save descriptive df")
    df.to_csv("/home/commando/marislab/facebook-posts/res/descriptive_only.csv", index=False)
    ic("Save finished")

def main(filename:str, 
         from_originals:bool, 
         datatype: str, 
         downsample_frequency=False):
    """Main function combining the rest
    filename: complete path to original json file
    from_originals: whether to load the original json or load a small sample
    datatype: "posts" or "comments" for Facebook data
    downsample_frequency: size of time bins if downsampling is desired
    """
    # Generate dataset with just numeric data needed for visuals
    #df = read_data(filename, from_originals)
    #output_descriptive_df(df)
    
    # Read in dataset with just numeric data for visuals
    # Check if this data has the same length and number of groups as the others did
    #ic("[INFO] Read in saved descriptive data")
    #df = pd.read_csv("/home/commando/marislab/facebook-posts/res/descriptive_only.csv")
    #ic(df.columns)

    ## WEEKLY COMPLETE DATA, DOWNSAMPLED
    root_path = "/home/commando/marislab/facebook-posts/"
    filename = f"{root_path}res/weekly_complete_daily_downsampled.csv"
    df = pd.read_csv(filename, sep=";")
    
    df["date"] = pd.to_datetime(df["date"], utc=True)

    ic("Select a few groups")
    small_df = df[df["group_id"].isin([3274, 3278, 3290, 3296, 3297, 4349])]
    #small_df = df[df["group_id"].isin([3297])]
    
    ic("Visualize:")
    ic("Basic lineplot")
    basics_lineplot(df, root_path, datatype)

    ic("Posts per day per group")
    posts_per_day_per_group_scatterplot(df, root_path, datatype)
    
    ic("Posts per group")
    posts_per_group_barplot(df, root_path, datatype)
    
    ic("Unique users per group")
    unique_users_per_group_barplot(df, root_path, datatype)
    
    ic("Unique users over time")
    unique_users_over_time_lineplot(df, root_path, datatype)
    
    ic("Posts vs Users: scatterplot")
    posts_users_scatterplot(df, root_path, datatype)

    ic("Toal lifespan per group")
    total_lifespan_per_group_pointplot(df, root_path, datatype)

def generate_numeric_test_df(n, begin_date):
    csiti = 23454
    units = list(range(0,n))
    texts = ["a word wow"] * n
    comment_count= [1]*n
    likes_count = [2]*n
    shares_count = [3]*n

    test_dataset = pd.DataFrame({'group_id':csiti, 
                   'id':units,
                   'text':texts,
                   'comment_count':comment_count,
                   'likes_count':likes_count,
                   'shares_count':shares_count,
                   'created':pd.date_range(begin_date, periods=n)})
    
    return test_dataset

def weekly_daily_create_grouplists(filename, from_originals, downsample_frequency, root_path):
    """This removes irrelevant/broken data and downsamples to weekly sets and then outputs the group_ids that
        post every single week
    """
    df = read_data(filename, from_originals)

    # Use a test dataset instead
    #df = generate_numeric_test_df(n=30, begin_date='2010-10-05')
    ic(df.head())
    
    ic("Original length", len(df))

    filename = f"{root_path}res/less_than_7.txt"
    with open(filename) as f:
        lines = f.readlines()
        group_ids_less_than_7 = [int(line.rstrip()) for line in lines]
    
    df = df[~df["group_id"].isin(group_ids_less_than_7)].reset_index(drop=True)
    ic("Removed groups less than 7 days", len(df), len(df.group_id.unique()))

    df["day"] = pd.to_datetime(df["created"]).dt.date
    day_fb_groups_were_introduced = pd.to_datetime("2010-10-06").date()
    df = df[df["day"] > day_fb_groups_were_introduced].reset_index(drop=True)
    ic("Removed posts from before FB groups were launched", len(df), len(df.group_id.unique()))
    ic(df.head())

    #out = df[df["year"] < 2010]
    #out.to_csv("pre2010_data.csv", index=False, sep=";")
    
    #df = df.drop([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    if downsample_frequency:
        df["date"] = pd.to_datetime(df["created"], utc=True)
        df, complete_groups, incomplete_groups = downsample_per_group(df, frequency=downsample_frequency)
        
        filename = f"{root_path}res/weekly_complete_group_ids.txt"
        with open(filename, 'w') as file:
            file.write('\n'.join(str(group) for group in complete_groups))
        filename = f"{root_path}res/weekly_incomplete_group_ids.txt"
        with open(filename, 'w') as file:
            file.write('\n'.join(str(group) for group in incomplete_groups))
        filename = f"{root_path}res/weekly_downsampled.csv"
        df.to_csv(filename, index=False, sep=";")
        ic(len(df))

    #visualize_posts_per_week(df, root_path, datatype)

def generate_analyzed_df(filename, from_originals, downsample_frequency, root_path):
    """This uses the previously found weekly groups, but downsamples the df to dates, 
        all non-posted dates will just have 0s.
    """
    df = read_data(filename, from_originals)

    # Use a test dataset instead
    #df = generate_numeric_test_df(n=30, begin_date='2010-10-05')
    ic(df.head())
    
    ic("Original length", len(df), len(df.group_id.unique()))

    filename = f"{root_path}res/weekly_complete_group_ids.txt"
    with open(filename) as f:
        lines = f.readlines()
        weekly_complete_groups = [int(line.rstrip()) for line in lines]
    
    df = df[df["group_id"].isin(weekly_complete_groups)].reset_index(drop=True)
    ic("Removed incomplete groups", len(df), len(df.group_id.unique()))
    
    #df = df.drop([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    if downsample_frequency:
        df["date"] = pd.to_datetime(df["created"], utc=True)
        df, complete_groups, incomplete_groups = downsample_per_group(df, frequency=downsample_frequency)
        
        filename = f"{root_path}res/weekly_complete_daily_downsampled.csv"
        df.to_csv(filename, index=False, sep=";")
        ic(len(df))

    #visualize_posts_per_week(df, root_path, datatype)


def visualize_posts_per_week(root_path, datatype, df=False):
    if not df:
        filename = f"{root_path}res/weekly_downsampled.csv"
        df = pd.read_csv(filename, sep=";")
    
    df["date"] = pd.to_datetime(df["date"])
    ic("[INFO] Visualize")
    ic("Base plot settings")
    fig, ax1, palette = set_base_plot_settings(fontsize=30, if_palette = True)
    ic("Line plot")
    
    ax1 = sns.scatterplot(x="date", y="id", hue="group_id",
                      #palette = palette[4], 
                        s = 100, data = df, legend=False)
    ax1 = sns.lineplot(x="date", y="id", hue="group_id",
                      #palette = palette[4], 
                        linewidth = 3, data = df, legend=False)
    
    ic("Late plot settings")
    fig, ax1 = set_late_plot_settings(fig, ax1, if_dates = True)
    
    ax1.xaxis_date(tz="UTC")
    date_form = mdates.DateFormatter("%b-%Y")
    ax1.xaxis.set_major_formatter(date_form)

    ic("Save image")
    plot_name = f"{root_path}out/fig/specs/{datatype}_posts_per_week_per_group_scatterlinepot.png"
    fig.savefig(plot_name, bbox_inches='tight')
    
    ic("DONE")

if __name__ == '__main__':
    data_path = "/data/datalab/danish-facebook-groups/raw-export/*"
    temp_path = "/home/commando/marislab/facebook-posts/tmp/tmp_df.csv"
    root_path = "/home/commando/marislab/facebook-posts/"

    from_originals = True
    datatype = "posts"
    downsample_frequency = '1D'
    
    files = glob.glob(data_path)
    filename = [i for i in files if datatype in i]
    filename = filename[0]
    ic(filename)
    
    main(filename, from_originals, datatype)
    #weekly_daily_create_grouplists(filename, from_originals, downsample_frequency, root_path)
    #generate_analyzed_df(filename, from_originals, downsample_frequency, root_path)

    visualize_posts_per_week(root_path, datatype)
    ic("DONE")