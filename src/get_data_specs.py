"""Write out an overview of the dataset
Author: Maris Sala
Date: 8th Nov 2021
"""
import re
import ast
import glob
import json
import pandas as pd
from icecream import ic
from re import match
import jsonlines
import pprint

import seaborn as sns; sns.set()
import pyplot_themes as themes
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.cbook.deprecation import MatplotlibDeprecationWarning


def read_tmp_data(data_path):
    df = pd.read_csv(data_path, sep=";")
    ic(df.columns)
    ic(df.head())
    return df

def read_json_data(filename):
    ic("Read data")
    with open(filename) as f:
        giant_string = f.read()
    ic(len(giant_string))
    giant_string = giant_string[:-5]
    ic("Base cleaning")
    
    giant_string = re.sub('\\n', ' ', giant_string)
    giant_string = f'[{giant_string}]'
    
    ic("Json loads")
    data = json.loads(giant_string)
    
    ic(len(data))
    del giant_string
    df = pd.DataFrame.from_records(data)
    return df

def read_json_line_by_line(filename): # Doesnt work, leaving for now
    data = []
    with jsonlines.open(filename) as f:
        for line in f:
            print(line)
            data.append(json.loads(line))
    df = pd.DataFrame.from_records(data)
    return df

def read_data(filename, from_originals):
    if from_originals:
        df = read_json_data(filename) # read_json_data(filename)
    else:
        df = pd.read_csv("ds.csv")
    if "text" not in df.columns:
        df["text"] = df["message"]
    return df

def downsample(df,
            frequency='1T',
            if_list=False):
    """
    df: pandas DataFrame with columns "date" and "text"
    frequency: time interval for downsampling, default 1T - creates 1min timebins

    Returns downsampled pandas DataFrame
    """
    df = df.dropna().reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    print(len(df["text"]))
    print(len(df["date"]))
    df["text"] = pd.Series(df["text"])
    df["date"] = pd.Series(df["date"])

    df.index = pd.to_datetime(df["date"])
    df = df.drop("date", axis = 1)
    if if_list:
        df = df.resample(frequency).agg({"text": lambda x: list(x)})
    else:
        df = df.resample(frequency).agg({"text": ' '.join, "id": 'count', "comment_count": 'sum', "likes_count": 'sum', "shares_count": 'sum'})

    df = df.reset_index(col_fill = "date")
    #df = df[["text", "date"]]
    
    if not if_list:
        df["text"] = df["text"].astype(str)
    df["date"] = df["date"].astype(str)
    
    return df


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

# columns; 'id', 'group_id', 'user_id', 'user_name', 'message', 'name',
#                       'description', 'link', 'application', 'comment_count', 'likes_count',
#                       'shares_count', 'picture', 'story', 'created', 'updated'

def set_base_plot_settings(fontsize, if_palette):
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

def set_late_plot_settings(fig, ax1, if_dates):
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

def set_late_barplot_settings(fig, ax1):
    ax1.set(xlabel="", ylabel = "")
    ax1.xaxis.get_label().set_fontsize(40)
    ax1.yaxis.get_label().set_fontsize(40)
    return fig, ax1

# columns; 'id', 'group_id', 'user_id', 'user_name', 'message', 'name',
#                       'description', 'link', 'application', 'comment_count', 'likes_count',
#                       'shares_count', 'picture', 'story', 'created', 'updated', 'word_count'

def basics_lineplot(df, root_path, datatype):
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
    plot_name = f"{root_path}out/fig/{datatype}_word_count.png"
    fig.savefig(plot_name, bbox_extra_artists=(leg,), bbox_inches='tight')
    
    ic("Save figure done\n------------------\n")

def posts_per_day_per_group_scatterplot(ori_df, root_path, datatype):
    df = ori_df.groupby(["date", "group_id"]).agg({"id": 'count'}).reset_index()
    ic(df.head())

    ic("Base plot settings")
    fig, ax1, palette = set_base_plot_settings(fontsize=30, if_palette = True)
    ic("Line plot")

    import numpy as np
    def jitter(values,j):
        return values + np.random.normal(j,0.1,values.shape)

    ax1 = sns.scatterplot(x="date", y=jitter(df["id"],2), 
                        hue="group_id", 
                        s = 10,
                        #linewidth = 3, 
                        legend=False, data = df)
    
    ic("Late plot settings")
    fig, ax1 = set_late_plot_settings(fig, ax1, if_dates = True)
    
    ax1.xaxis_date(tz="UTC")
    date_form = mdates.DateFormatter("%Y")
    ax1.xaxis.set_major_formatter(date_form)

    ic("Save image")
    plot_name = f"{root_path}out/fig/{datatype}_posts_per_day_per_group_scatter.png"
    fig.savefig(plot_name, bbox_inches='tight')
    
    ic("Save figure done\n------------------\n")

def posts_per_group_barplot(ori_df, root_path, datatype):
    df = ori_df.groupby("group_id").agg({"id": 'nunique'}).reset_index()
    ic(sum(df.id))
    ic(df.describe()) # Report this in the paper!
    df = df.sort_values('id', ascending=False)[0:50]

    ic("Base plot settings")
    fig, ax1, palette = set_base_plot_settings(fontsize=30, if_palette = True)
    ic("Bar plot")
    df["group_id"] = df["group_id"].astype(str)
    ax1 = sns.barplot(y="group_id", x="id",
                      color = palette[7], 
                      #order = df.sort_values('id', ascending=False).group_id,
                      data = df)

    ic("Late plot settings")
    fig, ax1 = set_late_barplot_settings(fig, ax1)

    ax1.tick_params(labelsize=20)

    ic("Save image")
    plot_name = f"{root_path}out/fig/{datatype}_posts_per_group.png"
    fig.savefig(plot_name, bbox_inches='tight')
    ic("Save figure done\n------------------\n")

def unique_users_per_group_barplot(ori_df, root_path, datatype):
    df = ori_df.groupby("group_id").agg({"user_id": 'nunique'}).reset_index()
    ic(sum(df.user_id))
    ic(df.describe())
    ic(df.head())
    df = df.sort_values("user_id", ascending=False)[:50]

    ic("Base plot settings")
    fig, ax1, palette = set_base_plot_settings(fontsize=30, if_palette = True)
    ic("Bar plot")
    df["group_id"] = df["group_id"].astype(str)
    ax1 = sns.barplot(y="group_id", x="user_id",
                      color = palette[7], 
                      #order = df.sort_values('user_id', ascending=False).group_id,
                      data = df)

    ic("Late plot settings")
    fig, ax1 = set_late_barplot_settings(fig, ax1)

    ax1.tick_params(labelsize=20)

    ic("Save image")
    plot_name = f"{root_path}out/fig/{datatype}_unique_users_per_group.png"
    fig.savefig(plot_name, bbox_inches='tight')
    ic("Save figure done\n------------------\n")

def unique_users_over_time_lineplot(ori_df, root_path, datatype):
    df = ori_df.groupby("date").agg({"user_id": 'count'}).reset_index()
    ic(df.head())
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
    plot_name = f"{root_path}out/fig/{datatype}_unique_users_over_time.png"
    fig.savefig(plot_name, bbox_inches='tight')
    
    ic("Save figure done\n------------------\n")

def total_users_over_time(df, root_path, datatype):

    return 0

def output_descriptive_df(df):
    """Output a file that keeps numeric data only needed for visuals
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

def main(filename, 
         from_originals, 
         datatype: str, 
         downsample_frequency=False):
    """
    """
    # Generate dataset with just numeric data needed for visuals
    #df = read_data(filename, from_originals)
    #output_descriptive_df(df)
    
    # Read in dataset with just numeric data for visuals
    ic("[INFO] Read in saved descriptive data")
    df = pd.read_csv("/home/commando/marislab/facebook-posts/res/descriptive_only.csv")
    ic(df.columns)
    
    if downsample_frequency:
        df["date"] = pd.to_datetime(df["created"], utc=True)
        df = downsample(df, frequency=downsample_frequency)
        df.to_csv("ds.csv", index=False)
    
    df["date"] = pd.to_datetime(df["date"], utc=True)

    ic("Select a few groups")
    small_df = df[df["group_id"].isin([3274, 3278, 3290, 3296, 3297, 4349])]
    #small_df = df[df["group_id"].isin([3297])]
    ic(small_df.head())
    ic(len(small_df))
    
    ic("Visualize:")
    #ic("Basic lineplot")
    #basics_lineplot(df, root_path, datatype)

    #ic("Posts per day per group")
    #posts_per_day_per_group_scatterplot(df, root_path, datatype)
    
    ic("Posts per group")
    posts_per_group_barplot(df, root_path, datatype)
    
    ic("Unique users per group")
    unique_users_per_group_barplot(df, root_path, datatype)
    
    #ic("Unique users over time")
    #unique_users_over_time_lineplot(small_df, root_path, datatype)
    
    #ic("Total users over time")

    

if __name__ == '__main__':
    data_path = "/data/datalab/danish-facebook-groups/raw-export/*"
    temp_path = "/home/commando/marislab/facebook-posts/tmp/tmp_df.csv"
    root_path = "/home/commando/marislab/facebook-posts/"

    from_originals = True
    datatype = "posts"
    downsample_frequency = '1M'
    
    files = glob.glob(data_path)
    filename = [i for i in files if datatype in i]
    filename = filename[0]
    ic(filename)
    
    main(filename, from_originals, datatype)
    
    ic("DONE")