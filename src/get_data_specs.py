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

import sys
sys.path.insert(1, r'/home/commando/marislab/facebook-posts/src/')
from read_data import readData
from downsampling import downsample, downsampleGroup
from visualize import PlotSettings, PlotVisuals

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
    df = readData(filename, from_originals)

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
        df, complete_groups, incomplete_groups = downsampleGroup(df, frequency=downsample_frequency)
        
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
    df = readData(filename, from_originals)

    # Use a test dataset instead
    #df = generate_numeric_test_df(n=30, begin_date='2010-10-05')
    ic(df.head())
    
    ic("Original length", len(df), len(df.group_id.unique()))

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

    # Ignore weekly complete groups for now
    #filename = f"{root_path}res/weekly_complete_group_ids.txt"
    #with open(filename) as f:
    #    lines = f.readlines()
    #    weekly_complete_groups = [int(line.rstrip()) for line in lines]
    
    #df = df[df["group_id"].isin(weekly_complete_groups)].reset_index(drop=True)
    #ic("Removed incomplete groups", len(df), len(df.group_id.unique()))
    
    #df = df.drop([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    if downsample_frequency:
        df["date"] = pd.to_datetime(df["created"], utc=True)
        df, complete_groups, incomplete_groups = downsampleGroup(df, frequency=downsample_frequency, if_complete=False)
        ic(len(df), ic(len(df.group_id.unique())))
        ic(len(complete_groups))
        ic(len(incomplete_groups))
        #filename = f"{root_path}res/weekly_complete_daily_downsampled.csv"
        df.text = df.text.str.replace('[^\w\s]','') # REMOVE PUNCTUATION FROM TEXT so I can use ; in saving the df
        filename = f"{root_path}res/weekly_incomplete_daily_downsampled.csv"
        df.to_csv(filename, index=False, sep=";")
        ic(len(df))

    #visualize_posts_per_week(df, root_path, datatype)

def main(filename:str,
        from_originals:bool,
        datatype:str,
        downsample_frequency=False):
    """Main function combining the rest
    filename: complete path to original json file
    from_originals: whether to load the original json or load a small sample
    datatype: "posts" or "comments" for Facebook data
    downsample_frequency: size of time bins if downsampling is desired
    """
    ic(datatype)
    # Generate dataset with just numeric data needed for visuals
    #df = readData(filename, from_originals)
    #df["date"] = pd.to_datetime(df["created"], utc=True)
    #output_descriptive_df(df)
    
    # Read in dataset with just numeric data for visuals
    # Check if this data has the same length and number of groups as the others did
    #ic("[INFO] Read in saved descriptive data")
    #df = pd.read_csv("/home/commando/marislab/facebook-posts/res/descriptive_only.csv")
    #ic(df.columns)
    #df = df[df["group_id"].isin([3274, 3278, 3290, 3296, 3297, 4349])]
    #if downsample_frequency:
    #    df["date"] = pd.to_datetime(df["date"], utc=True)
    #    df, complete_groups, incomplete_groups = downsampleGroup(df, frequency=downsample_frequency, if_complete=False)
    #    ic(len(df), ic(len(df.group_id.unique())))
    #    #filename = f"{root_path}res/weekly_complete_daily_downsampled.csv"
    #    df = df.reset_index(drop=True)
    #    ic(len(df))
    #    ic(df.head())

    ## WEEKLY COMPLETE DATA, DOWNSAMPLED
    #root_path = "/home/commando/marislab/facebook-posts/"
    #filename = f"{root_path}res/weekly_complete_daily_downsampled.csv"
    #df = pd.read_csv(filename, sep=";")

    ## WEEKLY INCOMPLETE DATA, DOWNSAMPLED
    #root_path = "/home/commando/marislab/facebook-posts/"
    filename = f"{root_path}res/weekly_complete_daily_downsampled.csv"
    df = pd.read_csv(filename, sep=";")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    ic("Original len:", len(df))
    df = df.drop_duplicates().reset_index(drop=True)
    ic("Dropped duplicates length: ", len(df))
    #df["day"] = pd.to_datetime(df["created"]).dt.date
    #day_fb_groups_were_introduced = pd.to_datetime("2010-10-06").date()
    #df = df[df["day"] > day_fb_groups_were_introduced].reset_index(drop=True)

    #ic("Select a few groups")
    #small_df = df[df["group_id"].isin([3274, 3278, 3290, 3296, 3297, 4349])]
    #small_df = df[df["group_id"].isin([3297])]

    ic("Describe")
    ic(df.columns)
    ic(len(df))
    ic(len(df.group_id.unique()))
    ic(sum(df.id))
    ic(sum(df.user_id))
    ic(sum(df.comment_count))
    ic(sum(df.likes_count))
    ic(sum(df.shares_count))
    
    ic("Visualize:")
    ic("Basic lineplot")
    datatype = datatype + "_weekly_incomplete_minus_2010_"
    PlotVisuals.basicsLineplot(df, root_path, datatype)

    ic("Posts per day per group")
    PlotVisuals.posts_per_day_per_group_scatterplot(df, root_path, datatype)
    
    ic("Posts per group")
    PlotVisuals.posts_per_group_barplot(df, root_path, datatype)
    
    ic("Unique users per group")
    PlotVisuals.unique_users_per_group_barplot(df, root_path, datatype)
    
    ic("Unique users over time")
    PlotVisuals.unique_users_over_time_lineplot(df, root_path, datatype)
    
    ic("Posts vs Users: scatterplot")
    PlotVisuals.posts_users_scatterplot(df, root_path, datatype)

    ic("Toal lifespan per group")
    PlotVisuals.total_lifespan_per_group_pointplot(df, root_path, datatype)

    ic("Visualize posts per week")
    PlotVisuals.visualize_posts_per_week(root_path, datatype)


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
    
    #weekly_daily_create_grouplists(filename, from_originals, downsample_frequency, root_path)
    #generate_analyzed_df(filename, from_originals, downsample_frequency, root_path)
    main(filename, from_originals, datatype, downsample_frequency='1W')

    ic("DONE")