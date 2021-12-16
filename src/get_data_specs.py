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
### LOAD DATA                                                                  ###
##################################################################################

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

def which_dataset_to_load(ori_filename, 
                            dataset_type:str, 
                            downsample_frequency=False):
    """Based on an if statement allows to load a differently preprocessed dataset
    Args:
    ori_filename: 
    dataset_type: "numeric", "original", "weekly_incomplete", "weekly_complete"
    downsample_frequency: 

    Returns: desired datasets with necessary modifications
    """
    if dataset_type == "numeric":
        # Generate dataset with just numeric data needed for visuals
        #df = readData(filename, from_originals)
        #df["date"] = pd.to_datetime(df["created"], utc=True)
        #output_descriptive_df(df)
        
        # Read in dataset with just numeric data for visuals
        ic("[INFO] Read in saved descriptive data")
        df = pd.read_csv("/home/commando/marislab/facebook-posts/res/descriptive_only.csv")
        ic(df.columns)
        if downsample_frequency:
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df, complete_groups, incomplete_groups = downsampleGroup(df, frequency=downsample_frequency, if_complete=False)
            ic(len(df), ic(len(df.group_id.unique())))
            #filename = f"{root_path}res/weekly_complete_daily_downsampled.csv"
            df = df.reset_index(drop=True)
            ic(len(df))
            ic(df.head())
    elif dataset_type == "original":
        df = readData(ori_filename, from_originals)
        df["date"] = pd.to_datetime(df["created"], utc=True)
    elif dataset_type == "weekly_incomplete":
        ## WEEKLY INCOMPLETE DATA, DOWNSAMPLED
        root_path = "/home/commando/marislab/facebook-posts/"
        filename = f"{root_path}res/weekly_complete_daily_downsampled.csv"
        df = pd.read_csv(filename, sep=";")
        df["date"] = pd.to_datetime(df["date"], utc=True)
        ic("Original len:", len(df))
        df = df.drop_duplicates().reset_index(drop=True)
        ic("Dropped duplicates length: ", len(df))
        #df["day"] = pd.to_datetime(df["created"]).dt.date
        #day_fb_groups_were_introduced = pd.to_datetime("2010-10-06").date()
        #df = df[df["day"] > day_fb_groups_were_introduced].reset_index(drop=True)
    elif dataset_type == "weekly_complete":
        ## WEEKLY COMPLETE DATA, DOWNSAMPLED
        root_path = "/home/commando/marislab/facebook-posts/"
        filename = f"{root_path}res/weekly_complete_daily_downsampled.csv"
        df = pd.read_csv(filename, sep=";")
    return df


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
### GENERATE LISTS OF GROUP IDS AS DESIRED                                     ###
##################################################################################

def weekly_daily_create_grouplists(filename, from_originals, downsample_frequency, root_path):
    """This removes irrelevant/broken data and downsamples to weekly sets and then outputs the group_ids that
        post every single week, as well as weekly incomplete posting groups.
        Removes posts made before groups were introduced, and groups that are less than 7 days in existence.
    Args:
    filename: filename for the data to load
    from_originals: boolean, whether to generate data from the original ndjson file
    downsample_frequency: frequency of resampling the dataset
    root_path: path to this directory
    
    Returns: saves three files: ids for complete groups, incomplete groups, and a dataframe.
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

##################################################################################
### GENERATE ANALYSED DATAFRAME                                                ###
##################################################################################

def generate_analyzed_df(filename:str, 
                        from_originals:bool, 
                        downsample_frequency, 
                        root_path:str):
    """This uses the previously found weekly groups, but downsamples the df to dates, 
        all non-posted dates will just have 0s.
    Args:
    filename: filename for the data to load
    from_originals: boolean, whether to generate data from the original ndjson file
    downsample_frequency: frequency of resampling the dataset
    root_path: path to this directory
    
    Returns: saves a dataset that should be used for the analysis
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

##################################################################################
### GENERATE ADDITIONAL HELPER DATASETS                                        ###
##################################################################################

def generate_numeric_test_df(n:int, 
                            begin_date:str):
    """Generates a test dataset that does not include "text". For testing purposes
    Args:
    n: number of rows
    begin_date: string in the format 'YYYY-MM-DD', starts sampling dates from this date

    Returns: sample dataset
    """
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

def save_group_df_lengths(df, 
                            root_path:str):
    """Finds the number of posts made in each group, orders them and saves in file
    Args:
    df: pandas DataFrame
    root_path: path to the current directory
    Returns: saves a resource file of group ids and their lengths
    """
    df = df.groupby("group_id").agg({'id': 'nunique'}).reset_index()
    df = df[["group_id", "id"]]
    df = df.rename(columns={'id': 'length'})
    df = df.sort_values("length", ascending=False)
    filename = f"{root_path}/res/group_lengths.csv"
    df.to_csv(filename, index=False)

def select_sample_df(list_of_groups:lst, 
                    df):
    """Selects specific groups from df
    Args:
    list_of_groups: group numbers in a list, if just one group, specify as a list anyway
    df: pandas DataFrame
    
    Returns: small dataframe with the selected groups and reindexed
    """
    small_df = df[df["group_id"].isin(list_of_groups)].reset_index(drop=True)
    return small_df

##################################################################################
### MAIN - DESCRIBE & VISUALISE DATA                                           ###
##################################################################################

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
    if_testing = False

    df = which_dataset_to_load(filename, dataset_type="weekly_incomplete")
    save_group_df_lengths(df, root_path)
    comment = "weekly_incomplete"
    
    if if_testing:
        ic("Select a few groups")
        list_of_groups = [3297]
        df = select_sample_df(list_of_groups, df)

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
    
    PlotVisuals.basicsLineplot(df, root_path, datatype, comment)

    ic("Posts per day per group")
    PlotVisuals.posts_per_day_per_group_scatterplot(df, root_path, datatype, comment)
    
    ic("Posts per group")
    PlotVisuals.posts_per_group_barplot(df, root_path, datatype, comment)
    
    ic("Unique users per group")
    PlotVisuals.unique_users_per_group_barplot(df, root_path, datatype, comment)
    
    ic("Unique users over time")
    PlotVisuals.unique_users_over_time_lineplot(df, root_path, datatype, comment)
    
    ic("Posts vs Users: scatterplot")
    PlotVisuals.posts_users_scatterplot(df, root_path, datatype, comment)

    ic("Toal lifespan per group")
    PlotVisuals.total_lifespan_per_group_pointplot(df, root_path, datatype, comment)

    ic("Visualize posts per week")
    PlotVisuals.visualize_posts_per_week(root_path, datatype, comment, df)

if __name__ == '__main__':
    data_path = "/data/datalab/danish-facebook-groups/raw-export/*"
    temp_path = "/home/commando/marislab/facebook-posts/tmp/tmp_df.csv"
    root_path = "/home/commando/marislab/facebook-posts/"

    from_originals = True
    datatype = "posts"
    downsample_frequency = '1D'
    if_full_pipeline = False
    
    files = glob.glob(data_path)
    filename = [i for i in files if datatype in i]
    filename = filename[0]
    ic(filename)
    
    if if_full_pipeline:
        weekly_daily_create_grouplists(filename, from_originals, downsample_frequency, root_path)
        generate_analyzed_df(filename, from_originals, downsample_frequency, root_path)
        main(filename, from_originals, datatype, downsample_frequency='1W')
    else:
        main(filename, from_originals, datatype, downsample_frequency='1W')

    ic("DONE")