"""Function to read in the data, makes small samples of the datasets to work with for code dev
Author: Maris Sala
Date: 8th Nov
"""
import re
import json
import ast
import glob
import pandas as pd
import pprint
from icecream import ic

# dk_groups_clean.json  dk_groups_comments_clean.json  dk_groups_posts_clean.json

def read_tmp_data(data_path: str):
    """Reads a temporary smaller dataset for testing
    Args:
    data_path: complete path of temporary dataset (.csv)
    
    Returns:
    pandas DataFrame
    """
    df = pd.read_csv(data_path, sep=";")
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
    del giant_string
    ic("[INFO] Convert to pandas df")
    df = pd.DataFrame.from_records(data)
    return df

def readData(filename: str, 
              from_originals: bool):
    """Reads in the original large json file, or a smaller saved sample
    Args:
    filename: complete path to json file
    from_originals: if user wants to load original json or a sample dataset
    
    Returns:
    pandas DataFrame
    """
    ic("[START] Reading in data")
    if from_originals:
        df = read_json_data(filename)
    else:
        df = pd.read_csv("ds.csv")
    if "text" not in df.columns:
        df["text"] = df["message"]
    ic("[END] Finished reading in data")
    return df


if __name__ == '__main__':
    ic("START PIPELINE---------------------------------")
    data_path = "/data/datalab/danish-facebook-groups/raw-export/*"
    filenames = glob.glob(data_path)
    df = readData(filenames[2])
    #dff = df[:100]
    #dff.to_csv("/home/commando/marislab/facebook-posts/tmp/tmp_df.csv", index = False, sep=";")
    ic(df.head())
    ic("ALL DONE---------------------------------------")