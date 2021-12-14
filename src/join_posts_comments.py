"""Saves descriptive only (no texts/messages) of comments and posts
Author: Maris Sala
Date: 13th December 2021
"""

import re
import json
import pandas as pd
import glob
from icecream import ic

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

def read_json_data_comments(filename: str):
    """Reads large json file that has problems with newlines
    Args:
    filename: complete path to json file

    Returns:
    pandas DataFrame
    """
    ic("[INFO] Read string")
    with open(filename) as f:
        giant_string = f.read()
    giant_string = giant_string[:-1]
    
    ic("[INFO] Base cleaning")
    giant_string = re.sub('\\n', ' ', giant_string)
    giant_string = f'[{giant_string}]'

    ic("[INFO] Json loads")
    data = json.loads(giant_string)
    ic(len(data))
    del giant_string
    df = pd.DataFrame.from_records(data)
    return df

def main():
    ic("[INFO] Read in posts")
    datatype = "posts"

    files = glob.glob(data_path)
    filename = [i for i in files if datatype in i]
    filename = filename[0]
    ic(filename)
    posts = read_json_data(filename)
    ic("[INFO] keep descriptive columns")
    posts = posts[["id", "group_id", "user_id", "comment_count", "likes_count", "shares_count", "created"]].drop_duplicates().reset_index(drop=True)
    posts["type"] = "posts"
    ic("[INFO] Save dataset")
    filename = f"{root_path}res/posts_descriptive.csv"
    posts.to_csv(filename, sep=";", index=False)
    
    ic("[INFO] Read in comments")
    datatype = "comments"

    files = glob.glob(data_path)
    filename = [i for i in files if datatype in i]
    filename = filename[0]
    ic(filename)
    comments = read_json_data_comments(filename)
    ic("[INFO] keep descriptive columns")
    comments = comments[["id", "post_id", "user_id", "likes_count", "created"]].drop_duplicates().reset_index(drop=True)
    comments = comments.rename(columns={'id': 'comment_id', 'post_id': 'id'})
    comments["type"] = "comments"
    ic("[INFO] Save dataset")
    filename = f"{root_path}res/comments_descriptive.csv"
    comments.to_csv(filename, sep=";", index=False)


if __name__ == '__main__':
    data_path = "/data/datalab/danish-facebook-groups/raw-export/*"
    temp_path = "/home/commando/marislab/facebook-posts/tmp/tmp_df.csv"
    root_path = "/home/commando/marislab/facebook-posts/"

    ic("STARTING")
    main()
    ic("DONE")