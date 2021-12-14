"""Storing downsampling based on date codes here
Author: Maris Sala
Date: 14th Dec 2021
"""

import pandas as pd
from icecream import ic

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
        df = df.resample(frequency).agg({"text": ' '.join, 
                                        "id": 'count', 
                                        "comment_count": 'sum', 
                                        "likes_count": 'sum', 
                                        "shares_count": 'sum'})

    df = df.reset_index(col_fill = "date")
    
    if not if_list:
        df["text"] = df["text"].astype(str)
    df["date"] = df["date"].astype(str)
    
    return df

def downsampleGroup(ori_df,
            frequency='1T',
            if_list=False,
            if_complete=True):
    """Downsamples based on date, concatenates texts. Per group.
    Args:
    ori_df: pandas DataFrame with columns "date" and "text"
    frequency: time interval for downsampling, default 1T - creates 1min timebins
    if_complete: whether you want downsampling-dimension-wise complete dataset as output

    Returns:
    Downsampled pandas DataFrame
    """
    ic("[START] Downsampling per group")
    group_ids = list(set(ori_df["group_id"].dropna()))
    ic(len(group_ids))

    resampled = pd.DataFrame()
    complete_groups = []
    incomplete_groups = []
    
    ic("[INFO] Looping over the groups...")
    for group_id in group_ids:
        df = ori_df[ori_df["group_id"] == group_id].reset_index(drop=True)
        df["date"] = pd.Series(df["date"])

        df.index = pd.to_datetime(df["date"])
        df = df.drop("date", axis = 1)
        
        if "text" in df.columns:
            df["text"] = df["text"].astype(str)
            df["text"] = pd.Series(df["text"])
            df = df.resample(frequency).agg({"text": ' '.join, 
                                            "id": 'count', 
                                            "user_id": 'nunique',
                                            "comment_count": 'sum', 
                                            "likes_count": 'sum', 
                                            "shares_count": 'sum'})
            df["text"] = df["text"].astype(str)
        else:
            df = df.resample(frequency).agg({"id": 'nunique', 
                                            "user_id": 'nunique',
                                            "comment_count": 'sum', 
                                            "likes_count": 'sum', 
                                            "shares_count": 'sum'})
        df = df.reset_index(col_fill = "date")
        df["date"] = df["date"].astype(str)
        df["group_id"] = group_id
        df.index = pd.to_datetime(df["date"]).dt.date
        missing = df[df.id == 0].index
        df = df.drop(df[df.id == 0].index)

        if len(missing) == 0:
            resampled = resampled.append(df)
            complete_groups.append(str(int(group_id)))
        else:
            if not if_complete:
                resampled = resampled.append(df)
            incomplete_groups.append(str(int(group_id)))
    
    ic(len(complete_groups))
    ic(len(incomplete_groups))
    ic("[END] Downsampling per group finished")
    return resampled, complete_groups, incomplete_groups
