"""Trying to easily find interesting/noninteresting groups based on group description
Author: Maris Sala
Date: 21 Dec 2021
"""
import re
import pandas as pd
from icecream import ic

def save_uninteresting_group_ids(uninteresting_group_ids, OUT_PATH):
    filename = f"{OUT_PATH}res/uninteresting_group_ids.txt"
    textfile = open(filename, "w")
    for element in uninteresting_group_ids:
        textfile.write(str(int(element)) + "\n")
    textfile.close()

def regex_match_uninteresting_words(df):
    uninteresting_group_ids = []

    df["group_name"] = df["group_name"].str.lower()
    df["group_description"] = df["group_description"].str.lower()
    # they also say "you cannot sell buy here" so thats a bit complicated
    searchfor = [#'crochet', 'hækle', 
                'sell', 'selling', 'sælge', 'købe', 'køb', 'salg', 'bytte',
                'store', 'marketplace', 'markedsplads', 'butik', 'marked', 'market',
                'genbrug', 'reuse', 'second hand', 'secondhand', '2hand', 'storskrald', 'brugtmarked', 'brugt', 'discount', 'rabat',
                'housing', 'bolig', 'lejlighed',
                'tøj',
                'gratis']
    
    one = df[df.group_name.str.contains('|'.join(searchfor))]
    ic("buy/sell", len(one))
    ic(one.head())
    uninteresting_group_ids.extend(list(one["group_id"].unique()))
    
    searchfor = ['crochet', 'hækle', 'hekling',
                'sew', 'syning',
                'knit', 'strikke', 'strikning',
                'garn', 'fabric', 'yarn']
    one = df[df.group_name.str.contains('|'.join(searchfor))]
    ic("handicrafts", len(one))
    uninteresting_group_ids.extend(list(one["group_id"].unique()))

    searchfor = ['single', 'dating', 'kærlighed',
                'kærste', 'kæreste']
    one = df[df.group_name.str.contains('|'.join(searchfor))]
    ic("dating", len(one))
    uninteresting_group_ids.extend(list(one["group_id"].unique()))
    
    searchfor = ['fotograf', 'billede', 'photography']
    one = df[df.group_name.str.contains('|'.join(searchfor))]
    ic("photography", len(one))
    uninteresting_group_ids.extend(list(one["group_id"].unique()))

    ic(len(uninteresting_group_ids))
    ic(len(list(set(uninteresting_group_ids))))

    return uninteresting_group_ids
    
def main(OUT_PATH, datatype):
    filename = f"{OUT_PATH}res/group_descriptions.csv"
    df = pd.read_table(filename)
    #df.columns: Index(['ordering', 'Contry', 'group_id', 'group_name', 'group_description']

    ic("[INFO] Original length of df", len(df))
    
    df = df.dropna() # Might keep the NaNs in actually
    ic("[INFO] Length of df", len(df))
    ic(df.head())

    uninteresting_group_ids = regex_match_uninteresting_words(df)
    save_uninteresting_group_ids(uninteresting_group_ids, OUT_PATH)
    
    return 0

if __name__ == '__main__':
    OUT_PATH = "/home/commando/marislab/facebook-posts/"
    datatype="posts"

    main(OUT_PATH, datatype)