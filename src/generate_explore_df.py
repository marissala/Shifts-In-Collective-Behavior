"""Explore what's up in a group
Author: Maris Sala
Date: 21 Dec 2021
"""

import pandas as pd

def main(OUT_PATH, datatype, group):
    filename = f"{OUT_PATH}out/{datatype}_LDA_posts.csv"
    df = pd.read_csv(filename, sep=";")

    explore = df[df["group_id"] == group].reset_index(drop=True).sort_values("topic_nr")

    explore.to_csv("explore.csv", index=False, sep=";")

if __name__ == '__main__':
    OUT_PATH = "/home/commando/marislab/facebook-posts/"
    datatype="posts"

    group = 4340
    main(OUT_PATH, datatype, group)
    print("DONE")
