"""Developing applying Hidden Markov Models on topic distributions
Author: Maris Sala
Date: 12th Nov 2021
"""
import os
import pickle
import pandas as pd
import numpy as np
from hmmlearn import hmm

from icecream import ic
np.random.seed(42)

def load_from_premade_model(OUT_PATH, datatype):
    ic("[INFO] Loading novelty-resonance...")
    with open(os.path.join(OUT_PATH, "out", "novelty-resonance", "{}_novelty-resonance.pcl".format(datatype)), 'rb') as f:
        out = pickle.load(f)
    return out

def main(OUT_PATH, datatype):
    out = load_from_premade_model(OUT_PATH, datatype)    
    group_id = "4349"

    novelty = out[group_id]["novelty"]
    resonance = out[group_id]["resonance"]

if __name__ == '__main__':
    OUT_PATH = "/home/commando/marislab/facebook-posts/"
    datatype="posts"
    main(OUT_PATH, datatype)