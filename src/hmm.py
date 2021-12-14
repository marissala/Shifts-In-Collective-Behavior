"""Developing applying Hidden Markov Models on topic distributions
Author: Maris Sala
Date: 12th Nov 2021
"""
import os
import pickle
import traceback
import pandas as pd
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
import ruptures as rpt
import networkx as nx
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.signal import savgol_filter
from hmmlearn import hmm
from icecream import ic
import signal
from contextlib import contextmanager

import sys
sys.path.insert(1, r'/home/commando/marislab/facebook-posts/src/')
from visualize import PlotHMMCPD

np.random.seed(42)

##################################################################################
### LOAD DATA                                                                  ###
##################################################################################

def load_from_premade_model(OUT_PATH: str, 
                            datatype: str):
    """Load the novelty-resonance values
    OUT_PATH: path of the directory
    datatype: posts or comments, in the filename

    Returns:
    dict of novelty-resonance value lists per group
    """
    ic("[INFO] Loading novelty-resonance...")
    with open(os.path.join(OUT_PATH, "out", "novelty-resonance", "{}_novelty-resonance.pcl".format(datatype)), 'rb') as f:
        out = pickle.load(f)
    return out

##################################################################################
### LIMIT TIME ALLCOATED TO FUNCTIONS                                          ###
##################################################################################

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

##################################################################################
### BIC & MODEL SELECTION                                                      ###
##################################################################################

def bic_general(likelihood_fn, k, X):
    """Calculates the BIC score for a model
    likelihood_fn: Function. Should take as input X and give out the log likelihood
                  of the data under the fitted model.
    k - int. Number of parameters in the model. The parameter that we are trying to optimize.
                    For HMM it is number of states.
                    For GMM the number of components.
    X - array. Data that been fitted upon.

    Returns:
    BIC score per model
    """
    bic = np.log(len(X))*k - 2*likelihood_fn(X)
    return bic

def bic_hmmlearn(X):
    """Tests which number of hidden states is optimal based on BIC values
    X: np.array of values for the HMM model

    Returns:
    Winning model and the BIC score associated
    """
    lowest_bic = np.infty
    bic = []
    n_states_range = range(1,3) #Might want to change the range
    
    for n_components in n_states_range:
        hmm_curr = hmm.MultinomialHMM(n_components=n_components)
        hmm_curr.fit(X)

        # Calculate number of free parameters
        # free_parameters = for_means + for_covars + for_transmat + for_startprob
        # for_means & for_covars = n_features*n_components
        n_features = hmm_curr.n_features
        free_parameters = 2*(n_components*n_features) + n_components*(n_components-1) + (n_components-1)

        bic_curr = bic_general(hmm_curr.score, free_parameters, X)
        bic.append(bic_curr)
        if bic_curr < lowest_bic:
            lowest_bic = bic_curr
            best_hmm = hmm_curr

    return best_hmm, bic

##################################################################################
### MAIN HMM CODE                                                              ###
##################################################################################

def everything_hmm(n_components,
                    n_iter,
                    input_X):
    """Fits the best HMM model on the data and returns the hidden states prediction
    n_components: 
    n_iter: 
    input_X: 

    Returns:
    Z: the predicted hidden states per datapoint
    """
    X = np.array(input_X).reshape(-1,1)

    #model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", n_iter=n_iter)
    model, bic = bic_hmmlearn(X)
    ic(bic)

    model.fit(X)
    Z = model.predict(X)

    ic(model.monitor_.converged)
    ic(model.score(X))

    return model, Z

##################################################################################
### CHANGE POINT DETECTION                                                     ###
##################################################################################

def change_point_detection(OUT_PATH, datatype, signal):
    algo = rpt.Pelt(model="rbf").fit(signal)
    result = algo.predict(pen=10)
    return result

def flatten(listOfLists):
    "Flatten one level of nesting"
    return chain.from_iterable(listOfLists)

##################################################################################
### MAIN                                                                       ###
##################################################################################

def main_novelty_resonance(OUT_PATH, datatype):
    out = load_from_premade_model(OUT_PATH, datatype)
    group_ids = ["5186"] #["12445"] #["3274", "3278"]#, "3290", "3296", "3297", "4349"]
    n_components = 3
    n_iter = 3200

    for group_id in group_ids:
        ic(group_id)
        try:
            #novelty = savgol_filter(out[group_id]["novelty"], 401, 1)
            #resonance = savgol_filter(out[group_id]["resonance"], 401, 1)
            comment = "GaussianHMM"
            novelty = savgol_filter(out[group_id]["novelty"], 21, 1)
            resonance = savgol_filter(out[group_id]["resonance"], 21, 1)
            
            ic("[INFO] HMM on novelty")
            model_nov, Z_nov = everything_hmm(n_components,
                            n_iter,
                            novelty)

            ic("[INFO] HMM on resonance")
            model_res, Z_res = everything_hmm(n_components,
                            n_iter,
                            resonance)
            
            #visualize_HMM(OUT_PATH, comment, group_id, novelty, resonance, Z_nov, Z_res)

            ic("[INFO] Change point detection: novelty")
            change_points_nov = change_point_detection(OUT_PATH, datatype, novelty)
            ic("[INFO] Change point detection: resonance")
            change_points_res = change_point_detection(OUT_PATH, datatype, resonance)

            ic("[INFO] Visualize")
            PlotVisuals.visualize_HMM_CPD(OUT_PATH, comment, group_id, novelty, resonance, Z_nov, Z_res, change_points_nov, change_points_res)
            ic(model_nov.transmat_)
            nr_of_states = len(set(Z_nov))
            simple_states = list(range(0,nr_of_states))
            lst1 = simple_states * nr_of_states # [0,1,2] * 2 = [0,1,2,0,1,2]
            lst2 = [[i]*nr_of_states for i in simple_states] # [0,0,1,1,2,2]
            lst2 = list(flatten(lst2))
            states = list(zip(lst1,lst2))
            ic(states)
            PlotVisuals.visualize_HMM_model(model_nov.transmat_, states, nr_of_states, OUT_PATH, group_id, comment)
            ic(f'Done with group: {group_id}')
            ic('------------------------------')
        except Exception:
            ic(f'Failed to process: {group_id}')
            ic(len(novelty))
            print(traceback.format_exc())
            ic('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

def generate_and_test(model, group_id, observ_name):
    observations, states = model.sample(1000)
    comment = "TEST"
    PlotHMMCPD.visualize_HMM_CPD(OUT_PATH, comment, group_id, 
                                        observ_name=observ_name, observations=list(flatten(observations)),
                                        states=states, change_points=[])

def main(OUT_PATH, datatype):
    #out = load_from_premade_model(OUT_PATH, datatype)
    observ_name = "daily_topics"
    filename = f"{OUT_PATH}out/{datatype}_LDA_posts.csv"
    df = pd.read_csv(filename, sep=";")
    ic(df.head())

    group_ids = df.group_id.unique() #["5186"] #["12445"] #["3274", "3278"]#, "3290", "3296", "3297", "4349"]
    n_components = 3
    n_iter = 3200

    for group_id in group_ids:
        ic(group_id)
        try:
            comment = "MultimodalHMM"

            df = df[df["group_id"]==group_id]
            df["date"] = pd.to_datetime(df["date"])
            ic(df.date)
            ic(len(df))
            df.index = df.date
            ic(len(df))

            daily_topics = np.array(df[df["group_id"]==group_id]["topic_nr"])
            ic("Unique topics: ", len(list(set(daily_topics))))

            ic("[INFO] HMM on daily topic nr")
            model_daily, Z_daily = everything_hmm(n_components,
                            n_iter,
                            daily_topics)

            generate_and_test(model_daily, group_id, observ_name)
            
            ic("[INFO] Change point detection: daily topic")
            change_points = change_point_detection(OUT_PATH, datatype, daily_topics)

            ic("[INFO] Visualize")
            PlotHMMCPD.visualize_HMM_CPD(OUT_PATH, comment, group_id, 
                                        observ_name=observ_name, observations=daily_topics,
                                        states=Z_daily, change_points=change_points)
            ic(model_daily.transmat_)
            nr_of_states = len(set(Z_daily))
            simple_states = list(range(0,nr_of_states))
            lst1 = simple_states * nr_of_states # [0,1,2] * 2 = [0,1,2,0,1,2]
            lst2 = [[i]*nr_of_states for i in simple_states] # [0,0,1,1,2,2]
            lst2 = list(flatten(lst2))
            states = list(zip(lst1,lst2))
            ic(states)
            PlotHMMCPD.visualize_HMM_model(observ_name, model_daily.transmat_, states, nr_of_states, OUT_PATH, group_id, comment)
            ic(f'Done with group: {group_id}')
            ic('------------------------------')
        except Exception:
            ic(f'Failed to process: {group_id}')
            ic(len(df))
            print(traceback.format_exc())
            ic('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')


if __name__ == '__main__':
    OUT_PATH = "/home/commando/marislab/facebook-posts/"
    datatype="posts"
    main(OUT_PATH, datatype)
