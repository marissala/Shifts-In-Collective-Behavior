"""Developing applying Hidden Markov Models on topic distributions
Author: Maris Sala
Date: 12th Nov 2021
"""
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

from scipy.signal import savgol_filter

from hmmlearn import hmm

from icecream import ic
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
    n_states_range = range(1,7)
    for n_components in n_states_range:
        hmm_curr = hmm.GaussianHMM(n_components=n_components, covariance_type='diag')
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

    return (best_hmm, bic)

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

    return Z

##################################################################################
### VISUALIZE                                                                  ###
##################################################################################

def visualize(X, Z):
    ic(len(X))
    x = np.linspace(0, len(X), num=len(X))
    ic(len(x))
    y1 = X
    #y2 = Z

    plt.figure(num = 3, figsize=(8, 5))
    # larger window length means smaller signal
    plt.plot(x, savgol_filter(y1, 101, 3), alpha = 0.3)
    plt.plot(x, savgol_filter(y1, 301, 1), 
            color='red',   
            linewidth=1#,  
            #linestyle='--' 
            )

    plt.savefig("fig1.png")

def vis_test(OUT_PATH, group_id, novelty, resonance, nov_states, res_states):
    colorlist = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

    #### NOVELTY ####
    novelty = savgol_filter(novelty, 401, 1)
    df = pd.DataFrame(dict(novelty=novelty, state=nov_states)).reset_index()
    df.columns = ["time", "novelty", "state"]
    fig, ax = plt.subplots()
    my_states = list(set(nov_states))
    colors = {}
    for i in my_states:
        colors[i] = colorlist[i]
    ax.scatter(df['time'], df['novelty'], 
                c=df['state'].map(colors),
                s=0.2)
    filename = f"{OUT_PATH}out/fig/hmm/{group_id}_HMM_gaussian_novelty.png"
    plt.savefig(filename)
    ic("DONE")

    #### RESONANCE ####
    resonance = savgol_filter(resonance, 401, 1)
    df = pd.DataFrame(dict(resonance=resonance, state=res_states)).reset_index()
    df.columns = ["time", "resonance", "state"]
    fig, ax = plt.subplots()
    my_states = list(set(res_states))
    colors = {}
    for i in my_states:
        colors[i] = colorlist[i]
    ax.scatter(df['time'], df['resonance'], 
                c=df['state'].map(colors),
                s=0.2)
    filename = f"{OUT_PATH}out/fig/hmm/{group_id}_HMM_gaussian_resonance.png"
    plt.savefig(filename)
    res_states = pd.DataFrame(res_states)
    res_states.to_csv("res_states.csv", index=False)
    ic("DONE")

def main(OUT_PATH, datatype):
    out = load_from_premade_model(OUT_PATH, datatype)    
    group_ids = ["3274", "3278", "3290", "3296", "3297", "4349"]
    n_components = 3
    n_iter = 20

    for group_id in group_ids:
        novelty = savgol_filter(out[group_id]["novelty"], 401, 1)
        resonance = savgol_filter(out[group_id]["resonance"], 401, 1)

        Z_nov = everything_hmm(n_components,
                        n_iter,
                        novelty)

        Z_res = everything_hmm(n_components,
                        n_iter,
                        resonance)

        vis_test(OUT_PATH, group_id, novelty, resonance, Z_nov, Z_res)


if __name__ == '__main__':
    OUT_PATH = "/home/commando/marislab/facebook-posts/"
    datatype="posts"
    main(OUT_PATH, datatype)
