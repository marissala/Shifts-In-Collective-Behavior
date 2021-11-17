"""Developing applying Hidden Markov Models on topic distributions
Author: Maris Sala
Date: 12th Nov 2021
"""
    
import numpy as np
from hmmlearn import hmm
np.random.seed(42)

model = hmm.GaussianHMM(n_components=3, covariance_type="full")
print(model)
model.startprob_ = np.array([0.6, 0.3, 0.1])
print(model.startprob_)
model.transmat_ = np.array([[0.7, 0.2, 0.1],
                            [0.3, 0.5, 0.2],
                            [0.3, 0.3, 0.4]])
print(model.transmat_)
model.means_ = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])
print(model.means_)
model.covars_ = np.tile(np.identity(2), (3, 1, 1))

X, Z = model.sample(100)
print(X, Z)

lr = hmm.GaussianHMM(n_components=3, covariance_type="diag",
                     init_params="cm", params="cmt")
lr.startprob_ = np.array([1.0, 0.0, 0.0])
lr.transmat_ = np.array([[0.5, 0.5, 0.0],
                         [0.0, 0.5, 0.5],
                         [0.0, 0.0, 1.0]])
