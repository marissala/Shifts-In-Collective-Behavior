# Detecting shifts in collective behavior in online groups
Main author: Maris Sala

This codebase includes codes developed to study tipping points in online social groups, specifically in Danish Facebook groups, with an option to expand that to include subreddits on Reddit.
The analysis uses change point detection and Markov models.

## Structure of this repository
    .
    ├── logs/                               # Log files - automatic and where the manually created logs remain
    ├── mdl/                                # Topic distributions stored as a .pcl file
    ├── out/                                # Results of different stages: figures and .pcl files
    │   ├── fig/       
    │   │   ├── hmm                         # Hidden Markov Model visuals as time series
    │   │   ├── hmm_model                   # Visualized model itself (states and probabilities)
    │   │   ├── specs                       # Descriptive figure of the dataset used
    │   ├── novelty-resonance/              # Novelty-resonance values stored as .pcl
    │   ├──group_sizes.csv                  #   Group ids with length of posts in group
    │   ├──posts_LDA_posts.csv              #   LDA results for Facebook posts: which post in which topic
    │   └──posts_LDA_topics.csv             #   LDA results for Facebook posts: which top keywords for the topics
    ├── res/                                # Resources
    ├── src/                                # Source codes
    │   ├── __pycache__         
    │   ├── __init__.py
    │   ├── choosing_groups.py              #   Uses regex to match group descriptions to choose interesting/uninteresting groups
    │   ├── downsampling.py                 #   Codes for downsampling based on date
    │   ├── generate_explore_df.py          #   Generates a short dataframe based on given group ID to explore the topic modeling results qualitatively
    │   ├── get_data_specs.py               #   Cleans the data and generates sample dataframes, visualizes basics to out/fig/specs/
    │   ├── hmm.py                          #   Hidden Markov Model codes together with Change Point Detection
    │   ├── join_posts_comments.py          #   In development, currently halted: joining Facebook posts with Facebook comments
    │   ├── main.py                         #   Main codes for reading in desired sample, doing LDA per group, calculates novelty-resonance, saves results
    │   ├── read_data.py                    #   Reads different forms of datasets, as needed
    │   └── visalize.py                     #   All necessary visualization codes
    ├── tmp/                                # Temporary folder with temp dfs for testing reasons
    ├── .gitignore
    ├── LICENCE
    └── README.md                           # Main information for this repository

## To use the codes
You'll need newsFluxus codes for getting novelty-resonance out, as well as a few other codebases that I might integrate into this codebase if I have the time

The basic pipeline looks something like this, given that the codes are run from the root folder:
```shell
python3 src/get_data_specs.py
python3 src/main.py
python3 src/hmm.py
```
``main.py`` generates the LDA outputs that are then used by ``hmm.py`` to detect the Markov states and change points. Results are stored in ``out/``. 