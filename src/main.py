"""Topic modeling FB posts
Author: Maris Sala
Date: 15th Nov 2021
"""

import re
import os
import json
import glob
import sys
import time
import string
import logging
import pandas as pd
import numpy as np
import spacy
import pickle
import traceback
from icecream import ic

import gensim.corpora as corpora

import seaborn as sns; sns.set()
import pyplot_themes as themes
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.cbook.deprecation import MatplotlibDeprecationWarning

sys.path.insert(1, r'/home/commando/marislab/newsFluxus/src/')
from tekisuto.preprocessing import Tokenizer
from tekisuto.models import TopicModel
from tekisuto.models import InfoDynamics
from tekisuto.metrics import jsd
from preparations.preptopicmodeling import prepareTopicModeling
from visualsrc.visualsrc import plotVisualsrc
sys.path.insert(1, r'/home/commando/marislab/facebook-posts/src/') # current repository
from PreProcess.preprocess_texts import sent_to_words, parallelized_lemmatizer, make_lemmas

mgp = MovieGroupProcess(K=50, alpha=0.1, beta=0.1, n_iters=7)
preTM = prepareTopicModeling
pV = plotVisualsrc

##################################################################################
### GENSIM LDA FUNCTIONS                                                       ###
##################################################################################

def lda_modelling(df,
                  tokens: list,
                  OUT_PATH: str,
                  tune_topic_range=[10,30,50],
                  estimate_topics=False,
                  plot_topics=False,
                  **kwargs):
    """Runs basic LDA on texts, saves and returns a file with results
    Args:
    df: pandas DataFrame
    tokens: list of strings (document)
    OUT_PATH: location for saving the output
    tune_topic_range: number of topics to fit
    estimate topics: whether to search a range of topics
    plot_topics: quality check, plot coherence by topics
    **kwargs: other arguments to LDAmulticore
    
    Returns:
    Results of topic modeling
    """
    if estimate_topics:
        tm = TopicModel(tokens,
                        chunksize=2000, #how many documents at a time
                        passes = 20, #how many times does the whole corpus go through
                        iterations = 400, #how often we repeat a loop for a  document
                        alpha = 'auto',
                        eta = 'auto'
                        )
        n, n_cohers = tm.tune_topic_range(
            ntopics=tune_topic_range,
            plot_topics=plot_topics)
        ic(f"[INFO] Optimal number of topics is {n}")
        logging.info(f"LDA: Optimal number of topics is {n}")
        tm.fit(n, **kwargs)
    else:
        tm = TopicModel(tokens)
        n = 10
        n = tune_topic_range[0]
        tm.fit(n, **kwargs)
        n_cohers=0
    
    thetas = tm.get_topic_distribution()
    topics = tm.model.show_topics(num_topics=-1, num_words=15) # used to be 10 but need more meaningful topics
    topic_number = [np.argmax(thetas[post]) for post in range(len(thetas))]
    max_thetas = [max(thetas[post]) for post in range(len(thetas))]

    best_post_per_topic_df = pd.DataFrame({'topic_nr': topic_number, 
                                           "max_theta": max_thetas}).reset_index().groupby("topic_nr").max("max_theta").reset_index()
    out_best_topic_per_post = pd.DataFrame({'topic_nr': topic_number,
                                        'text': df['text'],
                                        'date': df["created"]})
    
    #id_top_posts = best_post_per_topic_df["index"]
    #top_posts = [tokens[i] for i in id_top_posts]
    ori_topic_numbers = best_post_per_topic_df["topic_nr"]
    topic_words = [re.findall(r'([a-zA-Z]+)', list(topics[i])[1]) for i in ori_topic_numbers]

    ic("[INFO] Writing content to file...")
    out_topics = pd.DataFrame({'topic_nr': ori_topic_numbers, 'topic_words': topic_words})
    #, 'top_post_tokens': top_posts, 
    #                    'topic_tune': [tune_topic_range]*len(ori_topic_numbers), 
    #                    'n_cohers': [n_cohers]*len(ori_topic_numbers)})
    
    out_topics.to_csv(os.path.join(OUT_PATH, "out", "{}_LDA_topics.csv".format(datatype)), index=False, sep=";")
    out_best_topic_per_post.to_csv(os.path.join(OUT_PATH, "out", "{}_LDA_posts.csv".format(datatype)), index=False, sep=";")
    
    dates = pd.to_datetime(df["created"])
    out = export_model_and_tokens(tm, 
                            n, 
                            tokens, 
                            thetas, 
                            dates, 
                            OUT_PATH, 
                            datatype)
    return out

def lda_modelling_per_group(ori_df,
                  tokens: list,
                  OUT_PATH: str,
                  tune_topic_range=[10,30,50],
                  estimate_topics=False,
                  plot_topics=False,
                  **kwargs):
    """Runs basic LDA modelling per group, saves and returns results
    Args:
    ori_df: pandas DataFrame
    tokens: list of strings (document)
    OUT_PATH: location for saving the output
    tune_topic_range: number of topics to fit
    estimate topics: whether to search a range of topics
    plot_topics: quality check, plot coherence by topics
    **kwargs: other arguments to LDAmulticore
    
    Returns:
    Results of topic modeling
    """
    out_topics = pd.DataFrame(columns=["group_id", "topic_nr", "topic_words"])
    out_best_topic_per_post = pd.DataFrame(columns=["group_id", "topic_nr", "text", "date"])

    group_ids = list(set(ori_df["group_id"]))
    ic(len(group_ids))

    out = {}
    for group_id in group_ids:
        ic(group_id)
        df = ori_df[ori_df["group_id"] == group_id]
        ic(len(df))
        tokens = df["tokens"]
        try:
            if estimate_topics:
                tm = TopicModel(tokens)
                n, n_cohers = tm.tune_topic_range(
                    ntopics=tune_topic_range,
                    plot_topics=plot_topics)
                ic(f"[INFO] Optimal number of topics is {n}")
                logging.info(f"LDA: Optimal number of topics is {n}")
                tm = TopicModel(tokens)
                tm.fit(n, **kwargs)
            else:
                tm = TopicModel(tokens)
                n = 10
                n = tune_topic_range[0]
                tm.fit(n, **kwargs)
                n_cohers=0
        except:
            ic(f"[INFO] Topic modeling of {group_id} failed")
            continue
    
        thetas = tm.get_topic_distribution()
        topics = tm.model.show_topics(num_topics=-1, num_words=15) # used to be 10 but need more meaningful topics
        topic_number = [np.argmax(thetas[post]) for post in range(len(thetas))]
        max_thetas = [max(thetas[post]) for post in range(len(thetas))]

        best_post_per_topic_df = pd.DataFrame({'topic_nr': topic_number, 
                                           "max_theta": max_thetas}).reset_index().groupby("topic_nr").max("max_theta").reset_index()
        ori_topic_numbers = best_post_per_topic_df["topic_nr"]
        topic_words = [re.findall(r'([a-zA-Z]+)', list(topics[i])[1]) for i in ori_topic_numbers]

        out_topics_group = pd.DataFrame({'group_id': group_id,'topic_nr': ori_topic_numbers, 'topic_words': topic_words})
        if "created" in df.columns:
            dates = df["created"]
        else:
            dates = df["date"]
        out_best_topic_per_post_group = pd.DataFrame({'group_id': group_id,
                                                      'topic_nr': topic_number,
                                                      'text': df['text'],
                                                      'date': dates})
        
        out_topics = pd.concat([out_topics, out_topics_group])
        out_best_topic_per_post = pd.concat([out_best_topic_per_post, out_best_topic_per_post_group])
        
        out = export_model_and_tokens_per_group(out, group_id, tm, n, tokens, thetas, dates)#, OUT_PATH, datatype)
    
    ic("[INFO] Writing content to file...")
    #out_topics.to_csv(os.path.join(OUT_PATH, "out", "{}_LDA_topics.csv".format(datatype)), index=False, sep=";")
    out_best_topic_per_post.to_csv(os.path.join(OUT_PATH, "out", "{}_LDA_posts.csv".format(datatype)), index=False, sep=";")
    
    #with open(os.path.join(OUT_PATH, "mdl", "topic_dist_per_group_{}.pcl".format(datatype)), "wb") as f:
    #    pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return out

def export_model_and_tokens_per_group(out,
                            id_nr,
                            tm, 
                            n:int, 
                            tokens:list, 
                            theta:list, 
                            dates:list):
    """Exports a json dictionary with topic modeling results per group
    Args:
    out: dictionary of main results
    id_nr: group_id of the group
    tm: topic model
    n: number of topics used for topic modeling
    tokens: list of tokens
    theta: list of theta values
    dates: list of dates

    Returns:
    fills the original dictionary with values for a single group's tm results
    """
    id_nr = str(int(id_nr))
    out[id_nr] = {}
    out[id_nr]["model"] = tm.model
    out[id_nr]["nr_of_topics"] = n
    out[id_nr]["tokenlists"] = tm.tokenlists
    out[id_nr]["tokens"] = tokens
    out[id_nr]["theta"] = theta
    out[id_nr]["dates"] = dates
    return out

def export_model_and_tokens(tm, 
                            n:int, 
                            tokens:list, 
                            theta:list, 
                            dates:list, 
                            OUT_PATH:str, 
                            datatype:str):
    """Exports a json dictionary with topic modeling results
    Args:
    tm: topic model
    n: number of topics used for topic modeling
    tokens: list of tokens
    theta: list of theta values
    dates: list of dates
    OUT_PATH: path to current directory
    datatype: "posts" or "comments" for Facebook data

    Returns:
    a dictionary with topic modeling results
    """
    out = {}
    out["model"] = tm.model
    out["nr_of_topics"] = n
    out["tokenlists"] = tm.tokenlists
    out["tokens"] = tokens
    out["theta"] = theta
    out["dates"] = dates
    with open(os.path.join(OUT_PATH, "mdl", "topic_dist_{datatype}.pcl"), "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out
    
def load_from_premade_model(OUT_PATH:str, 
                            datatype:str):
    """Loads topic distribution from premade model
    Args:
    OUT_PATH: path to current directory
    datatype: "posts" or "comments" for Facebook data
    
    Returns:
    premade model as a dictionary
    """
    ic("[INFO] Loading previous topic model...")
    with open(os.path.join(OUT_PATH, "mdl", "topic_dist_{}.pcl".format(datatype), 'rb')) as f:
        out = pickle.load(f)
    return out

def topicModeling(OUT_PATH, datatype, df, tokens, TOPIC_TUNE, file_exists=False):
    if file_exists:
        out = load_from_premade_model(OUT_PATH, datatype)
    else:
        tic = time.perf_counter()
        out = lda_modelling_per_group(df, tokens,
                            OUT_PATH,
                            TOPIC_TUNE,
                            estimate_topics=True)
    return out

##################################################################################
### PREPARE DATA                                                               ###
##################################################################################
    
def prepare_data(df, 
                LANG:str):
    """Lemmatizes and tokenizes texts
    Args:
    df: pandas DataFrame
    LANG: 2-letter language indicator, currently expects Danish only

    Returns:
    list of tokens and the input dataframe with added column of "tokens"
    """
    if LANG == "da":
        nlp = spacy.load("da_core_news_lg", disable=['parser', 'ner'])
    
    df["text"] = df["text"].astype(str)
    df["text"] = df["text"].str.lower()

    # Remove urls
    URL_pattern = r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))'''
    df['text'] = df['text'].str.replace(URL_pattern, '')
    df["text"] = df['text'].str.replace('[{}]'.format(string.punctuation), '')

    nlp.add_pipe('sentencizer')
    ic("[INFO] Lemmatizing...")
    tmp_df = df["text"].reset_index()
    tmp_df['text'] = tmp_df['text'].astype(str)
    lemmas = parallelized_lemmatizer(tmp_df)

    ic("[INFO] Tokenizing...")
    to = Tokenizer()
    tokens = to.doctokenizer(lemmas)
    ic(len(tokens), ic(len(df)))
    df["tokens"] = tokens
    return tokens, df

def read_json_data(filename:str):
    """Read the original large json file (Facebook data), slightly clean
    Args:
    filename: complete path to the .json file

    Returns:
    pandas DataFrame
    """
    ic("Read data")
    with open(filename) as f:
        giant_string = f.read()
    ic(len(giant_string))
    giant_string = giant_string[:-5]
    ic("Base cleaning")
    
    giant_string = re.sub('\\n', ' ', giant_string)
    giant_string = f'[{giant_string}]'
    
    ic("Json loads")
    data = json.loads(giant_string)
    
    ic(len(data))
    del giant_string
    df = pd.DataFrame.from_records(data)
    return df

##################################################################################
### NOVELTY, RESONANCE                                                         ###
##################################################################################

def extract_novelty_resonance(df, 
                             theta:list, 
                             dates:list, 
                             window:int):
    """Calculate novelty, transcience and resonance
    df: pandas DataFrame
    theta: list of theta values (list of lists)
    dates: list of dates
    window: int of the window size

    Returns:
    pandas DataFrame with novelty-resonance values
    """
    idmdl = InfoDynamics(data = theta, time = dates, window = window)
    idmdl.novelty(meas = jsd)
    idmdl.transience(meas = jsd)
    idmdl.resonance(meas = jsd)

    df["novelty"] = idmdl.nsignal
    df["transience"] = idmdl.tsignal
    df["resonance"] = idmdl.rsignal
    df["nsigma"] = idmdl.nsigma
    df["tsigma"] = idmdl.tsigma
    df["rsigma"] = idmdl.rsigma
    return df

def test_windows_extract_novelty_resonance(df, 
                             theta:list, 
                             dates:list, 
                             windows:list):
    """Calculate novelty, transcience and resonance
    df: pandas DataFrame
    theta: list of theta values (list of lists)
    dates: list of dates
    window: int of the window size

    Returns:
    pandas DataFrame with novelty-resonance values
    """
    for window in windows:
        idmdl = InfoDynamics(data = theta, time = dates, window = window)
        idmdl.novelty(meas = jsd)
        idmdl.transience(meas = jsd)
        idmdl.resonance(meas = jsd)
        
        window = str(window)
        df[f"novelty{window}"] = idmdl.nsignal
        #df[f"transience{window}"] = idmdl.tsignal
        df[f"resonance{window}"] = idmdl.rsignal
        #df[f"nsigma{window}"] = idmdl.nsigma
        #df[f"tsigma{window}"] = idmdl.tsigma
        #df[f"rsigma{window}"] = idmdl.rsigma
    return df

def export_novelty_per_group(out:dict,
                            id_nr:str,
                            novelty:list,
                            resonance:list):
    """Export novelty and resonance values per group
    Args:
    out: dictionary that holds values for all groups
    id_nr: group id of the group at hand
    novelty: list of novelty values
    resonance: list of resonance values

    Returns:
    dictionary
    """
    out[id_nr] = {}
    out[id_nr]["novelty"] = novelty
    out[id_nr]["resonance"] = resonance
    return out

def extractandplotNoveltyResonance(df, out, WINDOW, OUT_PATH):
    """Main function for extracting and visualising novelty-resonance,
        saves nov-res values in a file
    Args:
    df: 
    out: 
    WINDOW:
    OUT_PATH: 

    Returns
    """

    group_ids = []
    for group_id, group_info in out.items():
        ic(group_id)
        group_ids.append(group_id)
    
    ic(len(group_ids))
    nr_out = {}
    for group_id in group_ids:
        sample_df = df[df["group_id"] == group_id].reset_index(drop=True)
        sample_df = sample_df.sort_values("date")
        try:
            group_id = str(int(group_id))
            ic(group_id)
            nr_df = extract_novelty_resonance(sample_df, out[group_id]["theta"], out[group_id]["dates"], WINDOW)
            #novelty_transcience_resonance_lineplot(nrdf, OUT_PATH, datatype, group_id)
            ic("[INFO] Get novelty, resonance, beta1")
            time_var, novelty, resonance, beta1, xz, yz = pV.extract_adjusted_main_parameters(nr_df, WINDOW)
            
            pV.plot_initial_figures_facebook(novelty=nr_df["novelty"],
                            resonance=nr_df["resonance"],
                            xz=xz,
                            yz=yz,
                            OUT_PATH=OUT_PATH,
                            group_id=group_id,
                            datatype=datatype,
                            window=str(WINDOW))

            export_novelty_per_group(nr_out,
                            group_id,
                            novelty=nr_df["novelty"],
                            resonance=nr_df["resonance"])
            del nr_df
        except Exception:
            ic(f"[INFO] Failed to process {group_id}")
            ic(len(df))
            print(traceback.format_exc())
            continue
    with open(os.path.join(OUT_PATH, "out", "novelty-resonance", "{}_novelty-resonance.pcl".format(datatype)), "wb") as f:
        pickle.dump(nr_out, f, protocol=pickle.HIGHEST_PROTOCOL)


##################################################################################
### TESTING FUNCTIONS                                                          ###
##################################################################################

def generate_simple_test_df(n, begin_date):
    csiti = 23454
    units = list(range(0,n))
    texts = ["a word wow"] * n
    comment_count= [1]*n
    likes_count = [2]*n
    shares_count = [3]*n

    test_dataset = pd.DataFrame({'group_id':csiti, 
                   'id':units,
                   'text':texts,
                   'comment_count':comment_count,
                   'likes_count':likes_count,
                   'shares_count':shares_count,
                   'created':pd.date_range(begin_date, periods=n)})
    
    return test_dataset

def test_windows(datatype:str, 
        DATA_PATH:str, 
        OUT_PATH:str, 
        LANG:str):
    """Main function
    datatype: "posts" or "comments" for Facebook data
    DATA_PATH: path to data
    OUT_PATH: path to current directory
    LANG: two-letter abbreviation to language of dataset, currently only supports Danish ("da")
    """
    data_path = "/data/datalab/danish-facebook-groups/raw-export/*"
    WINDOW = 3
    windows = [1,3,7,30]

    logging.info("----- New iteration -----")

    ic("[INFO] Reading in the data")
    logging.info("[INFO] Reading in the data")
    ## DAILY COMPLETE DATA, DOWNSAMPLED
    df = pd.read_csv('{}tmp/tmp_df.csv'.format(OUT_PATH))
    ic(df.head())
    ic(len(df))
    
    if "text" not in df.columns:
        old_column_name = f"{datatype[:-1]}_text"
        try:
            df["text"] = df["message"]
            df = df.drop("message", axis=1)
        except:
            df["text"] = df[old_column_name]

    df['word_count'] = df.text.str.split().str.len()
    df["created"] = df["date"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    
    ic("[INFO] Prepare data...")
    logging.info("[INFO] Prepare data...")
    tic = time.perf_counter()
    tokens, df = prepare_data(df, LANG)
    toc = time.perf_counter()
    time_info = f"Prepared the data in {toc - tic:0.4f} seconds"
    ic(time_info)
    logging.info(f"Prepared the data in {toc - tic:0.4f} seconds")

    ic("[INFO] LDA modelling...")
    file_exists=False
    if file_exists:
        out = load_from_premade_model(OUT_PATH, datatype)
    else:
        TOPIC_TUNE = [10, 15]#,40,50]
        logging.info("[INFO] LDA modelling...")
        tic = time.perf_counter()
        out = lda_modelling_per_group(df, tokens,
                            OUT_PATH,
                            TOPIC_TUNE,
                            estimate_topics=True)
        toc = time.perf_counter()
        time_info = f"Finished LDA modelling in {toc - tic:0.4f} seconds"
        ic(time_info)
        logging.info(f"Finished LDA modelling in {toc - tic:0.4f} seconds")
    
    group_ids = list(set(df["group_id"]))
    ic(type(out["group_id"][0]))
    
    ic("[INFO] extracting novelty and resonance...")
    nr_out = {}
    for group_id in group_ids:
        sample_df = df[df["group_id"] == group_id].reset_index(drop=True)
        sample_df = sample_df.sort_values("date")
        try:
            #group_id = str(int(group_id))
            ic(type(out[group_id][0]))
            nr_df = test_windows_extract_novelty_resonance(sample_df, out[group_id]["theta"], out[group_id]["dates"], windows)
            ic("[INFO] Get novelty, resonance, beta1")
            time_var, novelty, resonance, beta1, xz, yz = pV.extract_adjusted_main_parameters(nr_df, WINDOW)
            out = pV.test_windows_extract_adjusted_main_parameters(nr_df, windows)
            
            for window in windows:
                window = str(window)
                pV.plot_initial_figures_facebook(novelty=nr_df[f"novelty{window}"],
                                resonance=nr_df[f"resonance{resonance}"],
                                xz=xz,
                                yz=yz,
                                OUT_PATH=OUT_PATH,
                                group_id=group_id,
                                datatype=datatype,
                                window=window)

            #export_novelty_per_group(nr_out,
            #                group_id,
            #                novelty=nr_df["novelty"],
            #                resonance=nr_df["resonance"])

            del nr_df
        except Exception:
            ic(f"[INFO] Failed to process {group_id}")
            ic(len(df))
            ic(traceback.format_exc())
            continue
    #with open(os.path.join(OUT_PATH, "out", "novelty-resonance", "{}_novelty-resonance.pcl".format(datatype)), "wb") as f:
    #    pickle.dump(nr_out, f, protocol=pickle.HIGHEST_PROTOCOL)

    ic("[INFO] PIPELINE FINISHED")
    logging.info("----- Finished iteration -----")

##################################################################################
### LOAD DATA AND FIXING FUNCTIONS                                             ###
##################################################################################

def check_columns(df):
    if "text" not in df.columns:
        old_column_name = f"{datatype[:-1]}_text"
        try:
            df["text"] = df["message"]
            df = df.drop("message", axis=1)
        except:
            df["text"] = df[old_column_name]

    if "created" in df.columns:
        df["date"] = pd.to_datetime(df["created"])
    else:
        df["date"] = pd.to_datetime(df["date"])

    return df

def exclude_broken_dates_rows(df):
    """Workaround function for some kind of data malfunction where dates get text instead of dates
        Loses only a few rows so could be alright
    Args:
    df: pandas df with row "date"

    Returns: df that does not include rows with broken dates, reset index
    """
    ic("Original length:", len(df))
    for index, row in df.iterrows():
        try:
            pd.to_datetime(row["date"])
        except:
            df = df.drop(index)
    df = df.reset_index(drop=True)
    ic("Dropped broken rows:", len(df))
    return df

def which_dataset_to_load(data_path, dataset_type:str):
    if dataset_type == "sample":
        ## SAMPLE DATA
        df = pd.read_table(f"tmp/tmp_df.csv", 
                            encoding='utf-8',
                            sep=";",
                            nrows=5000)
    elif dataset_type == "original":
        ## ORIGINAL WHOLE DATA
        files = glob.glob(data_path)
        filename = [i for i in files if datatype in i]
        filename = filename[0]
        df = read_json_data(filename)#[:10000]
        #df = df[df["group_id"].isin([3274, 3278, 3290, 3296, 3297, 4349])]
    elif dataset_type == "weekly_incomplete":
        ## WEEKLY INCOMPLETE DATA, DAILY DOWNSAMPLED
        filename = f"{OUT_PATH}res/weekly_incomplete_daily_downsampled.csv"
        df = pd.read_csv(filename, sep=";")
        ic(len(df))
        df = exclude_broken_dates_rows(df)
    elif dataset_type == "weekly_complete":
        ## WEEKLY COMPLETE DATA, DAILY DOWNSAMPLED
        filename = f"{OUT_PATH}res/weekly_complete_daily_downsampled.csv"
        df = pd.read_csv(filename, sep=";")
    df = df.sort_values("date")

    # Drop groups that have less than 10 datapoints
    dff = df.groupby("group_id").agg({"id":"nunique"}).reset_index()
    dff = dff[dff["id"] <= 10]
    dropped_group_ids = dff.group_id.unique()
    del dff
    logging.info("Dropped these groups due to lack of data: ")
    logging.info(dropped_group_ids)

    df = df[~df["group_id"].isin(dropped_group_ids)].reset_index(drop=True)
    return df

##################################################################################
### RUN THE ENTIRE PIPELINE PER FACEBOOK GROUP                                 ###
##################################################################################

def main(datatype:str, 
        DATA_PATH:str, 
        OUT_PATH:str, 
        LANG:str):
    """Main function
    datatype: "posts" or "comments" for Facebook data
    DATA_PATH: path to data
    OUT_PATH: path to current directory
    LANG: two-letter abbreviation to language of dataset, currently only supports Danish ("da")
    """
    data_path = "/data/datalab/danish-facebook-groups/raw-export/*"
    WINDOW = 7
    TOPIC_TUNE = [5, 10, 15]

    logging.info("----- New iteration -----")

    ic("[INFO] Reading in the data")
    logging.info("[INFO] Reading in the data")
    df = which_dataset_to_load(data_path, dataset_type = "weekly_incomplete")

    some_groups = df.group_id.unique()
    ic(some_groups)
    # Take the biggest groups only
    #some_groups = [4340,5722,6695,7339,7567,7892,8814,9352,9458,10124,11259,11816,12451,12451] #[5186]
    #df = df[df["group_id"].isin(some_groups)]
    ic(df.head())
    ic(len(df))
    ic(len(df.group_id.unique()))
    df = check_columns(df)

    ic("[INFO] Prepare data...")
    logging.info("[INFO] Prepare data...")
    tic = time.perf_counter()
    tokens, df = prepare_data(df, LANG)
    toc = time.perf_counter()
    time_info = f"Prepared the data in {toc - tic:0.4f} seconds"
    ic(time_info)
    logging.info(f"Prepared the data in {toc - tic:0.4f} seconds")

    ic("[INFO] LDA modelling...")
    logging.info("[INFO] LDA modelling...")
    tic = time.perf_counter()
    out = topicModeling(OUT_PATH, datatype, df, tokens, TOPIC_TUNE, file_exists=False)
    toc = time.perf_counter()
    time_info = f"Finished LDA modelling in {toc - tic:0.4f} seconds"
    ic(time_info)
    logging.info(f"Finished LDA modelling in {toc - tic:0.4f} seconds")
    
    ic("[INFO] Extracting novelty and resonance...")
    logging.info("[INFO] Novelty, resonance...")
    tic = time.perf_counter()
    extractandplotNoveltyResonance(df, out, WINDOW, OUT_PATH)
    toc = time.perf_counter()
    time_info = f"Finished novelty-resonance in {toc - tic:0.4f} seconds"
    ic(time_info)
    logging.info(f"Finished novelty-resonance in {toc - tic:0.4f} seconds")
    
    ic("[INFO] PIPELINE FINISHED")
    logging.info("----- Finished iteration -----")

if __name__ == '__main__':
    timestr = time.strftime("%Y_%m_%d-%H:%M")
    filename = f"logs/main_{timestr}.log"
    logging.basicConfig(filename=filename,
                        level=logging.DEBUG)
    datatype = "posts"
    activated = spacy.prefer_gpu()
    np.random.seed(1984)
    DATA_PATH = "/data/datalab/danish-facebook-groups"
    OUT_PATH = "/home/commando/marislab/facebook-posts/"
    LANG = 'da'

    main(datatype, DATA_PATH, OUT_PATH, LANG)
    #test_windows(datatype, DATA_PATH, OUT_PATH, LANG)
    