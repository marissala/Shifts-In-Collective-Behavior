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
import logging
import pandas as pd
import numpy as np
import spacy
import pickle
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
sys.path.insert(1, r'/home/commando/marislab/gsdmm/')
from gsdmm import MovieGroupProcess
sys.path.insert(1, r'/home/commando/marislab/compare-short-topic-modeling/src/')
from PreProcess.preprocess_texts import sent_to_words, parallelized_lemmatizer, make_lemmas

mgp = MovieGroupProcess(K=50, alpha=0.1, beta=0.1, n_iters=7)
preTM = prepareTopicModeling

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
    """
    Args:
    tokens: list of strings (document)
    OUT_PATH: location for saving the output
    estimate topics: whether to search a range of topics
    tune_topic_range: number of topics to fit
    plot_topics: quality check, plot coherence by topics
    **kwargs: other arguments to LDAmulticore
    
    Returns:
    Nothing
    """
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

def lda_modelling_per_group(df,
                  tokens: list,
                  OUT_PATH: str,
                  tune_topic_range=[10,30,50],
                  estimate_topics=False,
                  plot_topics=False,
                  **kwargs):
    """
    Args:
    tokens: list of strings (document)
    OUT_PATH: location for saving the output
    estimate topics: whether to search a range of topics
    tune_topic_range: number of topics to fit
    plot_topics: quality check, plot coherence by topics
    **kwargs: other arguments to LDAmulticore
    
    Returns:
    Nothing
    """
    group_ids = list(set(df["group_id"]))
    out_topics = pd.DataFrame(columns=["group_id", "topic_nr", "topic_words"])
    out_best_topic_per_post = pd.DataFrame(columns=["group_id", "topic_nr", "text", "date"])
    
    ic(group_ids)
    
    out = {}
    for group_id in group_ids:
        ic(group_id)
        df = df[df["group_id"] == group_id]
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
        out_best_topic_per_post_group = pd.DataFrame({'group_id': group_id,
                                                      'topic_nr': topic_number,
                                                      'text': df['text'],
                                                      'date': df["created"]})
        
        out_topics = pd.concat([out_topics, out_topics_group])
        out_best_topic_per_post = pd.concat([out_best_topic_per_post, out_best_topic_per_post_group])
        
        dates = pd.to_datetime(df["created"])
        out = export_model_and_tokens_per_group(out, group_id, tm, n, tokens, thetas, dates, OUT_PATH, datatype)
    
    ic("[INFO] Writing content to file...")
    out_topics.to_csv(os.path.join(OUT_PATH, "out", "{}_LDA_topics.csv".format(datatype)), index=False, sep=";")
    out_best_topic_per_post.to_csv(os.path.join(OUT_PATH, "out", "{}_LDA_posts.csv".format(datatype)), index=False, sep=";")
    
    with open(os.path.join(OUT_PATH, "mdl", "topic_dist_per_group_{}.pcl".format(datatype)), "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return out

def export_model_and_tokens_per_group(out,
                            id_nr,
                            tm, 
                            n, 
                            tokens, 
                            theta, 
                            dates, 
                            OUT_PATH, 
                            datatype):
    id_nr = id_nr.astype(str)
    out[id_nr]["model"] = tm.model
    out[id_nr]["nr_of_topics"] = n
    #out["id2word"] = tm.id2word
    #out["corpus"] = tm.corpus
    out[id_nr]["tokenlists"] = tm.tokenlists
    out[id_nr]["tokens"] = tokens   #add?
    out[id_nr]["theta"] = theta
    out[id_nr]["dates"] = dates
    return out

def export_model_and_tokens(tm, 
                            n, 
                            tokens, 
                            theta, 
                            dates, 
                            OUT_PATH, 
                            datatype):
    out = {}
    out["model"] = tm.model
    out["nr_of_topics"] = n
    #out["id2word"] = tm.id2word
    #out["corpus"] = tm.corpus
    out["tokenlists"] = tm.tokenlists
    out["tokens"] = tokens   #add?
    out["theta"] = theta
    out["dates"] = dates
    with open(os.path.join(OUT_PATH, "mdl", "topic_dist_{datatype}.pcl"), "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out
    
def load_from_premade_model(OUT_PATH, datatype):
    ic("[INFO] Loading previous topic model...")
    with open(os.path.join(OUT_PATH, "mdl", "topic_dist_{}.pcl".format(datatype), 'rb')) as f:
        out = pickle.load(f)
    return out

##################################################################################
### PREPARE DATA                                                               ###
##################################################################################
    
def prepare_data(df, LANG):
    if LANG == "da":
        nlp = spacy.load("da_core_news_lg", disable=['parser', 'ner'])
    #texts = df.text.values.tolist()
    #texts_words = list(sent_to_words(texts))
    
    nlp.add_pipe('sentencizer')
    ic("[INFO] Lemmatizing...")
    #tmp = map(' '.join, texts_words)
    #lemmas = preTM.make_lemmas(df, nlp, LANG)
    tmp_df = df["text"].reset_index()
    tmp_df['text'] = tmp_df['text'].astype(str)
    lemmas = parallelized_lemmatizer(tmp_df) #make_lemmas(texts_words, nlp) #spacy_lemmatize(tmp, nlp)
    
    ic("[INFO] Tokenizing...")
    to = Tokenizer()
    tokens = to.doctokenizer(lemmas)
    ic(len(tokens), ic(len(df)))
    df["tokens"] = tokens
    #tokens, df = remove_invalid_entries(tokens, df)
    return tokens, df

def read_json_data(filename):
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
### NOVELTY, RESONANCE, HURST EXPONENT                                         ###
##################################################################################

def extract_novelty_resonance(df, 
                             theta: list, 
                             dates: list, 
                             window: int):
    """
    df: pandas DataFrame
    theta: list of theta values (list of lists)
    dates: list of dates
    window: int of the window size

    Returns pandas DataFrame with novelty-resonance values
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

def hurst_exp(resonance: list, 
              OUT_PATH: str):
    """
    resonance:  list of resonance values
    OUT_PATH: path for where the output is saved to

    Returns hurst exponent hurst_r
    """
    nolds.hurst_rs(resonance, nvals=None, fit='poly', debug_plot=True, plot_file=None, corrected=True, unbiased=True)
    fignameH = os.path.join(OUT_PATH, "fig", "H_plot.png")
    hurst_r = nolds.hurst_rs(resonance, nvals=None, fit='poly', debug_plot=True, plot_file=fignameH, corrected=True, unbiased=True)
    return hurst_r

### 
### VISUALIZE
###

def set_base_plot_settings(fontsize, if_palette):
    matplotlib.rc('ytick', labelsize=fontsize)
    matplotlib.rc('xtick', labelsize=fontsize)
    themes.theme_minimal(grid=False, ticks=False, fontsize=fontsize)
    a4_dims = (25,15)
    
    if if_palette:
        #          0 black      1 orange  2 L blue   3 green    4 L orange  5 D blue  6 D orange 7 purple
        palette = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
    else:
        palette = 0
    
    fig, (ax1) = plt.subplots(1,1, figsize=a4_dims)
    sns.set(font_scale = 2)

    return fig, ax1, palette

def set_late_plot_settings(fig, ax1, if_dates):
    ax1.set(xlabel="", ylabel = "")
    ax1.xaxis.get_label().set_fontsize(40)
    ax1.yaxis.get_label().set_fontsize(40)

    ax1.grid(color='darkgrey', linestyle='-', linewidth=0.5, which= "both")
    if if_dates:
        # Define the date format
        ax1.xaxis_date(tz="UTC")
        date_form = mdates.DateFormatter("%d-%b")
        ax1.xaxis.set_major_formatter(date_form)

    #ax1.set(ylim=(0, None))
    return fig, ax1

def set_late_barplot_settings(fig, ax1):
    ax1.set(xlabel="", ylabel = "")
    ax1.xaxis.get_label().set_fontsize(40)
    ax1.yaxis.get_label().set_fontsize(40)
    return fig, ax1

# columns; 'id', 'group_id', 'user_id', 'user_name', 'message', 'name',
#                       'description', 'link', 'application', 'comment_count', 'likes_count',
#                       'shares_count', 'picture', 'story', 'created', 'updated', 'word_count'

def novelty_transcience_resonance_lineplot(df, root_path, datatype):
    df = df[df["date"] >= '2014-01-01']
    ic("Base plot settings")
    fig, ax1, palette = set_base_plot_settings(fontsize=30, if_palette = True)
    ic("Line plot")
    ax1 = sns.lineplot(x="date", y="novelty",
                      palette = palette[1], 
                      label = "Novelty",
                        linewidth = 3, data = df)

    ax1 = sns.lineplot(x="date", y="transience",
                      palette = palette[2], 
                      label = "Transience",
                        linewidth = 3, data = df)

    ax1 = sns.lineplot(x="date", y="resonance",
                      palette = palette[3], 
                      label = "Resonance",
                        linewidth = 3, data = df)
    
    ic("Late plot settings")
    fig, ax1 = set_late_plot_settings(fig, ax1, if_dates = True)

    ic("Add legend")
    leg = plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', facecolor='white')
    # set the linewidth of each legend object
    for legobj in leg.legendHandles:
        legobj.set_linewidth(10.0)
    
    ax1.xaxis_date(tz="UTC")
    date_form = mdates.DateFormatter("%b-%Y")
    ax1.xaxis.set_major_formatter(date_form)

    ic("Save image")
    plot_name = f"{root_path}out/fig/{datatype}_novelty-resonance-transcience.png"
    fig.savefig(plot_name, bbox_extra_artists=(leg,), bbox_inches='tight')
    
    ic("Save figure done\n------------------\n")


def main(datatype, DATA_PATH, OUT_PATH, LANG):
    data_path = "/data/datalab/danish-facebook-groups/raw-export/*"

    logging.info("----- New iteration -----")

    ic("[INFO] Reading in the data")
    logging.info("[INFO] Reading in the data")
    df = pd.read_table(f"tmp/tmp_df.csv", 
                        encoding='utf-8',
                        sep=";",
                        nrows=5000)
    files = glob.glob(data_path)
    filename = [i for i in files if datatype in i]
    filename = filename[0]
    df = read_json_data(filename)[:5000]
    ic(df.head())
    ic(df.columns)

    if "text" not in df.columns:
        old_column_name = f"{datatype[:-1]}_text"
        try:
            df["text"] = df["message"]
        except:
            df["text"] = df[old_column_name]

    df['word_count'] = df.text.str.split().str.len()
    
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
    
    ic("[INFO] extracting novelty and resonance...")
    dates = pd.to_datetime(df["created"])
    WINDOW = 3
    df = extract_novelty_resonance(df, out["theta"], out["dates"], WINDOW)
    ic(df.head())    
    df["date"] = pd.to_datetime(df["created"])
    novelty_transcience_resonance_lineplot(df, OUT_PATH, datatype)

    ic("[INFO] PIPELINE FINISHED")
    logging.info("----- Finished iteration -----")


if __name__ == '__main__':
    logging.basicConfig(filename='logs/main.log',
                        #encoding='utf-8',
                        level=logging.DEBUG)
    datatype = "posts"
    activated = spacy.prefer_gpu()
    np.random.seed(1984)
    DATA_PATH = "/data/datalab/danish-facebook-groups"
    OUT_PATH = "/home/commando/marislab/facebook-posts/"
    LANG = 'da'

    main(datatype, DATA_PATH, OUT_PATH, LANG)
    