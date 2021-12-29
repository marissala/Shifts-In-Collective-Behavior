"""Preprocess and lemmatize texts - optimized for speed
Author: Maris Sala
Date: 3rd Nov 2021
"""
import re
import sys
import spacy
import time

import numpy as np
import pandas as pd
from icecream import ic

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

sys.path.insert(1, r'/home/commando/marislab/newsFluxus/src/')
from tekisuto.preprocessing import Tokenizer
from preparations.preptopicmodeling import prepareTopicModeling

from nltk.corpus import stopwords
stop_words = stopwords.words('danish')
en_stop_words = stopwords.words('english')
stop_words.extend(en_stop_words)

preTM = prepareTopicModeling
activated = spacy.prefer_gpu()
np.random.seed(1984)
LANG = 'da'
if LANG == 'da':
    nlp = spacy.load("da_core_news_lg", disable=['tagger', 'parser', 'ner'])
    

def remove_invalid_entries(tokens: list,
                           df):
    """Removes lines from dataset where tokens are empty/invalid

    tokens: list of tokens
    df: pandas DataFrame
    """
    invalid_entries = [index for index in range(len(tokens)) if (tokens[index] == [] or len(tokens[index]) < 3)]
    ic(len(invalid_entries))
    ic(f'Invalid entries removed at {invalid_entries}')#': {df.loc[invalid_entries,0]}')
    df["tokens"] = tokens
    df = df[df["tokens"].apply(lambda x: len(x)) >= 3].reset_index(drop=True)
    tokens = df["tokens"]
    return tokens, df

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

def make_lemmas(texts_words, nlp):
    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(texts_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[texts_words], threshold=100)  

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    # Remove Stop Words
    data_words_nostops = remove_stopwords(texts_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    #nlp = spacy.load('da', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    return data_lemmatized

def prepare_data(df, LANG):
    ic("[INFO] Lemmatizing...")
    if LANG == "da":
        nlp = spacy.load("da_core_news_lg", disable=['parser', 'ner'])
    
    lemmas = make_lemmas(df, nlp, LANG)
    ROOT_PATH = r"/home/commando/marislab/gsdmm/"
    lemmas = preTM.preprocess_for_topic_models(lemmas, ROOT_PATH, lang=LANG)
    ic("[INFO] Tokenizing...")
    
    to = Tokenizer()
    tokens = to.doctokenizer(lemmas)
    tokens, df = remove_invalid_entries(tokens, df)
    return tokens, df

def cleaner(df, limit=-2):
    "Extract relevant text from DataFrame using a regex"
    # Regex pattern for only alphanumeric, hyphenated text with 3 or more chars
    #pattern = re.compile(r"[A-Za-z0-9\-]{3,50}")
    pattern = re.compile(r"\b\w{3,50}")
    url_pattern = re.compile(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})')
    df['cleantext'] = df['text'].str.replace(url_pattern, '')#.str.join(' ')
    df['cleantext'] = df['cleantext'].str.findall(pattern).str.join(' ')
    
    if limit > 0:
        return df.iloc[:limit, :].copy()
    else:
        return df

from joblib import Parallel, delayed

def lemmatize_pipe(doc):
    lemma_list = [str(tok.lemma_).lower() for tok in doc
                  if tok.is_alpha and tok.text.lower() not in stop_words]
    return lemma_list

def chunker(iterable, total_length, chunksize):
    return (iterable[pos: pos + chunksize] for pos in range(0, total_length, chunksize))

def flatten(list_of_lists):
    "Flatten a list of lists to a combined list"
    return [item for sublist in list_of_lists for item in sublist]

def process_chunk(texts):
    preproc_pipe = []
    for doc in nlp.pipe(texts, batch_size=20):
        preproc_pipe.append(lemmatize_pipe(doc))
    return preproc_pipe

def preprocess_parallel(df_preproc, texts, chunksize=100):
    executor = Parallel(n_jobs=7, backend='multiprocessing', prefer="processes")
    do = delayed(process_chunk)
    tasks = (do(chunk) for chunk in chunker(texts, len(df_preproc), chunksize=chunksize))
    result = executor(tasks)
    return flatten(result)

def parallelized_lemmatizer(df):
    df_preproc = cleaner(df)
    lemmas = preprocess_parallel(df_preproc, df_preproc["cleantext"], chunksize=1000)
    return lemmas

def spacy_lemmatize(texts: list, 
                        nlp, 
                        **kwargs):
    """
    texts: input texts as list
    nlp: specifies spacy language model
    **kwargs: other arguments to spacy NLP pipe

    Returns lemmas for all documents in a list
    """
    nlp.add_pipe('sentencizer')
    docs = nlp.pipe(texts, **kwargs)
        
    def __lemmatize(doc):
        lemmas = []
        for sent in doc.sents:
            for token in sent:
                lemmas.append(token.lemma_)
        return lemmas

    return [__lemmatize(doc) for doc in docs]

if __name__ == '__main__':
    # Testing pipeline
    datatype = "posts"
    
    activated = spacy.prefer_gpu()
    np.random.seed(1984)
    LANG = 'da'
    if LANG == 'da':
        nlp = spacy.load("da_core_news_lg", disable=['tagger', 'parser', 'ner'])
    
    ic('Read in data')
    df = pd.read_table(f"/data/datalab/danish-facebook-groups/dk_{datatype}.csv", encoding='utf-8', nrows=1000)
    if "text" not in df.columns:
        old_column_name = f"{datatype[:-1]}_text"
        df["text"] = df[old_column_name]
    
    ic('Preprocess texts')
    texts = df.text.values.tolist()
    tic = time.perf_counter()
    texts_words = list(sent_to_words(texts))
    toc = time.perf_counter()
    time_info = f"Preprocessing: {toc - tic:0.4f} seconds"
    ic(time_info)
    print(texts_words[0:3])

    ic('Make lemmas:')
    ic('---Basic---')
    tic = time.perf_counter()
    lemmas = make_lemmas(texts_words, nlp)
    toc = time.perf_counter()
    time_info = f"Lemmatizing: {toc - tic:0.4f} seconds"
    ic(time_info)
    print(lemmas[:3])

    ic('---Spacy pipes---')
    tmp = map(' '.join, texts_words)
    tic = time.perf_counter()
    lemmas = spacy_lemmatize(tmp, nlp)
    toc = time.perf_counter()
    time_info = f"Lemmatizing: {toc - tic:0.4f} seconds"
    ic(time_info)
    print(lemmas[:3])

    ic('---3rd paralell one---')
    tic = time.perf_counter()
    lemmas = parallelized_lemmatizer(df)
    toc = time.perf_counter()
    time_info = f"Lemmatizing: {toc - tic:0.4f} seconds"
    ic(time_info)
    print(lemmas[:3])
    
    ic('Tokenize')
    to = Tokenizer()
    tic = time.perf_counter()
    tokens = to.doctokenizer(lemmas)
    toc = time.perf_counter()
    time_info = f"Tokenizing: {toc - tic:0.4f} seconds"
    ic(time_info)
    print(tokens[:3])

    ic('---PIPELINE FINISHED---')
