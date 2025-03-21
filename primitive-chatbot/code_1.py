import spacy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import time
import datetime
from scipy.sparse import vstack
import pickle
import os
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
'''
should i use a class?? bo! 

i should add history! (?) think about it :) 
'''

nlp = spacy.load("en_core_web_sm")   #https://spacy.io/usage/models, multi-language, chosen for accuracy

start_time=time.time()
now=datetime.datetime.now()
print('starting time:', now)
train_df=pd.read_csv('train_responses.csv')
test_df=pd.read_csv('dev_responses.csv')


# we build a vocabulary with dependency-based word bigrams 'lemma_head' from a list of documents 'corpus'
def build_vocabulary(corpus, cache_file='vocabulary_cache.pkl'):
    # load vocabulary from cache
    if os.path.exists(cache_file):
        print(f"loading vocabulary from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            vocabulary = pickle.load(f)
        return vocabulary
    
    # build vocabulary at first run
    print("building vocabulary from scratch...")
    vocabulary = set(['{}_{}'.format(token.lemma_.lower(), token.head.lemma_.lower()) 
                    for doc in corpus for token in nlp(doc)])
    
    print(f"saving vocabulary to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(vocabulary, f)
    
    return vocabulary

def get_discrete_rep(corpus, vocab, visualize):
    parsed_docs = [' '.join('{}_{}'.format(token.lemma_, token.head.lemma_) for token in nlp(doc)) for doc in corpus]       # format sentences as bigrams lemma_head
    # print(parsed_docs[:20])
    vectorizer=CountVectorizer( vocabulary=vocab)                                                                           # use vectorizer to count occurrencies of 'lemma_head' words in our sentences 
    X = vectorizer.fit_transform(parsed_docs)                                                                                  # create matrix: each row is a prompt, each column a couple lemma_head, entries are frequencies
    reps_list=[row for row in X]                                                                                            # this is a 1xnum(docs) sparse matrix
    if reps_list[0].shape[1]!=len(vocab):                                                                                   # dim of these vectors is len(vocabulary), their number is len(parsed_docs) which differs from train to test
        sys.exit('error: dim of representation vectors is not equal to the length of vocabulary')
    if len(reps_list)!=len(parsed_docs):
        sys.exit('error: number of representation vectors is not equal to the num of docs')

    if visualize=='y':
        X_array=X.toarray()                                                                                                     # auxiliary function to visualize df
        print(X[:4])
        feature_names = vectorizer.get_feature_names_out()
        df_features = pd.DataFrame(X_array, columns=feature_names, index=parsed_docs[:len(X_array)])
        print(df_features)

    return reps_list

def get_bestmatch(queries, keys, keys_df, metric=cosine_similarity):    
    queries_matrix = vstack(queries)
    keys_matrix = vstack(keys)
    print('building similarity matrix...')
    similarity_matrix = metric(queries_matrix, keys_matrix)
    print('retrieving best matches...')
    best_match_indices = similarity_matrix.argmax(axis=1)    
    return [keys_df.iloc[i]['model_response'] for i in best_match_indices]                                 

def get_results(queries_df, keys_df, get_rep, use_cache=True):
    print("let's start!\n")
    keys_prompts = keys_df['user_prompt'].to_list()
    queries_prompts = queries_df['user_prompt'].to_list()
    
    print('building vocabulary...\n')
    # delete cache if i want to rebuild
    cache_file = 'vocabulary_cache.pkl'
    if not use_cache and os.path.exists(cache_file):
        os.remove(cache_file) 
    
    vocab_start_time = time.time()
    my_vocabulary = build_vocabulary(keys_prompts, cache_file)
    vocab_end_time = time.time()
    voc_time = vocab_end_time - vocab_start_time
    print(f'vocabulary built! time: {voc_time:.2f} s\n')
    print('computing representations...\n')
    train_reps=get_rep(keys_prompts, my_vocabulary, 'n')
    keys_df['vector_reps']=train_reps #add representations to df to pivot later
    train_time=time.time() - voc_time
    print(f'\ntraining set representations: done! time: {train_time:.2f} s\n')
    test_reps=get_rep(queries_prompts, my_vocabulary, 'n')
    test_time=time.time() - train_time
    print(f'test set representations: done! time: {test_time:.2f} s\n')
    matches=get_bestmatch(test_reps, train_reps, keys_df, cosine_similarity)
    match_time=time.time() - test_time
    print(f'matches retrieved! time: {match_time:.2f} s\n')
    queries_df['retrieved_response']=matches
    end_time=time.time() - start_time
    return queries_df, end_time

#compute BLEU score to assess performance
def compute_BLEU(data):
    smoothingfunction = SmoothingFunction()
    
    # check NaNs
    if data[data['model_response'].isna() | data['retrieved_response'].isna()].shape[0] > 0:
        print('NaNs values found!')
    
    def safe_bleu(row):
        try:
            # convert to string
            model_tokens = str(row['model_response']).split()
            retrieved_tokens = str(row['retrieved_response']).split()
            return sentence_bleu([model_tokens], retrieved_tokens, 
                               weights=(0.5, 0.5, 0, 0), 
                               smoothing_function=smoothingfunction.method3)
        except Exception as e:
            print(f"error in calculating BLEU for row: {row.name}, error: {e}")
            return 0
    
    data['bleu_score'] = data.apply(safe_bleu, axis=1)
    end_time = time.time() - start_time
    return data, end_time


discrete_results,discrete_run_time=get_results(test_df, train_df, get_discrete_rep)
evaluation, discrete_run_time = compute_BLEU(discrete_results)
bleu_list=evaluation['bleu_score'].to_list()
avg_BLEU=np.sum(i for i in bleu_list)/len(bleu_list)
print(f'DISCRETE CASE, TASK 1:\nRunning time: {discrete_run_time:.2f}\nResults:', evaluation.head())
print('\nAverage BLEU score (task 1):', avg_BLEU )

