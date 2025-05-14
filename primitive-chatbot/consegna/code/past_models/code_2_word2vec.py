import spacy
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import datetime
from scipy.sparse import vstack
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import gensim.downloader as api

'''
Track 2 with word2vec continuous representation, computing the mean of word representations to get sentence rep
word2vec BLEU score: 0.07729256362715167
runtime: lungo
'''

nlp = spacy.load("en_core_web_sm")   #https://spacy.io/usage/models, multi-language, chosen for accuracy

start_time=time.time()
now=datetime.datetime.now()
print('starting time:', now)
train_df=pd.read_csv('../data/train_responses.csv')
test_df=pd.read_csv('../data/dev_responses.csv')
print("loading word2vec model...")
word2vec_model = api.load('word2vec-google-news-300')
print("Word2vec model loaded!")

def get_continuous_rep(corpus, vocab=None, visualize='n'):
    document_vectors = []
    
    for doc in corpus:
        parsed_doc = nlp(doc)
        
        # Get embeddings for token-head pairs
        pair_embeddings = []
        for token in parsed_doc:
            token_lemma = token.lemma_.lower()
            head_lemma = token.head.lemma_.lower()
            
            # Try to get embeddings for both token and head
            token_vector = None
            head_vector = None
            
            if token_lemma in word2vec_model:
                token_vector = word2vec_model[token_lemma]
            
            if head_lemma in word2vec_model:
                head_vector = word2vec_model[head_lemma]
            
            # If both token and head have embeddings, concatenate them
            if token_vector is not None and head_vector is not None:
                # Concatenate the two vectors
                pair_vector = np.concatenate([token_vector, head_vector])
                pair_embeddings.append(pair_vector)
        
        # If document has no valid embeddings, use zeros
        if not pair_embeddings:
            document_vectors.append(np.zeros(600))  # 300 dims for token + 300 dims for head
        else:
            # Average all pair embeddings for the document
            document_vector = np.mean(pair_embeddings, axis=0)
            document_vectors.append(document_vector)
    
    if visualize == 'y':
        print(f"First document vector shape: {document_vectors[0].shape}")
        print(f"Sample of first document vector: {document_vectors[0][:10]}")
    
    return document_vectors

def get_bestmatch(queries, keys, keys_df, metric=cosine_similarity):
    queries_matrix = np.vstack(queries)
    keys_matrix = np.vstack(keys)
    
    print('building similarity matrix...')
    similarity_matrix = metric(queries_matrix, keys_matrix)
    print('retrieving best matches...')
    best_match_indices = similarity_matrix.argmax(axis=1)    
    return [keys_df.iloc[i]['model_response'] for i in best_match_indices]

def get_results(queries_df, keys_df, get_rep, use_cache=False):
    print("let's start!\n")
    keys_prompts = keys_df['user_prompt'].to_list()
    queries_prompts = queries_df['user_prompt'].to_list()
    
    print('computing representations...\n')
    
    rep_start_time = time.time()
    train_reps = get_rep(keys_prompts)
    keys_df['vector_reps'] = train_reps  # add representations to df to pivot later
    train_time = time.time() - rep_start_time
    print(f'\ntraining set representations: done! time: {train_time:.2f} s\n')
    
    test_reps = get_rep(queries_prompts)
    test_time = time.time() - train_time - rep_start_time
    print(f'test set representations: done! time: {test_time:.2f} s\n')
    
    matches = get_bestmatch(test_reps, train_reps, keys_df, cosine_similarity)
    match_time = time.time() - test_time - train_time - rep_start_time
    print(f'matches retrieved! time: {match_time:.2f} s\n')
    
    queries_df['retrieved_response'] = matches
    end_time = time.time() - start_time
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

continuous_results, continuous_run_time = get_results(test_df, train_df, get_continuous_rep)
evaluation, continuous_run_time = compute_BLEU(continuous_results)
bleu_list = evaluation['bleu_score'].to_list()
avg_BLEU = np.sum(i for i in bleu_list)/len(bleu_list)
print(f'CONTINUOUS CASE, TASK 2:\nRunning time: {continuous_run_time:.2f}\nResults:', evaluation.head())
print('\nAverage BLEU score (task 2):', avg_BLEU)