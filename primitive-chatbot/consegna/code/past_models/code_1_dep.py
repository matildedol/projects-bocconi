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
Track 1 with bigrams head_lemma discrete representation - seemed like a good idea but is not

'''

nlp = spacy.load("en_core_web_sm")   #https://spacy.io/usage/models, multi-language, chosen for accuracy

start_time=time.time()
now=datetime.datetime.now()
print('starting time:', now)
train_df=pd.read_csv('train_responses.csv')
test_df=pd.read_csv('dev_responses.csv')


# we build a vocabulary with dependency-based word bigrams 'lemma_head' from a list of documents 'corpus'
def build_vocabulary(corpus, cache_file='vocabulary_cache.pkl'):
    # Check if cache file exists
    if os.path.exists(cache_file):
        print(f"loading vocabulary from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            vocabulary = pickle.load(f)
        return vocabulary
    
    # Build vocabulary if no cache exists
    print("building vocabulary from scratch...")
    vocabulary = set(['{}_{}'.format(token.lemma_.lower(), token.head.lemma_.lower()) 
                    for doc in corpus for token in nlp(doc)])
    
    # Save vocabulary to cache
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
    #instaed of keys[i] i want to put the value in the keys_df under responses at index i
    
    return [keys_df.iloc[i]['model_response'] for i in best_match_indices]                                  # list of best matches orderd as queries list

# do a function that takes the rep, goes to keys_df (aggiornato),
# takes the corresponding response, stores the response in a pivot_dict as the value to the QUERIES! rep
# then create a list with all responses in order and add to df
# (i think there's a cleaner way of doing this but whatever)

# def retrieve_responses(best_matches_reps, keys_df): 
#     print('retrieving corresponding reponses...')
#     computed_responses=[]
#     keys_reps_list=keys_df['vector_reps'].to_list()
#     for q in best_matches_reps:
#         for k in keys_reps_list:
#             if k.all()==q.all():
#                 # let's try to do it again relying on order implicitly remaining the same  
#                 computed_responses.append(keys_df.iloc[k]['model_response'])
#     if len(computed_responses)!=len(best_matches_reps):
#         sys.exit('watch out! the list of computed responses has a wrong length :/')
#     return computed_responses

# def compute_cosinesim(query,key):
#     cos_sim = np.dot(query,key) / (np.linalg.norm(query)*np.linalg.norm(key))
#     return cos_sim

def get_results(queries_df, keys_df, use_cache=True):
    print("let's start!\n")
    keys_prompts = keys_df['user_prompt'].to_list()
    queries_prompts = queries_df['user_prompt'].to_list()
    
    print('building vocabulary...\n')
    cache_file = 'vocabulary_cache.pkl'
    if not use_cache and os.path.exists(cache_file):
        os.remove(cache_file)  # Delete cache if not using it
    
    vocab_start_time = time.time()
    my_vocabulary = build_vocabulary(keys_prompts, cache_file)
    vocab_end_time = time.time()
    voc_time = vocab_end_time - vocab_start_time
    print(f'vocabulary built! time: {voc_time:.2f} s\n')
    print('computing representations...\n')
    train_reps=get_discrete_rep(keys_prompts, my_vocabulary, 'n')
    keys_df['vector_reps']=train_reps #add representations to df to pivot later
    train_time=time.time() - voc_time
    print(f'\ntraining set representations: done! time: {train_time:.2f} s\n')
    test_reps=get_discrete_rep(queries_prompts, my_vocabulary, 'n')
    test_time=time.time() - train_time
    print(f'test set representations: done! time: {test_time:.2f} s\n')
    matches=get_bestmatch(test_reps, train_reps, keys_df, cosine_similarity)
    match_time=time.time() - test_time
    print(f'matches retrieved! time: {match_time:.2f} s\n')
    queries_df['retrieved_response']=matches
    #responses=retrieve_responses(matches, keys_df)
    #queries_df['computed_responses']=responses
    end_time=time.time() - start_time
    return queries_df, end_time

#compute BLEU score to assess performance
def compute_BLEU(data):
    smoothingfunction = SmoothingFunction()
    
    # Check for non-string values and convert them to strings or handle them appropriately
    if data[data['model_response'].isna() | data['retrieved_response'].isna()].shape[0] > 0:
        print('NaNs values found!')
    
    def safe_bleu(row):
        try:
            # Convert to string first in case values are floats
            model_tokens = str(row['model_response']).split()
            retrieved_tokens = str(row['retrieved_response']).split()
            return sentence_bleu([model_tokens], retrieved_tokens, 
                               weights=(0.5, 0.5, 0, 0), 
                               smoothing_function=smoothingfunction.method3)
        except Exception as e:
            print(f"Error calculating BLEU for row: {row.name}, Error: {e}")
            return 0.0  # Return a default value or handle as appropriate
    
    data['bleu_score'] = data.apply(safe_bleu, axis=1)
    end_time = time.time() - start_time
    return data, end_time


results,run_time=get_results(test_df, train_df)
evaluation, run_time = compute_BLEU(results)
bleu_list=evaluation['bleu_score'].to_list()
avg_BLEU=np.sum(i for i in bleu_list)/len(bleu_list)
print(f'Running time: {run_time:.2f}\nResults:', evaluation.head())
print('\nAverage BLEU score:', avg_BLEU )