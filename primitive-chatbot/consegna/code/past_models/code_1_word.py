import spacy
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import datetime
from scipy.sparse import vstack
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from sklearn.feature_extraction.text import TfidfVectorizer

'''
runs in around 3 s
discrete representation: word ngrams
BLEU score: Average BLEU score (task 1): 0.07897303919524042
'''

nlp = spacy.load("en_core_web_sm")   #https://spacy.io/usage/models, multi-language, chosen for accuracy

start_time=time.time()
now=datetime.datetime.now()
print('starting time:', now)
train_df=pd.read_csv('data/train_responses.csv')
test_df=pd.read_csv('data/dev_responses.csv')


def get_discrete_rep(corpus, ref_corpus, visualize):
    vectorizer=TfidfVectorizer(analyzer='word', min_df=0.001, max_df=0.75,sublinear_tf=True)                                                                           
    vectorizer.fit(ref_corpus)                                                                                              # fit vectorizer to train data to get vocabulary
    X= vectorizer.transform(corpus)                                                                                         # create matrix: each row is a prompt, each column an ngram, entries are frequencies
    reps_list=[row for row in X]                                                                                            # this is a 1xnum(docs) sparse matrix

    if visualize=='y':
        X_array=X.toarray()                                                                                                     # auxiliary function to visualize df
        print(X[:4])
        feature_names = vectorizer.get_feature_names_out()
        df_features = pd.DataFrame(X_array, columns=feature_names, index=corpus[:len(X_array)])
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

def get_results(queries_df, keys_df, get_rep):
    print("let's start!\n")
    keys_prompts = keys_df['user_prompt'].to_list()
    queries_prompts = queries_df['user_prompt'].to_list()
    
    print('computing representations...\n')
    train_reps=get_rep(keys_prompts, keys_prompts, 'n')
    keys_df['vector_reps']=train_reps #add representations to df to pivot later
    train_time=time.time() - start_time
    print(f'\ntraining set representations: done! time: {train_time:.2f} s\n')
    test_reps=get_rep(queries_prompts, keys_prompts, 'n')
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
print(f'DISCRETE CASE, TASK 1:\nRunning time: {discrete_run_time:.2f} s\nResults:', evaluation.head())
print('\nAverage BLEU score (task 1):', avg_BLEU )

