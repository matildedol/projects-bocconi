import spacy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import time

nlp = spacy.load("en_core_web_sm")   #https://spacy.io/usage/models, multi-language, chosen for accuracy

start_time=time.time()
print('Starting time:', start_time)
train_df=pd.read_csv('train_responses.csv')
test_df=pd.read_csv('dev_responses.csv')

# we build a vocabulary with dependency-based word bigrams 'lemma_head' from a list of documents 'corpus'
def build_vocabulary(corpus):                                                                                                    
    vocabulary = set(['{}_{}'.format(token.lemma_.lower(), token.head.lemma_.lower()) for doc in corpus for token in nlp(doc)])             # in our case, docs are prompts
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

def get_bestmatch(queries, keys, metric):                                 # è per forza O(n^2) !! come farlo piu efficiente?
    best_matches=[]
    for query in queries:
        for key in keys:
            best_cos_sim=0
            cos_sim=metric(query, key)               
            if cos_sim>best_cos_sim:
                best_cos_sim=cos_sim
                best_prompt=key                                 
        best_matches.append(best_prompt)
    return best_matches                                         # list of best matches orderd as queries list

# def compute_cosinesim(query,key):
#     cos_sim = np.dot(query,key) / (np.linalg.norm(query)*np.linalg.norm(key))
#     return cos_sim

def get_results(queries_df, keys_df):
    keys_prompts = keys_df['user_prompt'].to_list()
    queries_prompts = queries_df['user_prompt'].to_list()
    my_vocabulary=build_vocabulary(keys_prompts)
    train_reps=get_discrete_rep(keys_prompts, my_vocabulary, 'n')
    test_reps=get_discrete_rep(queries_prompts, my_vocabulary, 'n')
    matches=get_bestmatch(test_reps, train_reps, cosine_similarity)
    queries_df['best_matches']=matches
    end_time=time.time() -start_time
    return queries_df, end_time


results, run_time =get_results(test_df, train_df).head()
print('Running time:', run_time, '\nResults:', results.head())

# compute BLEU score to assess performance
def compute_BLEU():
    pass


