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
import fasttext
from gensim.models import KeyedVectors


'''
how to use: change model path in line 103 inside main() with path to fast text pretrained model :) 
'''

'''
word2vec BLEU score: 0.07729256362715167
'''

nlp = spacy.load("en_core_web_sm")   #https://spacy.io/usage/models, multi-language, chosen for accuracy

start_time=time.time()
now=datetime.datetime.now()
print('starting time:', now)
train_df=pd.read_csv('train_responses.csv')
test_df=pd.read_csv('dev_responses.csv')

class ContinuousRep:
    def __init__(self, keys_df, queries_df, model_en_path, model_es_path, model_po_path):
        self.keys_df=keys_df
        self.queries_df=queries_df
        self.model_en_path=model_en_path
        self.model_es_path=model_es_path
        self.model_po_path=model_po_path
        # self.model_it_path=model_it_path
        # self.model_du_path=model_du_path

        # self.model=fasttext.load_model(self.model_path)
        print('loading models...')
        self.model_en = KeyedVectors.load_word2vec_format(self.model_en_path, binary=False)
        self.model_en.fill_norms()
        self.model_es = KeyedVectors.load_word2vec_format(self.model_es_path, binary=False)
        self.model_es.fill_norms()
        self.model_po = KeyedVectors.load_word2vec_format(self.model_po_path, binary=False)
        self.model_po.fill_norms()
        # self.model_it = KeyedVectors.load_word2vec_format(self.model_it_path, binary=False)
        # self.model_it.fill_norms()
        # self.model_du  = KeyedVectors.load_word2vec_format(self.model_du_path, binary=False)
        # self.model_du.fill_norms()


    def get_word_embedding(self, word):
        try:
            return self.model_en[word.lower()]
        except KeyError:
            try:
                return self.model_es[word.lower()]
            except KeyError:
                try:
                    return self.model_po[word.lower()]
                except KeyError:

                    return np.zeros(self.model.vector_size) 
                   
    def get_continuous_rep(self, corpus):                #returns a list of representations as for discrete reps
        reps=[]
        for sentence in corpus:
            words=sentence.lower().split()
            word_embeddings = [self.get_word_embedding(word) for word in words]
            sentence_rep=np.mean(word_embeddings, axis=0)
            reps.append(sentence_rep)
        return reps

    def get_bestmatch(self, queries, keys,  metric=cosine_similarity):    
        queries_matrix = vstack(queries)
        keys_matrix = vstack(keys)
        print('building similarity matrix...')
        similarity_matrix = metric(queries_matrix, keys_matrix)
        print('retrieving best matches...')
        best_match_indices = similarity_matrix.argmax(axis=1)    
        return [self.keys_df.iloc[i]['model_response'] for i in best_match_indices]                                 

    def get_results(self,  get_rep):
        print("let's start!\n")
        keys_prompts = self.keys_df['user_prompt'].to_list()
        queries_prompts = self.queries_df['user_prompt'].to_list()
        print('computing representations...\n')
        train_reps=get_rep(keys_prompts)
        self.keys_df['vector_reps']=train_reps #add representations to df to pivot later
        train_time=time.time() - start_time
        print(f'\ntraining set representations: done! time: {train_time:.2f} s\n')
        test_reps=get_rep(queries_prompts)
        test_time=time.time() - train_time
        print(f'test set representations: done! time: {test_time:.2f} s\n')
        matches=self.get_bestmatch(test_reps, train_reps,  cosine_similarity)
        match_time=time.time() - test_time
        print(f'matches retrieved! time: {match_time:.2f} s\n')
        self.queries_df['retrieved_response']=matches
        end_time=time.time() - start_time
        return self.queries_df, end_time

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
    
def main():
    get_responses=ContinuousRep( train_df,test_df,'fasttext_models/cc.en.300.vec.gz', 'fasttext_models/cc.es.300.vec.gz','fasttext_models/cc.pt.300.vec.gz')
    cont_results,cont_run_time=get_responses.get_results(get_responses.get_continuous_rep)
    evaluation, cont_run_time = get_responses.compute_BLEU(cont_results)
    bleu_list=evaluation['bleu_score'].to_list()
    avg_BLEU=np.sum(i for i in bleu_list)/len(bleu_list)
    print(f'CONTINUOUS CASE, TASK 2:\nRunning time: {cont_run_time:.2f}\nResults:', evaluation.head())
    print('\nAverage BLEU score (task 2):', avg_BLEU )

if __name__ == "__main__":
    main()

