import spacy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import time
import datetime
from scipy.sparse import vstack, csr_matrix
import pickle
import os
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import fasttext
from gensim.models import KeyedVectors
import functools
import hashlib

nlp = spacy.load("en_core_web_sm")   #https://spacy.io/usage/models, multi-language, chosen for accuracy

start_time=time.time()
now=datetime.datetime.now()
print('starting time:', now)

train_df=pd.read_csv('data/train_responses.csv')
test_df=pd.read_csv('data/dev_responses.csv')
model='crawl-300d-2M.vec'

class ContinuousRep:
    def __init__(self, keys_df, queries_df, model_path, cache_dir='model_cache'):
        self.keys_df = keys_df
        self.queries_df = queries_df
        self.model_path = model_path

        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        print('loading models...')
        # Cached model loading
        self.model = self._load_cached_model(self.model_path, 'global')


        # Memoization cache for word embeddings
        self.embedding_cache = {}

    def _load_cached_model(self, model_path, model_name):
        """
        Load word vectors with caching to improve startup performance
        """
        cache_path = os.path.join(self.cache_dir, f'{model_name}_model_cache.pkl')
        
        # Check if cached model exists
        if os.path.exists(cache_path):
            print(f'Loading cached {model_name} model...')
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        # Load model and cache it
        model = KeyedVectors.load_word2vec_format(model_path, binary=False)
        model.fill_norms()
        
        # Save to cache
        print(f'Caching {model_name} model...')
        with open(cache_path, 'wb') as f:
            pickle.dump(model, f)
        
        return model

    @functools.lru_cache(maxsize=10000)

    def get_word_embedding(self, word):
        try:
            return model[word.lower()]
        except KeyError:
            return np.zeros(self.model.vector_size)

    def get_continuous_rep(self, corpus, cache_path='cached_model.pkl'):
        """
        Cached continuous representation generation with new representation method
        """
        if os.path.exists(cache_path):
            print('Loading cached representations...')
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        reps = []
        for sentence in corpus:
            words = sentence.split()
            word_embeddings = [self.get_word_embedding(word) for word in words]
            sentence_rep = np.mean(word_embeddings, axis=0)
            reps.append(sentence_rep)
        
        # Cache the new representations
        with open(cache_path, 'wb') as f:
            pickle.dump(reps, f)
            
        return reps

    def get_bestmatch(self, queries, keys, metric=cosine_similarity):
        
        # Option 1: Top-K matches instead of single best match
        top_k = 3
        similarity_matrix = metric(queries, keys)
        
        # Get top-K indices for each query
        top_k_indices = np.argsort(similarity_matrix, axis=1)[:, -top_k:]
        
        # Aggregate or vote-based matching
        best_matches = []
        for indices in top_k_indices:
            candidate_responses = [self.keys_df.iloc[i]['model_response'] for i in indices]
            # Use voting or more sophisticated aggregation
            best_match = max(set(candidate_responses), key=candidate_responses.count)
            best_matches.append(best_match)
        
        return best_matches

    def get_results(self, get_rep):
        print("let's start!\n")
        keys_prompts = self.keys_df['user_prompt'].to_list()
        queries_prompts = self.queries_df['user_prompt'].to_list()

        print('Computing representations...\n')
        # Cached representations
        train_reps = get_rep(keys_prompts)
        self.keys_df['vector_reps'] = train_reps
        train_time = time.time() - start_time
        print(f'\nTraining set representations: done! Time: {train_time:.2f} s\n')
        test_reps = get_rep(queries_prompts)
        test_time = time.time() - train_time
        print(f'Test set representations: done! Time: {test_time:.2f} s\n')
        
        matches = self.get_bestmatch(test_reps, train_reps, cosine_similarity)
        match_time = time.time() - test_time
        print(f'Matches retrieved! Time: {match_time:.2f} s\n')
        
        self.queries_df['retrieved_response'] = matches
        end_time = time.time() - start_time
        return self.queries_df, end_time

    @staticmethod
    def compute_BLEU(data):
        smoothingfunction = SmoothingFunction()
        
        # Check NaNs
        if data[data['model_response'].isna() | data['retrieved_response'].isna()].shape[0] > 0:
            print('NaNs values found!')
        
        def safe_bleu(row):
            try:
                # Convert to string
                model_tokens = str(row['model_response']).split()
                retrieved_tokens = str(row['retrieved_response']).split()
                return sentence_bleu([model_tokens], retrieved_tokens, 
                                     weights=(0.5, 0.5, 0, 0), 
                                     smoothing_function=smoothingfunction.method3)
            except Exception as e:
                print(f"Error in calculating BLEU for row: {row.name}, error: {e}")
                return 0
        
        data['bleu_score'] = data.apply(safe_bleu, axis=1)
        end_time = time.time() - start_time
        return data, end_time

def main():
    get_responses = ContinuousRep(train_df, test_df,model)
    cont_results, cont_run_time = get_responses.get_results(get_responses.get_continuous_rep)
    evaluation, cont_run_time = get_responses.compute_BLEU(cont_results)
    bleu_list = evaluation['bleu_score'].to_list()
    avg_BLEU = np.sum(i for i in bleu_list) / len(bleu_list)
    print(f'CONTINUOUS CASE, TASK 2:\nRunning time: {cont_run_time:.2f}\nResults:', evaluation.head())
    print('\nAverage BLEU score (task 2):', avg_BLEU)

if __name__ == "__main__":
    main()