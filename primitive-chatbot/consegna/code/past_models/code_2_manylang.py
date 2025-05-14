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

'''
runs in about 16 mins
FastText changing language models
Average BLEU score (task 2): 0.07025617907951288
'''

nlp = spacy.load("en_core_web_sm")   #https://spacy.io/usage/models, multi-language, chosen for accuracy

start_time=time.time()
now=datetime.datetime.now()
print('starting time:', now)
train_df=pd.read_csv('../data/train_responses.csv')
test_df=pd.read_csv('../data/dev_responses.csv')

class ContinuousRep:
    def __init__(self, train_df, test_df, model_en_path, model_es_path, model_po_path, cache_dir='model_cache'):
        self.train_df = train_df
        self.test_df = test_df
        self.model_en_path = model_en_path
        self.model_es_path = model_es_path
        self.model_po_path = model_po_path
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        print('loading models...')
        # Cached model loading
        self.model_en = self._load_cached_model(self.model_en_path, 'en')
        self.model_es = self._load_cached_model(self.model_es_path, 'es')
        self.model_po = self._load_cached_model(self.model_po_path, 'po')

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
        """
        Cached word embedding retrieval with LRU caching
        """
        # Check if already in cache
        if word in self.embedding_cache:
            return self.embedding_cache[word]
        
        # Try different models
        models = [self.model_en, self.model_es, self.model_po]
        for model in models:
            try:
                embedding = model[word.lower()]
                self.embedding_cache[word] = embedding
                return embedding
            except KeyError:
                continue
        
        # Return zero vector if word not found
        return np.zeros(models[0].vector_size)

    def get_continuous_rep(self, corpus):
        """
        Cached continuous representation generation
        Uses memoization to avoid recomputing representations
        """
        # Create a hash of the corpus to use as a cache key
        corpus_hash = hashlib.md5(''.join(corpus).encode()).hexdigest()
        cache_path = os.path.join(self.cache_dir, f'rep_cache_{corpus_hash}.pkl')
        
        # Check if cached representation exists
        if os.path.exists(cache_path):
            print('Loading cached representations...')
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        # Generate representations
        reps = []
        for sentence in corpus:
            words = sentence.lower().split()
            word_embeddings = [self.get_word_embedding(word) for word in words]
            sentence_rep = np.mean(word_embeddings, axis=0)
            reps.append(sentence_rep)
        
        # Cache the representations
        with open(cache_path, 'wb') as f:
            pickle.dump(reps, f)
        
        return reps

    def get_bestmatch(self, queries, keys, metric=cosine_similarity):
        """
        Cached best match retrieval with optional caching of similarity matrix
        """
        cache_path = os.path.join(self.cache_dir, 'similarity_matrix_cache.pkl')
        
        # Convert numpy arrays to sparse matrices for vstack
        queries_matrix = vstack([csr_matrix(query.reshape(1, -1)) for query in queries])
        keys_matrix = vstack([csr_matrix(key.reshape(1, -1)) for key in keys])
        
        # Check if cached similarity matrix exists
        if os.path.exists(cache_path):
            print('Loading cached similarity matrix...')
            with open(cache_path, 'rb') as f:
                similarity_matrix = pickle.load(f)
        else:
            # Compute and cache similarity matrix
            print('Building similarity matrix...')
            similarity_matrix = metric(queries_matrix, keys_matrix)
            
            # Save to cache
            with open(cache_path, 'wb') as f:
                pickle.dump(similarity_matrix, f)
        
        print('Retrieving best matches...')
        best_match_indices = similarity_matrix.argmax(axis=1)    
        return [self.keys_df.iloc[i]['model_response'] for i in best_match_indices]

    def get_results(self, get_rep):
        print("let's start!\n")
        keys_prompts = self.train_df['user_prompt'].to_list()
        queries_prompts = self.test_df['user_prompt'].to_list()
        
        # Set keys and queries dataframes
        self.keys_df = self.train_df
        self.queries_df = self.test_df
        
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
    get_responses = ContinuousRep(
        train_df, 
        test_df, 
        '../fasttext_models/cc.en.300.vec.gz', 
        '../fasttext_models/cc.es.300.vec.gz',
        '../fasttext_models/cc.pt.300.vec.gz'
    )
    cont_results, cont_run_time = get_responses.get_results(get_responses.get_continuous_rep)
    evaluation, cont_run_time = get_responses.compute_BLEU(cont_results)
    bleu_list = evaluation['bleu_score'].to_list()
    avg_BLEU = np.sum(i for i in bleu_list) / len(bleu_list)
    print(f'CONTINUOUS CASE, TASK 2:\nRunning time: {cont_run_time:.2f}\nResults:', evaluation.head())
    print('\nAverage BLEU score (task 2):', avg_BLEU)

if __name__ == "__main__":
    main()