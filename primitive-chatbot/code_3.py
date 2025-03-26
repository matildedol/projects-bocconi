from sentence_transformers import SentenceTransformer, util
import time
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import datetime
import pandas as pd
import spacy
import numpy as np

# 1. Load a pretrained Sentence Transformer model

nlp = spacy.load("en_core_web_sm")   #https://spacy.io/usage/models, multi-language, chosen for accuracy

start_time=time.time()
now=datetime.datetime.now()
print('starting time:', now)
train_df=pd.read_csv('data/train_responses.csv')
test_df=pd.read_csv('data/dev_responses.csv')

class Track3:
    def __init__(self, keys_df, queries_df, model):
        self.keys_df=keys_df
        self.queries_df=queries_df
        self.model=model

    def get_sentence_embeddings(self, sentences):
        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        return embeddings
    
    def compute_similarity(self, queries, keys):
        cos_sim_matrix = util.cos_sim(queries, keys)
        return cos_sim_matrix
    
    # def get_bestmatches(self, sim_matrix):
    #     for query in sim_matrix:
    #         best_match_indices = query.argmax()    
    #     return [self.keys_df.iloc[i]['model_response'] for i in best_match_indices]      
    def get_bestmatches(self, sim_matrix):
        best_match_indices = [query.argmax().item() for query in sim_matrix]
        return [self.keys_df.iloc[i]['model_response'] for i in best_match_indices]


    def get_results(self, get_rep):
        print("let's start!\n")
        keys_prompts = self.keys_df['user_prompt'].to_list()
        queries_prompts = self.queries_df['user_prompt'].to_list()
        print('computing representations...\n')
        train_reps=get_rep(keys_prompts)
        self.keys_df['vector_reps']=train_reps.tolist() #add representations to df to pivot later
        train_time=time.time() - start_time
        print(f'\ntraining set representations: done! time: {train_time:.2f} s\n')
        test_reps=get_rep(queries_prompts)
        test_time=time.time() - train_time
        print(f'test set representations: done! time: {test_time:.2f} s\n')
        matches=self.get_bestmatches( self.compute_similarity(test_reps, train_reps))
        match_time=time.time() - test_time
        print(f'matches retrieved! time: {match_time:.2f} s\n')
        self.queries_df['retrieved_response']=matches
        end_time=time.time() - start_time
        return self.queries_df, end_time
    
    @staticmethod
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

def load_or_cache_model(model_name='all-MiniLM-L6-v2', cache_path='cached_model.pkl'):
    from joblib import dump, load
    import os
    if os.path.exists(cache_path):
        print("Loading model from cache...")
        model = load(cache_path)
    else:
        print("Downloading and caching model...")
        model = SentenceTransformer(model_name)
        dump(model, cache_path)
    return model

model = load_or_cache_model()

def main():
    get_responses=Track3(train_df, test_df, model)
    cont_results,cont_run_time=get_responses.get_results(get_responses.get_sentence_embeddings)
    evaluation, cont_run_time = get_responses.compute_BLEU(cont_results)
    bleu_list=evaluation['bleu_score'].to_list()
    avg_BLEU=np.sum(i for i in bleu_list)/len(bleu_list)
    print(f'MISC CASE, TASK 3:\nRunning time: {cont_run_time:.2f}\nResults:', evaluation.head())
    print('\nAverage BLEU score (task 3):', avg_BLEU )

if __name__ == "__main__":
    main()
