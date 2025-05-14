from sentence_transformers import SentenceTransformer, util
import time
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import datetime
import pandas as pd
import spacy
import numpy as np

'''
how to run: 
change paths to data from the global variables train_df, test_df
'''
'''
runs in aorund 24 s (from second run)
Average BLEU score (task 3): 0.10233936335610455
'''

nlp = spacy.load("en_core_web_sm")   #https://spacy.io/usage/models, multi-language, chosen for accuracy
start_time=time.time()
now=datetime.datetime.now()
print('starting time:', now)

# CHANGE PATHS HERE!
# load global vars (datasets)
train_df=pd.read_csv('../../code/data/train_responses.csv')
test_df=pd.read_csv('../../code/data/dev_responses.csv')

# load model with cache
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

class Track3:
    def __init__(self, keys_df, queries_df, model):
        self.keys_df=keys_df
        self.queries_df=queries_df
        self.model=model

    # get sentence embeddings
    def get_rep(self, sentences):
        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        return embeddings
    
    # compute similarity with built-in function
    def compute_similarity(self, queries, keys):
        cos_sim_matrix = util.cos_sim(queries, keys)
        return cos_sim_matrix
        
    # get best matches
    def get_bestmatches(self, sim_matrix):
        best_match_indices = [query.argmax().item() for query in sim_matrix]
        # for each best match, store the prompt text (for bleu computation) and index (to build final table)
        return [self.keys_df.iloc[i]['model_response'] for i in best_match_indices], [self.keys_df.iloc[i]['conversation_id'] for i in best_match_indices]


    def get_results(self):
        print("let's start!\n")
        keys_prompts = self.keys_df['user_prompt'].to_list()
        queries_prompts = self.queries_df['user_prompt'].to_list()
        print('computing representations...\n')
        # get train reps
        train_reps=self.get_rep(keys_prompts)
        self.keys_df['vector_reps']=train_reps.tolist() #add representations to df to pivot later
        train_time=time.time() - start_time
        print(f'\ntraining set representations: done! time: {train_time:.2f} s\n')
        # get test reps
        test_reps=self.get_rep(queries_prompts)
        test_time=time.time() - train_time
        print(f'test set representations: done! time: {test_time:.2f} s\n')
        # get best matches
        matches, matches_id=self.get_bestmatches( self.compute_similarity(test_reps, train_reps))
        match_time=time.time() - test_time
        print(f'matches retrieved! time: {match_time:.2f} s\n')
        # store matches and their id in queries df 
        self.queries_df['retrieved_response']=matches
        self.queries_df['response_id']=matches_id
        return self.queries_df
    
    @staticmethod
    # compute bleu score (on each match and then average out to get a measure of model performance)
    def compute_BLEU(data):
        smoothingfunction = SmoothingFunction()
        
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
        return data

def main():
    # create instance of class and get results
    get_responses=Track3(train_df, test_df, model)
    misc_results=get_responses.get_results(get_responses.get_rep)
    evaluation = get_responses.compute_BLEU(misc_results)
    bleu_list=evaluation['bleu_score'].to_list()
    # average out bleu
    avg_BLEU=np.sum(i for i in bleu_list)/len(bleu_list)
    print(f'MISC CASE, TASK 3:\nRunning time: {time.time() - start_time:.2f}\nResults:', evaluation.head())
    print('\nAverage BLEU score (task 3):', avg_BLEU )

if __name__ == "__main__":
    main()
