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
how to run: 
change paths to data from the global variables train_df, test_df
'''

'''
this code computes similarity between prompts from a test set and from a train set, matching test prompts with most similar response
TRACK 1/3: prompts are represented with a discrete representation. in particular a tf-idf score with 1 to 4 character ngrams 

runs in around 8 s
Average BLEU score (task 1): 0.09064968816651113
'''


nlp = spacy.load("en_core_web_sm")   #https://spacy.io/usage/models, multi-language, chosen for accuracy
start_time=time.time()
now=datetime.datetime.now()
print('starting time:', now)

# CHANGE PATHS HERE!
# load data
train_df=pd.read_csv('../../code/data/train_responses.csv')
test_df=pd.read_csv('../../code/data/dev_responses.csv')


class Track1:
    def __init__(self, keys_df, queries_df):
        self.keys_df=keys_df
        self.queries_df=queries_df

    # get static embedding for each prompt
    def get_rep(self, corpus, ref_corpus, visualize):
        vectorizer=TfidfVectorizer(analyzer='char', ngram_range=(1,4), sublinear_tf=True)                                                                           
        vectorizer.fit(ref_corpus)
        # create matrix: each row is a prompt, each column an ngram, entries are frequencies                                                                                              
        X= vectorizer.transform(corpus) 
        # store embeddings in a list (relies on order)                                                                                       
        reps_list=[row for row in X]                                                                                           

        # auxiliary function to visualize df
        if visualize=='y':
            X_array=X.toarray()                                                                                                    
            print(X[:4])
            feature_names = vectorizer.get_feature_names_out()
            df_features = pd.DataFrame(X_array, columns=feature_names, index=corpus[:len(X_array)])
            print(df_features)

        return reps_list

    # get matches for each test (query) prompt
    def get_bestmatch(self, queries, keys, metric=cosine_similarity):  
        # process sparse matrices with vstack 
        queries_matrix = vstack(queries)
        keys_matrix = vstack(keys)
        print('building similarity matrix...')
        # use sklearn built-in cosine_similarity function as metric
        similarity_matrix = metric(queries_matrix, keys_matrix)
        print('retrieving best matches...')
        best_match_indices = similarity_matrix.argmax(axis=1)    
        # for each best match, store the prompt text (for bleu compytation) and index (to build final table)
        return [self.keys_df.iloc[i]['model_response'] for i in best_match_indices],[self.keys_df.iloc[i]['conversation_id'] for i in best_match_indices]                             

    def get_results(self):
        print("let's start!\n")
        keys_prompts = self.keys_df['user_prompt'].to_list()
        queries_prompts = self.queries_df['user_prompt'].to_list()
        
        print('computing representations...\n')
        # get train set embeddings
        train_reps=self.get_rep(keys_prompts, keys_prompts, 'n')
        self.keys_df['vector_reps']=train_reps #add representations to df to pivot later
        train_time=time.time() - start_time
        print(f'\ntraining set representations: done! time: {train_time:.2f} s\n')
        # get test set embeddings
        test_reps=self.get_rep(queries_prompts, keys_prompts, 'n')
        test_time=time.time() - train_time
        print(f'test set representations: done! time: {test_time:.2f} s\n')
        # get matches
        matches, matches_id=self.get_bestmatch(test_reps, train_reps, cosine_similarity)
        match_time=time.time() - test_time
        print(f'matches retrieved! time: {match_time:.2f} s\n')
        # store matches (train most similar prompt to each test prompt) and their id in queries df
        self.queries_df['retrieved_response']=matches
        self.queries_df['response_id']=matches_id
        return self.queries_df

    #compute BLEU score to assess performance (bleu for each match, and then average out to have a measure for model)
    @staticmethod
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
    start_time = time.time()
    # create instance and get results (matches and avg bleu)
    get_responses = Track1(train_df, test_df)
    disc_results = get_responses.get_results()
    evaluation = get_responses.compute_BLEU(disc_results)
    bleu_list = evaluation['bleu_score'].to_list()
    avg_BLEU = np.mean(bleu_list)

    print(f'DISCRETE CASE, TASK 1:\nRunning time:  {time.time() - start_time:.2f}\nResults:', evaluation.head())
    print('\nAverage BLEU score (task 1):', avg_BLEU )

if __name__ == "__main__":
    main()

