import spacy
import pandas as pd
import numpy as np
import time
import datetime
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from gensim.models import KeyedVectors
from numpy.linalg import norm

'''
how to run: 
download the model crawl-300d-2M.vec at this link https://fasttext.cc/docs/en/english-vectors.html
change paths to data and fasttext model from the global variables train_df, test_df, fasttext_model_path
'''

'''
this code computes similarity between prompts from a test set and from a train set, matching test prompts with most similar response
TRACK 2/3: prompts are represented with a continuous representation, in particular using fasttext pre-trained models https://fasttext.cc/docs/en/english-vectors.html

runs in 20 to 30 mins first time, 8 mins then! 
Average BLEU score (task 2): 0.08803375476489712
'''


nlp = spacy.load("en_core_web_sm")   #https://spacy.io/usage/models, multi-language, chosen for accuracy
start_time=time.time()
now=datetime.datetime.now()
print('starting time:', now)

# CHANGE PATHS HERE!
# load global variables
train_df=pd.read_csv('../../code/data/train_responses.csv')
test_df=pd.read_csv('../../code/data/dev_responses.csv')
fasttext_model_path='../../code/fasttext_models/crawl-300d-2M.vec'
from gensim.models import KeyedVectors


class Track2:
    def __init__(self, keys_df, queries_df, model_path):
        self.keys_df = keys_df
        self.queries_df = queries_df
        self.model_path = model_path
        # use the caching function:
        self.model = KeyedVectors.load_word2vec_format(self.model_path, binary=False)

    def get_word_embedding(self, word):
        try:
            # get word embedding
            return self.model[word.lower()]
        except KeyError:
            # return 0 if word is not in test set: probably this is why we get low (0.088) score, having other languages != english in dataset
            return np.zeros(self.model.vector_size)

    def get_rep(self, corpus):
        # get prompts embeddings as mean of words reps
        reps = []
        for sentence in corpus:
            words = sentence.split()
            word_embeddings = [self.get_word_embedding(word) for word in words]
            sentence_rep = np.mean(word_embeddings, axis=0)
            reps.append(sentence_rep)
        return reps

    def get_bestmatch(self, queries, keys):
        # compute cosine similarity and retrieve most similar key prompt (train prompt) for each query (test prompt)
        best_match_indices = []
        for q in queries:
            best_sim = -1
            best_index = 0
            for idx, k in enumerate(keys):
                cos_sim = np.dot(q, k) / (norm(q) * norm(k)) if norm(q) != 0 and norm(k) != 0 else 0.0
                if cos_sim > best_sim:
                    best_sim = cos_sim
                    best_index = idx
            best_match_indices.append(best_index)
        # for each best match, store the prompt text (for bleu compytation) and index (to build final table)
        return [self.keys_df.iloc[i]['model_response'] for i in best_match_indices], [self.keys_df.iloc[i]['conversation_id'] for i in best_match_indices]

    def get_results(self):
        print("Let's start!\n")
        keys_prompts = self.keys_df['user_prompt'].to_list()
        print('CHECK: length of prompts is', len(keys_prompts))

        queries_prompts = self.queries_df['user_prompt'].to_list()
        print('Computing representations...\n')
        # get train embeddings
        train_reps = self.get_rep(keys_prompts)
        self.keys_df['vector_reps'] = train_reps
        # get test embeddings
        test_reps = self.get_rep(queries_prompts)
        # get best matches
        matches, matches_id = self.get_bestmatch(test_reps, train_reps)
        # store best matches in the queries df (we are relying on the order of test prompts remaining the same!)
        self.queries_df['retrieved_response'] = matches
        self.queries_df['response_id']= matches_id
        return self.queries_df

    @staticmethod
    def compute_BLEU(data):
        # compute bleu score for each match and then average out to get model performance on text similarity
        smoothingfunction = SmoothingFunction()
        
        def safe_bleu(row):
            try:
                model_tokens = str(row['model_response']).split()
                retrieved_tokens = str(row['retrieved_response']).split()
                return sentence_bleu([model_tokens], retrieved_tokens, 
                                     weights=(0.5, 0.5, 0, 0),
                                     smoothing_function=smoothingfunction.method3)
            except Exception as e:
                print(f"Error in calculating BLEU for row: {row.name}, error: {e}")
                return 0
        
        data['bleu_score'] = data.apply(safe_bleu, axis=1)
        return data


def main():
    start_time = time.time()
    # create instance of class and get results (matches and avg bleu)
    get_responses = Track2(train_df, test_df, fasttext_model_path)
    cont_results = get_responses.get_results()
    evaluation = get_responses.compute_BLEU(cont_results)
    bleu_list = evaluation['bleu_score'].to_list()
    avg_BLEU = np.mean(bleu_list)

    print(f'CONTINUOUS CASE, TASK 1:\nRunning time:  {time.time() - start_time:.2f}\nResults:', evaluation.head())
    print('\nAverage BLEU score (task 2):', avg_BLEU )

if __name__ == "__main__":
    main()
