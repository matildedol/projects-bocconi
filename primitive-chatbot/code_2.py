import spacy
import pandas as pd
import numpy as np
import time
import datetime
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from gensim.models import KeyedVectors
from numpy.linalg import norm

'''
takes around 8 minutes to run! 
'''

nlp = spacy.load("en_core_web_sm")   #https://spacy.io/usage/models, multi-language, chosen for accuracy

start_time=time.time()
now=datetime.datetime.now()
print('starting time:', now)

train_df=pd.read_csv('data/train_responses.csv')
test_df=pd.read_csv('data/dev_responses.csv')
fasttext_model_path='crawl-300d-2M.vec'
from gensim.models import KeyedVectors


class ContinuousRep:
    def __init__(self, keys_df, queries_df, model_path):
        self.keys_df = keys_df
        self.queries_df = queries_df
        self.model_path = model_path
        # Use the caching function:
        self.model = KeyedVectors.load_word2vec_format(self.model_path, binary=False)

    def get_word_embedding(self, word):
        try:
            return self.model[word.lower()]
        except KeyError:
            return np.zeros(self.model.vector_size)

    def get_continuous_rep(self, corpus):
        """
        Cached continuous representation generation with new representation method.
        """
        reps = []
        for sentence in corpus:
            words = sentence.split()
            word_embeddings = [self.get_word_embedding(word) for word in words]
            sentence_rep = np.mean(word_embeddings, axis=0)
            reps.append(sentence_rep)
        print("Example vector slice:", reps[0][:5])
        return reps

    def get_bestmatch(self, queries, keys):
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
        return [self.keys_df.iloc[i]['model_response'] for i in best_match_indices]

    def get_results(self, get_rep):
        print("Let's start!\n")
        keys_prompts = self.keys_df['user_prompt'].to_list()
        print('CHECK: length of prompts is', len(keys_prompts))

        queries_prompts = self.queries_df['user_prompt'].to_list()
        print('Computing representations...\n')

        train_reps = get_rep(keys_prompts)
        self.keys_df['vector_reps'] = train_reps

        test_reps = get_rep(queries_prompts)
        matches = self.get_bestmatch(test_reps, train_reps)
        self.queries_df['retrieved_response'] = matches
        return self.queries_df

    @staticmethod
    def compute_BLEU(data):
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

    # Load your DataFrames as before
    train_df = pd.read_csv('data/train_responses.csv')
    test_df = pd.read_csv('data/dev_responses.csv')
    fasttext_model_path = 'crawl-300d-2M.vec'

    get_responses = ContinuousRep(train_df, test_df, fasttext_model_path)
    cont_results = get_responses.get_results(get_responses.get_continuous_rep)
    evaluation = get_responses.compute_BLEU(cont_results)
    bleu_list = evaluation['bleu_score'].to_list()
    avg_BLEU = np.mean(bleu_list)

    print("\nAverage BLEU score:", avg_BLEU)
    print(f"Total runtime: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
