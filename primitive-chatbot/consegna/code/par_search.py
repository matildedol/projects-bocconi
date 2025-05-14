import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
from scipy.sparse import vstack
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import product

'''
parameters search for Track 1: ngrams min and max character numbers
'''

train_df = pd.read_csv('data/train_responses.csv')
test_df = pd.read_csv('data/dev_responses.csv')

def grid_search_tfidf(train_df, test_df, ngram_ranges=None):
    # define default ngram ranges if not provided
    if ngram_ranges is None:
        ngram_ranges = list(product(range(1, 4), range(2, 7)))
    
    # Store results
    results = []
    
    for min_n, max_n in ngram_ranges:
        # run code 1 for each pair of min and max number of character
        if min_n<=max_n:
            start_time = time.time()
            
            def get_discrete_rep(corpus, ref_corpus, visualize):
                vectorizer = TfidfVectorizer(
                    analyzer='char', 
                    ngram_range=(min_n, max_n), 
                    sublinear_tf=True
                )
                vectorizer.fit(ref_corpus)
                X = vectorizer.transform(corpus)
                reps_list = [row for row in X]
                return reps_list
            
            def get_bestmatch(queries, keys, keys_df, metric=cosine_similarity):    
                queries_matrix = vstack(queries)
                keys_matrix = vstack(keys)
                similarity_matrix = metric(queries_matrix, keys_matrix)
                best_match_indices = similarity_matrix.argmax(axis=1)    
                return [keys_df.iloc[i]['model_response'] for i in best_match_indices]                                 
            
            def get_results(queries_df, keys_df, get_rep):
                keys_prompts = keys_df['user_prompt'].to_list()
                queries_prompts = queries_df['user_prompt'].to_list()
                
                train_reps = get_rep(keys_prompts, keys_prompts, 'n')
                keys_df['vector_reps'] = train_reps
                test_reps = get_rep(queries_prompts, keys_prompts, 'n')
                matches = get_bestmatch(test_reps, train_reps, keys_df, cosine_similarity)
                queries_df['retrieved_response'] = matches
                return queries_df
            
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
                        print(f"error in calculating BLEU for row: {row.name}, error: {e}")
                        return 0
                
                data['bleu_score'] = data.apply(safe_bleu, axis=1)
                return data
            
            # get results
            discrete_results = get_results(test_df, train_df, get_discrete_rep)
            evaluation = compute_BLEU(discrete_results)
            bleu_list = evaluation['bleu_score'].to_list()
            avg_BLEU = np.sum(i for i in bleu_list) / len(bleu_list)
            
            # record and print how each pair of parameters performs
            results.append({
                'ngram_range': (min_n, max_n),
                'avg_bleu_score': avg_BLEU,
                'run_time': time.time() - start_time
            })
            
            print(f"Ngram Range {(min_n, max_n)}: Avg BLEU Score = {avg_BLEU}")
        
    # convert results to df for further analysis (not used)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('avg_bleu_score', ascending=False)
    
    return results_df

# run it!
grid_search_results = grid_search_tfidf(train_df, test_df)
