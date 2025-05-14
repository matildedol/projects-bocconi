from code_1 import Track1 
from code_2 import Track2
from code_3 import Track3, load_or_cache_model
import pandas as pd

'''
code to build final csv tables! communicates with code_1.py, code_2.py, code_3.py
'''

# datasets
train_df=pd.read_csv('../../code/data/train_responses.csv')
test_df=pd.read_csv('../../code/data/test_prompts.csv')

# models
fasttext_model_path='../../code/fasttext_models/crawl-300d-2M.vec'
sentence_transormer_model=load_or_cache_model()

# create instances 
track_1=Track1(train_df, test_df)
track_2=Track2(train_df, test_df, fasttext_model_path)
track_3=Track3(train_df, test_df, sentence_transormer_model)

# build final tables with conversation id and matched response id
def get_tables(track):
    # retreive conversation_id from test df
    x=test_df['conversation_id'].to_list()
    # retreive response_id from codes 1,2,3 computing similarity 
    matches_df=track.get_results()
    y=matches_df['response_id'].to_list()
    # create csv
    results_df=pd.DataFrame({'conversation_id':x, 'response_id':y})
    print(results_df[:5])
    return results_df

# create tables for each track 
get_tables(track_1).to_csv('track_1_test.csv', index=False)
get_tables(track_2).to_csv('track_2_test.csv', index=False)
get_tables(track_3).to_csv('track_3_test.csv', index=False) 

