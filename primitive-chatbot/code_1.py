import spacy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


nlp = spacy.load("en_core_web_sm")   #https://spacy.io/usage/models, multi-language, chosen for accuracy

# function to get representation of each prompt in a dataframe 
def get_representation(df):
    prompts=df['user_prompt'].to_list()
    # create dependency_based couples 'lemma_head'
    features = [' '.join('{}_{}'.format(token.lemma_, token.head.lemma_) for token in nlp(sent)) for sent in prompts]
    ## print(features[:3])
    # create matrix: each row is a prompt, each column a couple lemma_head, entries are frequencies
    vectorizer=CountVectorizer()
    X = vectorizer.fit_transform(features)
    # visualize it
    X_array=X.toarray()
    ## print(X[:4])
    feature_names = vectorizer.get_feature_names_out()
    df_features = pd.DataFrame(X_array, columns=feature_names, index=prompts[:len(X_array)])
    ## print(df_features)

    # store representation in df
    prompt_representation = [row for row in X]
    df['vector_representations'] = prompt_representation
    return df

### train_df = get_representation(pd.read_csv('train_prompts.csv'))
### dev_df=get_representation(pd.read_csv('dev_prompts.csv'))

# function to get cosine similarity between a vector query and n other vectors (keys)
def compute_cosinesim(query,key):
    cos_sim = np.dot(query,key) / (np.linalg.norm(query)*np.linalg.norm(key))
    return cos_sim

# now for each prompt in dev_df we want to: find all cosine similarities, store them in dictionary prompt_id:cos_sim and retrieve the key of highest value. or, we compute the

# è per forza O(n^2) !! come farlo piu efficiente?
def get_bestmatch(queries_df, keys_df):
    best_matches=[]
    best_matches_id=[]
    for query in queries_df.itertuples():
        for key in keys_df.itertuples():
            best_cos_sim=0
            cos_sim=compute_cosinesim(query[-1], key[-1]) #questi so che sono gli ultimi perche li ho messi io
            if cos_sim>best_cos_sim:
                best_cos_sim=cos_sim
                best_prompt=key[2] # qui imbroglio un po' eheh
                best_prompt_id=key[0] 
        best_matches.append(best_prompt)
        best_matches_id.append(best_prompt_id)
    queries_df['best_match']=best_matches
    queries_df['best_match_id']=best_matches_id
    return queries_df

train_df=get_representation(pd.read_csv('train_responses.csv'))
test_df=get_representation(pd.read_csv('dev_responses.csv'))

result=get_bestmatch(train_df, test_df)
print(result.head())

# compute BLEU score to assess performance
def compute_BLEU():
    pass






