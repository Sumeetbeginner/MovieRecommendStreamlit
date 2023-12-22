import joblib
import streamlit as st
import numpy as np #Working with arrays/multidim-arrays
import pandas as pd #data analysis and manipulation
import ast #abstract syntax tree 
from sklearn.feature_extraction.text import CountVectorizer
import nltk #The Natural Language Toolkit, or NLTK, is a powerful library in Python for working with human language data. 
from nltk.stem.porter import PorterStemmer
#cosine_similarity checks how much vectors are similar 
from sklearn.metrics.pairwise import cosine_similarity

#Importing Datasets
credits_df = pd.read_csv('credits.csv')
movies_df = pd.read_csv('movies.csv')

#Displaying full dataset as output
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#combining credit and movies dataset
movies_df = movies_df.merge(credits_df, on ='title') #Title Column will not merge
# movies_df.shape

movies_df = movies_df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]  #Adding only particular column 
# movies_df.head()

movies_df.dropna(inplace= True) #Drop All missing values from dataset
movies_df.isnull().sum()

#Taking object data with keywords name and appending it to L 
def convert(obj):
    L =[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L 

def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter !=3:
            L.append(i["name"])
            counter+=1
        else:
            break
        return L

movies_df['genres'] = movies_df['genres'].apply(convert)
movies_df['keywords'] = movies_df['keywords'].apply(convert)
movies_df['cast'] = movies_df['cast'].apply(convert3)    

def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies_df['crew'] = movies_df['crew'].apply(fetch_director)

movies_df['overview'] = movies_df['overview'].apply(lambda x : x.split())

#Removing Spaces
movies_df['genres'] = movies_df['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies_df['keywords'] = movies_df['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies_df['cast'] = movies_df['cast'].apply(lambda x: [i.replace(" ", "") if i is not None else None for i in x] if x is not None else None)
movies_df['crew'] = movies_df['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

#Put all columns tag
movies_df['tags'] = movies_df['overview'] + movies_df['genres'] + movies_df['keywords'] + movies_df['cast'] + movies_df['crew']

#Creating New Dataframe with title and final all tags and movie id
new_df = movies_df[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)

#Making all Lowercase for better predictio
new_df['tags'] = new_df['tags'].apply(lambda x: str(x).lower() if pd.notnull(x) else x)

#Transform Text into Vectors using TF IDF Method or frequency count
new_df = new_df.dropna(subset=['tags'])
cv = CountVectorizer(max_features=5000, stop_words = "english") #Stop Words Means ignores is,the... & max_features meaning 5000 most imp words 
# cv.fit_transform(new_df['tags']).toarray().shape 

#Converting Vector to array
vectors = cv.fit_transform(new_df['tags']).toarray()

ps = PorterStemmer()

def stem(text):
    y =[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)
cosine_similarity(vectors)
similarity = cosine_similarity(vectors)

#Saving the Trained Data
joblib.dump(new_df, 'new_df.csv')
joblib.dump(cv, 'count_vectorizer.pkl')
joblib.dump(similarity, 'cosine_similarity.pkl')


