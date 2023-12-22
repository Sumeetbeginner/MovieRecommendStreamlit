import joblib
import streamlit as st

# Load the saved model components

cv = joblib.load('count_vectorizer.pkl')
similarity = joblib.load('cosine_similarity.pkl')
new_df = joblib.load('new_df.csv')

#movie recommend
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances  = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:11]
    
    recommended_movies = [new_df.iloc[i[0]].title for i in movies_list]
    
    return recommended_movies

# Input box for user to enter a movie
user_input = st.text_input("Enter a movie title:", "Avatar")

if st.button("Get Recommendations"):
    recommendations = recommend(user_input)
    st.write("Recommended Movies:")
    for movie in recommendations:
        st.write(f"- {movie}")

