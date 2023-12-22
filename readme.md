# Movie Recommendation App

This is a simple movie recommendation app built using Streamlit, joblib, and various Python libraries for natural language processing and machine learning.

## Overview

The application recommends movies based on their similarity to the input movie title. It uses the concept of cosine similarity on movie descriptions, genres, keywords, cast, and crew information to generate recommendations.

## How to Run the App

To run the app, follow these steps:

1. Install the required dependencies by running:

    ```bash
    pip install -r requirements.txt
    ```

2. Make sure you have the necessary dataset files (`credits.csv` and `movies.csv`) in the same directory as your application.


3. To Train the Data First Run:

    ```bash
    python training.py
    ```
It will create a Trained Data in you current folder 

1. Execute the Streamlit app after Training:

    ```bash
    streamlit run moviesRec.py
    ```

2. Note, Once the data is trained you will not need to run the training.py again you can use the already trained model to run the app


## Data Preprocessing

- The application reads movie information from two CSV files: `credits.csv` and `movies.csv`.
- It merges the datasets, extracts relevant columns, and handles missing values.
- Text data (e.g., genres, keywords, cast, crew, overview) is processed and transformed into a format suitable for similarity calculations.

## Model Training and Saving

- The TF-IDF method is used to convert text into vectors.
- The cosine similarity matrix is calculated based on the vectors.
- The resulting data, vectorizer, and similarity matrix are saved using joblib for future use.

## Movie Recommendation

- User inputs a movie title.
- The application retrieves the pre-trained data, vectorizer, and similarity matrix.
- Cosine similarity is used to find the most similar movies.
- The top 10 recommended movies are displayed.

Contact for bugs sumeetgupta3690@gmail.com