from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import time
import random
import string
import json
from flask import render_template

app = Flask(__name__)
CORS(app)
# CORS(app, resources={r"/api/*": {"origins": ""}})
# CORS(app, resources={r"/api/*": {"origins": "*"}})
# app = Flask(__name__)
# CORS(app, origins=['*'],
#      methods=['GET', 'POST'])
# # Load your data
df_copy6 = pd.read_csv('Original_Dataset.csv')
umd = pd.read_csv('Unique_movie.csv')

# Preprocess demographic information
ohe = OneHotEncoder()
demographic_features = ['job', 'state', 'gender', 'age_group']
demographic_matrix = ohe.fit_transform(df_copy6[demographic_features])
demographic_df = pd.DataFrame(demographic_matrix.toarray(), columns=ohe.get_feature_names_out(demographic_features))
demographic_df['user_id'] = df_copy6['user_id']
user_profiles = demographic_df.groupby('user_id').mean()
user_sim_matrix = cosine_similarity(user_profiles)
user_ids = user_profiles.index
user_sim_df = pd.DataFrame(user_sim_matrix, index=user_ids, columns=user_ids)

# def recommend_based_on_demographics(user_id, top_n=5):
#     similar_users = user_sim_df[user_id].sort_values(ascending=False).index[1:]
#     similar_users_movies = df_copy6[df_copy6['user_id'].isin(similar_users) & (df_copy6['user_rating'] == 1)]
#     recommended_movies = similar_users_movies['movie_id'].value_counts().head(top_n).index
#     return umd[umd['movie_id'].isin(recommended_movies)][['movie_id', 'name']]

def recommend_based_on_demographics(user_id, top_n=5, pool_size=20):
    similar_users = user_sim_df[user_id].sort_values(ascending=False).index[1:]
    similar_users_movies = df_copy6[(df_copy6['user_id'].isin(similar_users)) & (df_copy6['user_rating'] == 1)]
    similar_users_movies = similar_users_movies[~similar_users_movies['movie_id'].isin(df_copy6[df_copy6['user_id'] == user_id]['movie_id'])]

    # Get the top movie IDs from similar users, ensuring there are no duplicates
    recommended_movies = similar_users_movies['movie_id'].value_counts().index
    if len(recommended_movies) < pool_size:
        recommended_movies = recommended_movies[:pool_size]
    else:
        # Introduce some randomness
        recommended_movies = recommended_movies.to_series().sample(frac=1, random_state=None).head(pool_size)

    final_recommendations = recommended_movies[:top_n]

    return umd[umd['movie_id'].isin(final_recommendations)][['movie_id', 'name']]



# Create user-item interaction matrix and apply SVD
user_item_matrix = df_copy6.pivot(index='user_id', columns='movie_id', values='user_rating').fillna(0)
svd = TruncatedSVD(n_components=50)
user_factors = svd.fit_transform(user_item_matrix)
item_factors = svd.components_
predicted_ratings = np.dot(user_factors, item_factors)
predicted_ratings_df = pd.DataFrame(predicted_ratings, index=user_item_matrix.index, columns=user_item_matrix.columns)

# def recommend_based_on_cf(user_id, top_n=5):
#     user_predicted_ratings = predicted_ratings_df.loc[user_id]
#     top_movie_ids = user_predicted_ratings.sort_values(ascending=False).head(top_n).index
#     return umd[umd['movie_id'].isin(top_movie_ids)][['movie_id', 'name']]

def recommend_based_on_cf(user_id, initial_top_n=20, final_top_n=5):
    # Get the user's predicted ratings
    user_predicted_ratings = predicted_ratings_df.loc[user_id]

    # Get the movies the user has already rated
    rated_movie_ids = df_copy6[df_copy6['user_id'] == user_id]['movie_id'].values

    # Filter out the movies the user has already rated
    user_predicted_ratings = user_predicted_ratings.drop(rated_movie_ids, errors='ignore')

    # Get the top initial_top_n movie IDs
    top_movie_ids = user_predicted_ratings.sort_values(ascending=False).head(initial_top_n).index

    # Return the top final_top_n recommended movies
    return umd[umd['movie_id'].isin(top_movie_ids)].head(final_top_n)[['movie_id', 'name']]

# Vectorize genres using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(umd['genre'].apply(lambda x: x.split(',')))
cosine_sim_matrix = cosine_similarity(genre_matrix, genre_matrix)
movie_ids = umd['movie_id'].values
cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=movie_ids, columns=movie_ids)

def recommend_based_on_genre(user_id, top_n=5):
    user_data = df_copy6[df_copy6['user_id'] == user_id]
    liked_movies = user_data[user_data['user_rating'] == 1]['movie_id'].values
    if len(liked_movies) == 0:
        return pd.DataFrame(columns=['movie_id', 'name'])
    sim_scores = cosine_sim_df[liked_movies].mean(axis=1)
    sim_scores = sim_scores.drop(liked_movies, errors='ignore')
    top_movie_ids = sim_scores.nlargest(top_n).index
    return umd[umd['movie_id'].isin(top_movie_ids)][['movie_id', 'name']]

# Compute similarity matrices for cast and director
def compute_similarity_matrix(column):
    mlb = MultiLabelBinarizer()
    matrix = mlb.fit_transform(umd[column].apply(lambda x: x.split(',')))
    cosine_sim = cosine_similarity(matrix, matrix)
    return pd.DataFrame(cosine_sim, index=umd['movie_id'], columns=umd['movie_id'])

cosine_sim_cast_df = compute_similarity_matrix('cast')
cosine_sim_director_df = compute_similarity_matrix('director')

def recommend_based_on_cast(user_id, top_n=5):
    user_data = df_copy6[df_copy6['user_id'] == user_id]
    liked_movies = user_data[user_data['user_rating'] == 1]['movie_id'].values
    if len(liked_movies) == 0:
        return pd.DataFrame(columns=['movie_id', 'name', 'cast'])
    sim_scores = cosine_sim_cast_df[liked_movies].mean(axis=1)
    sim_scores = sim_scores.drop(liked_movies, errors='ignore')
    top_movie_ids = sim_scores.nlargest(top_n).index
    return umd[umd['movie_id'].isin(top_movie_ids)][['movie_id', 'name', 'cast']]

def recommend_based_on_director(user_id, top_n=5):
    user_data = df_copy6[df_copy6['user_id'] == user_id]
    liked_movies = user_data[user_data['user_rating'] == 1]['movie_id'].values
    if len(liked_movies) == 0:
        return pd.DataFrame(columns=['movie_id', 'name', 'director'])
    sim_scores = cosine_sim_director_df[liked_movies].mean(axis=1)
    sim_scores = sim_scores.drop(liked_movies, errors='ignore')
    top_movie_ids = sim_scores.nlargest(top_n).index
    return umd[umd['movie_id'].isin(top_movie_ids)][['movie_id', 'name', 'director']]

# Popular Movies Recommender
def recommend_popular_movies(top_n=5):
    movie_popularity = df_copy6['movie_id'].value_counts().head(top_n).index
    return umd[umd['movie_id'].isin(movie_popularity)][['movie_id', 'name']]

# Helper function to get movie IDs from names
def get_movie_ids_from_names(movie_names):
    return umd[umd['name'].isin(movie_names)]['movie_id'].values

def get_age_group(age):
    try:
        age = int(age)
    except ValueError:
        return None
    if 10 <= age <= 14:
        return '10-14'
    elif 15 <= age <= 19:
        return '15-19'
    elif 20 <= age <= 24:
        return '20-24'
    elif 25 <= age <= 29:
        return '25-29'
    elif 30 <= age <= 34:
        return '30-34'
    elif 35 <= age <= 39:
        return '35-39'
    elif 40 <= age <= 44:
        return '40-44'
    elif 45 <= age <= 49:
        return '45-49'
    elif 50 <= age <= 54:
        return '50-54'
    elif 55 <= age <= 59:
        return '55-59'
    elif 60 <= age <= 64:
        return '60-64'
    elif 65 <= age <= 69:
        return '65-69'
    elif 70 <= age <= 74:
        return '70-74'
    elif 75 <= age <= 79:
        return '75-79'
    elif 80 <= age <= 84:
        return '80-84'
    else:
        return None

generated_ids = set()
def generate_unique_user_id():
    while True:
        # Generate a timestamp
        timestamp = int(time.time() * 1000)  # milliseconds since epoch

        # Generate a random string
        random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=6))

        # Combine timestamp and random string to form the user_id
        user_id = f"{timestamp}{random_str}"

        # Ensure the ID is unique
        if user_id not in generated_ids:
            generated_ids.add(user_id)
            return user_id

@app.route('/api/recommend/demographic', methods=['POST'])
def recommend_demographic():
    data = request.json
    user_id = generate_unique_user_id()
    age = data.get('age', 0)
    # Convert age to integer
    try:
        age = int(age)
    except ValueError:
        return jsonify({'error': 'Invalid age provided'}), 400
    
    age_group = get_age_group(age)

    if not age_group:
        return jsonify({'error': 'Invalid age provided'}), 400
    
    new_data = {
        'user_id': user_id,
        'languages': data.get('languages', ''),
        'job': data.get('job', ''),
        'state': data.get('state', ''),
        'dob': '',
        'gender': data.get('gender', ''),
        'movie_id': '',
        'user_rating': 0,
        'description': '',
        'm_languages': '',
        'released': '',
        'imdb_rating': 0.0,
        'writer': '',
        'director': '',
        'cast': '',
        'genre': '',
        'name': '',
        'age': age,
        'age_group': age_group
    }
    global df_copy6
    df_copy6 = pd.concat([df_copy6, pd.DataFrame([new_data])], ignore_index=True)

    # Reprocess the demographic information to update user profiles and similarity matrix
    demographic_matrix = ohe.fit_transform(df_copy6[demographic_features])
    demographic_df = pd.DataFrame(demographic_matrix.toarray(), columns=ohe.get_feature_names_out(demographic_features))
    demographic_df['user_id'] = df_copy6['user_id']
    user_profiles = demographic_df.groupby('user_id').mean()
    user_sim_matrix = cosine_similarity(user_profiles)
    user_ids = user_profiles.index
    global user_sim_df
    user_sim_df = pd.DataFrame(user_sim_matrix, index=user_ids, columns=user_ids)

    recommendations = recommend_based_on_demographics(user_id)
    return jsonify({'recommendations': recommendations.to_dict(orient='records')})

@app.route('/api/recommend/liked', methods=['POST'])
def recommend_liked():
    data = request.json
    user_id = generate_unique_user_id()
    liked_movie_names = data['likedMovies']
    liked_movie_ids = get_movie_ids_from_names(liked_movie_names)
    for movie_id in liked_movie_ids:
        new_data = {
            'user_id': user_id,
            'languages': '',
            'job': '',
            'state': '',
            'dob': '',
            'gender': '',
            'movie_id': movie_id,
            'user_rating': 1,
            'description': '',
            'm_languages': '',
            'released': '',
            'imdb_rating': 0.0,
            'writer': '',
            'director': '',
            'cast': '',
            'genre': '',
            'name': '',
            'age': 0,
            'age_group': ''
        }
        global df_copy6
        df_copy6 = pd.concat([df_copy6, pd.DataFrame([new_data])], ignore_index=True)

    # Recreate user-item interaction matrix and apply SVD
    user_item_matrix = df_copy6.pivot(index='user_id', columns='movie_id', values='user_rating').fillna(0)
    svd = TruncatedSVD(n_components=50)
    user_factors = svd.fit_transform(user_item_matrix)
    item_factors = svd.components_
    global predicted_ratings_df
    predicted_ratings = np.dot(user_factors, item_factors)
    predicted_ratings_df = pd.DataFrame(predicted_ratings, index=user_item_matrix.index, columns=user_item_matrix.columns)

    recommendations = recommend_based_on_cf(user_id)
    return jsonify({'recommendations': recommendations.to_dict(orient='records')})

@app.route('/api/recommend/genre', methods=['POST'])
def recommend_genre():
    data = request.json
    user_id = generate_unique_user_id()
    liked_movie_names = data['likedMovies']
    liked_movie_ids = get_movie_ids_from_names(liked_movie_names)
    for movie_id in liked_movie_ids:
        new_data = {
            'user_id': user_id,
            'languages': '',
            'job': '',
            'state': '',
            'dob': '',
            'gender': '',
            'movie_id': movie_id,
            'user_rating': 1,
            'description': '',
            'm_languages': '',
            'released': '',
            'imdb_rating': 0.0,
            'writer': '',
            'director': '',
            'cast': '',
            'genre': '',
            'name': '',
            'age': 0,
            'age_group': ''
        }
        global df_copy6
        df_copy6 = pd.concat([df_copy6, pd.DataFrame([new_data])], ignore_index=True)

    # Recompute cosine similarity matrix for genres
    genre_matrix = mlb.fit_transform(umd['genre'].apply(lambda x: x.split(',')))
    global cosine_sim_df
    cosine_sim_matrix = cosine_similarity(genre_matrix, genre_matrix)
    movie_ids = umd['movie_id'].values
    cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=movie_ids, columns=movie_ids)

    recommendations = recommend_based_on_genre(user_id)
    return jsonify({'recommendations': recommendations.to_dict(orient='records')})

@app.route('/api/recommend/cast-director', methods=['POST'])
def recommend_cast_and_director():
    data = request.json
    user_id = generate_unique_user_id()
    liked_movie_names = data['likedMovies']
    liked_movie_ids = get_movie_ids_from_names(liked_movie_names)
    for movie_id in liked_movie_ids:
        new_data = {
            'user_id': user_id,
            'languages': '',
            'job': '',
            'state': '',
            'dob': '',
            'gender': '',
            'movie_id': movie_id,
            'user_rating': 1,
            'description': '',
            'm_languages': '',
            'released': '',
            'imdb_rating': 0.0,
            'writer': '',
            'director': '',
            'cast': '',
            'genre': '',
            'name': '',
            'age': 0,
            'age_group': ''
        }
        global df_copy6
        df_copy6 = pd.concat([df_copy6, pd.DataFrame([new_data])], ignore_index=True)

    # Recompute similarity matrices for cast and director
    global cosine_sim_cast_df 
    cosine_sim_cast_df = compute_similarity_matrix('cast')
    global cosine_sim_director_df
    cosine_sim_director_df = compute_similarity_matrix('director')

    cast_recommendations = recommend_based_on_cast(user_id)

    return jsonify({'recommendations': cast_recommendations.to_dict(orient='records')})

@app.route('/api/recommend/director', methods=['POST'])
def recommend_sdirector():
    data = request.json
    user_id = generate_unique_user_id()
    liked_movie_names = data['likedMovies']
    liked_movie_ids = get_movie_ids_from_names(liked_movie_names)
    for movie_id in liked_movie_ids:
        new_data = {
            'user_id': user_id,
            'languages': '',
            'job': '',
            'state': '',
            'dob': '',
            'gender': '',
            'movie_id': movie_id,
            'user_rating': 1,
            'description': '',
            'm_languages': '',
            'released': '',
            'imdb_rating': 0.0,
            'writer': '',
            'director': '',
            'cast': '',
            'genre': '',
            'name': '',
            'age': 0,
            'age_group': ''
        }
        global df_copy6
        df_copy6 = pd.concat([df_copy6, pd.DataFrame([new_data])], ignore_index=True)

    # Recompute similarity matrices for cast and director
    global cosine_sim_cast_df
    cosine_sim_cast_df = compute_similarity_matrix('cast')
    global cosine_sim_director_df
    cosine_sim_director_df = compute_similarity_matrix('director')
    director_recommendations = recommend_based_on_director(user_id)

    return jsonify({'recommendations': director_recommendations.to_dict(orient='records')})

@app.route('/api/movies', methods=['GET'])
def get_movies():
    with open('movies.json', 'r') as file:
        movies = json.load(file)
    return jsonify(movies)


@app.route('/api/dataset', methods=['GET'])
def show_dataset():
    with open('movies.json', 'r') as file:
        movies = json.load(file)
    return render_template('dataset.html', movies=movies)

if __name__ == '__main__':
    app.run(debug=True)
