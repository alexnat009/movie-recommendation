import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


credits_df = pd.read_csv(f'./dataset/tmdb_5000_credits.csv')
movies_df = pd.read_csv(f'./dataset/tmdb_5000_movies.csv')
credits_df.columns = ['id', 'title', 'cast', 'crew']
movies_df = movies_df.merge(credits_df, left_on=['id', 'title'], right_on=['id', 'title'])

features1 = ['cast', 'crew', 'keywords', 'genres']

for feature in features1:
    movies_df[feature] = movies_df[feature].apply(literal_eval)


def get_director(x):
    """get_director() function extracts the name of the director of the movie."""
    for i in x:
        if i['job'] == "Director":
            return i['name']
    return np.nan


def get_list(x):
    """get_list() returns the top 3 elements or the entire list whichever is more."""
    if isinstance(x, list):
        names = [i['name'] for i in x]

        # in original code there was limit for 3 element, but removing it got me better results I think.
        # if len(names) > 3:
        #     names = names[:3]
        return names
    return []


"""we passed the “crew” information to the get_director() function, extracted the name, and created a new column “director”.
For the features cast, keyword and genres we extracted the top information by applying the get_list() function"""
movies_df['director'] = movies_df['crew'].apply(get_director)

features2 = ['cast', 'keywords', 'genres']

for feature in features2:
    movies_df[feature] = movies_df[feature].apply(get_list)


def clean_data(row):
    if isinstance(row, list):
        return [str.lower(i.replace(" ", "")) for i in row]
    else:
        if isinstance(row, str):
            return str.lower(row.replace(" ", ""))
        else:
            return ""


features3 = ['cast', 'keywords', 'director', 'genres']
for feature in features3:
    movies_df[feature] = movies_df[feature].apply(clean_data)


def create_soup(features):
    return f'{" ".join(features["keywords"])} ' \
           f'{" ".join(features["cast"])} ' \
           f'{" ".join(features["director"])} ' \
           f'{" ".join(features["genres"])} '


movies_df['soup'] = movies_df.apply(create_soup, axis=1)

ct = CountVectorizer(stop_words='english')
count_matrix = ct.fit_transform(movies_df['soup'])

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

"""Create a reverse mapping of movie titles to indices.
By this, we can easily find the title of the movie based on the index."""
indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()
movies_df = movies_df.reset_index()


def get_recommendations(title, df=movies_df, indicesSer=indices, cosine_sim=cosine_sim2):
    idx = indicesSer[title]
    similarity_score = list(enumerate(cosine_sim[idx]))
    similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    similarity_score = similarity_score[1:11]

    movies_indices = [ind[0] for ind in similarity_score]
    movies = df[["title", "release_date"]].iloc[movies_indices]
    movies["release_date"] = pd.to_datetime(movies['release_date'])
    return movies


print("Content Based System".center(80, "#"))
movieTitles = ["Up", "Shrek"]
for movie in movieTitles:
    print(f'Recommendations for {movie}')
    print(get_recommendations(movie, movies_df, indices, cosine_sim2), "\n")
