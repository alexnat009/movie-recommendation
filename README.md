# Movie Recommendation System - Content Based

## Overview

This repository contains a content-based movie recommendation system. The system uses features such as **cast**, **director**, **keywords**, and **genres** to calculate the similarity between movies and recommend similar ones. The project uses **pandas** for data manipulation, **scikit-learn** for text vectorization and similarity calculation, and **numpy** for numerical operations.

### Features:
- Extracts director, cast, keywords, and genres to create a comprehensive representation of movies.
- Cleans and processes data to handle various formats (strings, lists).
- Uses **cosine similarity** to find similar movies based on combined features.
- Generates movie recommendations based on similarity scores.


## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn


## Dataset

The project uses two datasets:
- `tmdb_5000_credits.csv`: Contains movie credits (id, title, cast, and crew).
- `tmdb_5000_movies.csv`: Contains metadata about the movies (id, title, genres, keywords, release date, etc.).

These datasets can be downloaded from [TMDB 5000 Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-5000-movie-dataset).


## Usage

To run the recommendation system and test it with different movie titles, execute `main.py`:


### Sample Output:

```
##############################Content Based System##############################
Recommendations for Up
                                                  title release_date
3403  Alpha and Omega: The Legend of the Saw Tooth Cave   2014-07-21
42                                          Toy Story 3   2010-06-16
2899                                 Legend of a Rabbit   2011-01-01
4286                                  The Lion of Judah   2011-06-03
77                                           Inside Out   2015-06-09
57                                               WALL·E   2008-06-22
459                    Spirit: Stallion of the Cimarron   2002-05-24
234                                          The Croods   2013-03-20
1695                                            Aladdin   1992-11-25
2114                               Return to Never Land   2002-02-14 
```
## Improvements

- **Model Expansion**: You could extend the system by using more advanced algorithms, such as collaborative filtering or hybrid models, for more personalized recommendations.
- **Scalability**: For large datasets, consider using **TF-IDF** (Term Frequency-Inverse Document Frequency) or other vectorization techniques instead of simple count vectors.
  

