import flask
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
import json
from difflib import get_close_matches


app = flask.Flask(__name__, template_folder= 'templates')

df = pd.read_csv("model/tmdb_5000_movies.csv")
df1 = pd.read_csv("model/tmdb_5000_credits.csv")
df1 = df1.rename(columns = {"movie_id":"id"})

result = pd.merge(df,df1[['id','cast','crew']],on = 'id')


def json_to_list(col):
    for index, row in result.iterrows():
        row_list= []
        x = json.loads(row[col])
        for genre in x:
            row_list.append(genre['name'])
        result.at[index, col] = row_list

cols =['genres', 'keywords', 'production_companies', 'production_countries','spoken_languages','cast']
for col in cols:
    json_to_list(col)

def reduce_list(col):
    for index, row in result.iterrows():
        li = row[col]
        li = li[:5]
        result.at[index, col] = li

cols = ['genres','cast','keywords']
for col in cols:
    reduce_list(col)

result['title'] = result['title'].apply(lambda x:x.lower())

result = result[result['vote_average']>6]

for index,row in result.iterrows():
    crew_list = json.loads(row['crew'])
    for x in crew_list:
        if x['job'] == 'Director':
            result.at[index,'crew'] = x['name']

result = result[['genres','keywords','overview','popularity','revenue','title','vote_average','vote_count','cast','crew']]

result.rename(columns = {'crew':'director'},inplace = True)

 
result.dropna()

cols = ['genres','keywords','cast']
for col in cols:
    result = result[result[col].map(lambda d: len(d)) > 0]


result = result.drop('overview',axis=1)
result.drop_duplicates(subset = 'title', inplace = True, keep = False)

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x] 

features = ['cast', 'keywords', 'genres']
for feature in features:
    result[feature] = result[feature].apply(clean_data)

result['director'] = result['director'].str.replace(' ','')
result['director'] = result['director'].str.lower()

def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + ' '.join(x['genres'])
result['soup'] = result.apply(create_soup, axis=1)
result['soup'] = result[['soup','director']].apply(lambda x:' '.join(x), axis=1) 

count_matrix = joblib.load("count_model_movie.pkl")

cosine_sim = cosine_similarity(count_matrix,count_matrix)

result = result.reset_index()
indices = pd.Series(result.index, index=result['title'])
all_titles = [result['title'][i] for i in range(len(result['title']))]

def get_recommendations(title, cos=cosine_sim):
    idx = indices[title]
    cos_scores = list(enumerate(cos[idx]))
    cos_scores = sorted(cos_scores, key=lambda x: x[1], reverse=True)
    cos_scores = cos_scores[1:11]  
    movie_indices = [i[0] for i in cos_scores]
    tit = result['title'].iloc[movie_indices]
    return_df = pd.DataFrame(columns = ['title'])
    return_df['title'] = tit
    return return_df
def close_matches(patterns,word):
    return get_close_matches(word,patterns,cutoff=0.4)


@app.route('/', methods = ['GET','POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))
    if flask.request.method == 'POST':
        m_name = flask.request.form['movie_name']
        if m_name not in all_titles:
            als = close_matches(result['title'],m_name)
            alternative_names = []
            for i in als:
                alternative_names.append(i)
            
            return (flask.render_template('negative.html',name = m_name, alternative_movies = alternative_names))
        else:
            result_final = get_recommendations(m_name)
            names = []
            for i in range(len(result_final)):
                names.append(result_final.iloc[i][0])
            return flask.render_template('positive.html',movies_names = names,search_name = m_name)

if __name__ == '__main__':
    app.run()