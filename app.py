import flask
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from difflib import get_close_matches


app = flask.Flask(__name__, template_folder= 'templates')

result = pd.read_csv('./model/tmdb.csv')
result_songs = pd.read_csv('./model/songs.csv')

count = CountVectorizer(stop_words='english')
count1 = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(result['soup'])
count_matrix_songs = count1.fit_transform(result_songs['soup'])

cosine_sim = cosine_similarity(count_matrix,count_matrix)
cosine_sim_songs = cosine_similarity(count_matrix_songs,count_matrix_songs)

result = result.reset_index()
indices = pd.Series(result.index, index=result['title'])
all_titles = [result['title'][i] for i in range(len(result['title']))]

result_songs = result_songs.reset_index()
indices_songs = pd.Series(result_songs.index, index=result_songs['title'])
all_titles_songs = [result_songs['title'][i] for i in range(len(result_songs['title']))]

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

def get_recommendations_songs(title, cos = cosine_sim_songs):
    idx_songs = indices_songs[title]
    cos_scores_songs = list(enumerate(cos[idx_songs]))
    cos_scores_songs = sorted(cos_scores_songs, key=lambda x: x[1], reverse=True)
    cos_scores_songs = cos_scores_songs[1:11]  
    song_indices = [i[0] for i in cos_scores_songs]
    tit_songs = result_songs['title'].iloc[song_indices]
    return_df_songs = pd.DataFrame(columns = ['title'])
    return_df_songs['title'] = tit_songs
    return return_df_songs

def close_matches(patterns,word):
    return get_close_matches(word,patterns,cutoff=0.4)


@app.route('/', methods = ['GET','POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))
    if flask.request.method == 'POST':
        m_name = flask.request.form['movie_name']
        m_name = m_name.lower()
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



@app.route('/song-recommender', methods = ['GET','POST'])
def main_song():
    if flask.request.method == 'GET':
        return(flask.render_template('index_songs.html'))
    if flask.request.method == 'POST':
        print("1")
        s_name = flask.request.form['song_name']
        s_name = s_name.lower()
        print("2")
        if s_name not in all_titles_songs:
            print("3")
            als = close_matches(result_songs['title'],s_name)
            alternative_names = []
            for i in als:
                alternative_names.append(i)
            return (flask.render_template('negative_songs.html',name = s_name,alternative_songs = alternative_names))
        else:
            result_final_songs = get_recommendations_songs(s_name)
            names_songs = []
            for i in range(len(result_final_songs)):
                names_songs.append(result_final_songs.iloc[i][0])
            return flask.render_template('positive_songs.html',songs_names = names_songs,search_name = s_name)

if __name__ == '__main__':
    app.run()