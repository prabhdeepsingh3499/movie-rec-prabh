import flask
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from difflib import get_close_matches


app = flask.Flask(__name__, template_folder= 'templates')

result = pd.read_csv('./model/tmdb.csv')

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(result['soup'])

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