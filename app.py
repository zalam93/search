#!flask/bin/python
from flask import Flask
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity



app = Flask(__name__)
@app.route('/<Query>')
def index(Query):
	data = pd.read_csv('employee_reviews.csv')
	text = data['pros']
	stop_words = stopwords.words('english')

	def process_text(text):
		text = re.sub('[^a-z\s]', '', text.lower())
		text = [w for w in text.split() if w not in set(stop_words)]
		return ' '.join(text)
		
	data['pros'] = data['pros'].apply(process_text)

	english_stemmer = SnowballStemmer('english')
	analyzer = CountVectorizer().build_analyzer()

	def stemming(text):
		return (english_stemmer.stem(w) for w in analyzer(text))

	count = CountVectorizer(analyzer = stemming)

	count_matrix = count.fit_transform(data['pros'])

	tfidf_transformer = TfidfTransformer()
	train_tfidf = tfidf_transformer.fit_transform(count_matrix)

	def get_search_results(query):
		query = process_text(query)
		query_matrix = count.transform([query])
		query_tfidf = tfidf_transformer.transform(query_matrix)
		sim_score = cosine_similarity(query_tfidf, train_tfidf)
		sorted_indexes = np.argsort(sim_score).tolist()
		return data['company'].iloc[sorted_indexes[0][-10:]]


	working = get_search_results(Query)
	working = set(working)
	return ' '.join(working)

if __name__ == '__main__':
    app.run(debug=True)
	
	
	


