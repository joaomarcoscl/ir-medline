import nltk
import string
import numpy as np
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

stemmer = PorterStemmer()
query = 'the crystalline lens in vertebrates, including humans'
text_trans = []

def retrieval(terms,matrix_dt):
    result_docs = []
    for term in terms:
        result_docs = result_docs + np.where(matrix_dt[:,term]>0)[0].tolist()[0]
    return set(result_docs)

def tokenize_stopwords_stemmer(text, stemmer):
    no_punctuation = text.translate(None, string.punctuation)
    tokens = nltk.word_tokenize(no_punctuation)
    text_filter = [w for w in tokens if not w in stopwords.words('english')]
    text_final = ''
    for k in range(0, len(text_filter)):
        text_final +=str(stemmer.stem(text_filter[k]))
        if k != len(text_filter)-1:
            text_final+=" "
            pass
    return text_final

def organizes_documents():
	files = open('med/MED.ALL', 'r').read().split('.I')
	for i in range(0,len(files)):
		text = files[i].replace('.W', '')
		text = text.replace(str(i), '')
		text_trans.append(tokenize_stopwords_stemmer(text.lower(), stemmer))
	generate_matrix()

def save_object(obj, filename):
    with open('objects/'+filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)

def generate_matrix():
    document_term = CountVectorizer()
    matrix_document_term = document_term.fit_transform(text_trans)
    save_object(document_term.get_feature_names(), 'terms.dt')
    matrix_dt = np.matrix(matrix_document_term.toarray())
    save_object(matrix_dt, 'matrix.dt')
    matrix_tt = np.dot(np.transpose(matrix_dt), matrix_dt)
    save_object(matrix_tt, 'matrix.tt')
    pass

def search_expanded(query, terms_dt, matrix_tt):
    terms = []
    for i in query:
        if i in terms_dt:
            key = terms_dt.index(i)
            terms_recommended = np.sort(matrix_tt[key])[:, len(matrix_tt)-5:len(matrix_tt)]
            for j in terms_recommended.tolist()[0]:
                terms.append(matrix_tt[key, :].tolist()[0].index(j))
                pass
        pass
    pass
    return terms

matrix_dt = load_object('objects/matrix.dt')
matrix_tt = load_object('objects/matrix.tt')
terms_dt = load_object('objects/terms.dt')
query_token = tokenize_stopwords_stemmer(query, stemmer)
terms = search_expanded(query_token.split(' '), terms_dt, matrix_tt)
print retrieval(terms, matrix_dt)