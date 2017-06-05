import nltk
import string
import numpy as np
import pickle
import operator
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

stemmer = PorterStemmer()
querys = []
expansion = 0
text_trans = []

def retrieval(terms,matrix_dt, terms_dt, query):
    result_docs = []
    for term in terms:
        #result_docs = result_docs + np.where(matrix_dt[:,term]>0)[0].tolist()
        sum_vector = np.sum(matrix_dt[:,term])
        norm = dict()
        for i in np.where(matrix_dt[:,term]>1)[0].tolist():
            norm[i] = float(matrix_dt[i, term])/float(sum_vector)
        norm_sort = sorted(norm.items(), key=operator.itemgetter(1),reverse=True)
        sum_norm_sort = 0
        for i in norm_sort:
            sum_norm_sort = sum_norm_sort + i[1]
            result_docs.append(i[0])
            if sum_norm_sort >= 0.5:#float(len(query))/float(len(terms)):
                break
            pass
    return set(result_docs)

def retrieval2(terms,matrix_dt,terms_dt, query):
    result_docs = []
    for term in terms:
        result_docs = result_docs + np.where(matrix_dt[:,term]>0)[0].tolist()
    return set(result_docs)

def tokenize_stopwords_stemmer(text, stemmer, query):
    no_punctuation = text.translate(None, string.punctuation)
    tokens = nltk.word_tokenize(no_punctuation)
    text_filter = [w for w in tokens if not w in stopwords.words('english')]
    text_final = ''
    
    if query == True:
        for k in range(0, len(text_filter)):
            for i in wn.synsets(text_filter[k]):
                for s in i.lemma_names():
                    text_filter.append(s)

    for k in range(0, len(text_filter)):
        text_final +=str(stemmer.stem(text_filter[k]))
        if k != len(text_filter)-1:
            text_final+=" "
            pass
    return text_final

def organizes_querys():
    files = open('med/MED.QRY', 'r').read().split('.I')
    for i in range(0,len(files)):
        text = files[i].replace('.W', '')
        text = text.replace(str(i), '')
        if len(text) > 0:
            querys.append(text)

def organizes_documents():
    files = open('med/MED.ALL', 'r').read().split('.I')
    for i in range(0,len(files)):
        text = files[i].replace('.W', '')
        text = text.replace(str(i), '')
        text_trans.append(tokenize_stopwords_stemmer(text.lower(), stemmer, False))
    generate_matrix()

def save_object(obj, filename):
    with open('objects/'+filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)

def generate_matrix():
    document_term = CountVectorizer()
    #document_term = TfidfVectorizer()
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
        if i in terms_dt:#se botar nada a ver
            key = terms_dt.index(i)
            terms_recommended = np.sort(matrix_tt[key])[:, len(matrix_tt)-expansion:len(matrix_tt)]
            for j in terms_recommended.tolist()[0]:
                terms
                terms.append(matrix_tt[key, :].tolist()[0].index(j))
            pass
            if key in terms == False or expansion == 0:
                terms.append(key)
        pass
    pass
    return set(terms)

def relevants_documents():
    relevants_resume = dict()
    relevants = open('med/MED.REL', 'r').readlines()
    for i in relevants:
        line = np.array(i.split(' ')).tolist()
        key = int(line[0])
        if key in relevants_resume:
            relevants_resume[key].append(int(line[2]))
        else:
            relevants_resume[key] = [int(line[2])]
        pass
    pass
    return relevants_resume

def main():
    #organizes_documents()
    organizes_querys()
    matrix_dt = load_object('objects/matrix.dt')
    matrix_tt = load_object('objects/matrix.tt')
    terms_dt = load_object('objects/terms.dt')

    amount_documents = len(matrix_dt)
    mean_precision = 0
    mean_recall = 0
    mean_acuracy = 0
    for i in xrange(0, len(querys)):
        query_token = tokenize_stopwords_stemmer(querys[i], stemmer, True)
        terms = search_expanded(set(query_token.split(' ')), terms_dt, matrix_tt)
        documents_retrieval = retrieval(terms, matrix_dt, terms_dt, query_token.split(' '))
        documents_relevants = relevants_documents()[i+1]
        
        precision = float(len(documents_retrieval.intersection(documents_relevants))) /  float(len(documents_retrieval))
        recall = float(len(documents_retrieval.intersection(documents_relevants))) / float(len(documents_relevants))
        acuracy = float(len(documents_retrieval.intersection(documents_relevants)) + amount_documents - len(documents_retrieval))/float(amount_documents)
        
        print len(documents_retrieval)
        print len(documents_relevants)
        print len(documents_retrieval.intersection(documents_relevants))
        print '#############'
        
        mean_precision = mean_precision  + precision
        mean_recall = mean_recall + recall
        mean_acuracy = mean_acuracy + acuracy
        pass
    
    pass
    print "Precision: " + str(round(mean_precision/len(querys),2))
    print "Recall: " + str(round(mean_recall/len(querys),2))
    print "Acuracy: " + str(round(mean_acuracy/len(querys),2))
main()