from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from twitterTokenizer import Tokenizer
import re, codecs, sys, subprocess, scipy, numpy as np, json
from scipy.sparse import csr_matrix, hstack
from sklearn import svm, ensemble
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from my_utils import *

f = codecs.open('data/subtaskBD.downloaded.tsv').read().splitlines()
g = codecs.open('data/subtaskBD.dev.downloaded.all.tsv').read().splitlines()
f = [i.split("\t") for i in f if i.split("\t")[-1] != 'Not Available'] + [i.split("\t") for i in g if i.split("\t")[-1] != 'Not Available']#+ [i.split("\t") for i in m if i.split("\t")[-1] != 'Not Available']
tweetText, categories = [i[-1] for i in f], [i[2] for i in f]


#Load test data (development)
g = codecs.open('data/100_topics_XXX_tweets.topic-two-point.subtask-BD.devtest.gold.downloaded.txt').read().splitlines() #devtest data for tuning
#g = codecs.open('../SemEval2016-task4-test.subtask-BD.txt', encoding='utf8').read().splitlines() #Test data to generate final predictions
g = [i.split("\t") for i in g if i.split("\t")[-1] != 'Not Available']
tweetTest, categories_test = [i[-1] for i in g], [i[2] for i in g]


l = [i[1] for i in g] #This is to group tweets by topic. Can by improved!!
cnt = Counter(l)
yo = [0]
test_cats = []
for i in range(len(set(l))):
    num = cnt[l[yo[i]]]
    test_cats.append(l[num+yo[i]-1])
    yo.append(num+yo[i])


tokenizer = Tokenizer()
ngram = HashingVectorizer(strip_accents='unicode', binary=True, ngram_range=(1,4), stop_words=None, lowercase=True,  tokenizer=tokenizer.tokenize, n_features=10000) #N-gram feature vectorizer
character_gram = HashingVectorizer(strip_accents='unicode', binary=True, ngram_range=(4,5), stop_words=None, lowercase=True, analyzer='char', tokenizer=tokenizer.tokenize, n_features=22000) #Char-gram feature vectorizer

n_power = float(sys.argv[1]) #parameter of the n_power transformation, I used 0.9 for submission

#Linguistic, POS, sentiment disctionaries etc.
pos1, pos_features1, different_pos_tags1, pos_text1 = get_pos_tags_and_hashtags(tweetText+tweetTest) #Get POS of everything
pos, pos_features, different_pos_tags, pos_text =  pos1[:len(categories)], pos_features1[:len(categories)], different_pos_tags1, pos_text1[:len(categories)] #Split train-test again
pos_test, pos_features_test, different_pos_tags_test, pos_text_test = pos1[len(categories):], pos_features1[len(categories):], different_pos_tags1, pos_text1[len(categories):] #Split train-test again

ngram_features = ngram.fit_transform(tweetText) #Get n-gram features
character_gram_features = character_gram.fit_transform(tweetText) #Get char-gram features
ngram_features.data **= n_power #a-power transformation
character_gram_features.data **= n_power #a-power transformation

ngram_features_test = ngram.transform(tweetTest)
character_gram_features_test = character_gram.transform(tweetTest)
ngram_features_test.data **= n_power
character_gram_features_test.data **= n_power

x_train, y_train = createDataMatrix(ngram_features, character_gram_features, tweetText, pos, pos_features, different_pos_tags, pos_text, voca_clusters, categories) #Combine all  features (train)
x_test, y_test = createDataMatrix(ngram_features_test, character_gram_features_test, tweetTest, pos_test, pos_features_test, different_pos_tags_test, pos_text_test, voca_clusters, categories_test)# Combine feat test


print "SVMs crammer singer"
for c in np.logspace(-3,4,8): #used 100 for submission
    clf = svm.LinearSVC(C=c, loss='squared_hinge', penalty='l2', class_weight='balanced', multi_class='crammer_singer', max_iter=4000, dual=True, tol=1e-6)
    clf.fit(x_train, y_train)
    print "Hold-out",  showMyKLD(y_test, clf.predict(x_test), yo), c

