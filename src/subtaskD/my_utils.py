from __future__ import division
from sklearn.feature_extraction.text import CountVectorizer
from twitterTokenizer import Tokenizer
import re, codecs, sys, subprocess, scipy, numpy as np, os, tempfile, math
from scipy.sparse import csr_matrix
from twitterTokenizer import Tokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter, OrderedDict
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
import string
punct = string.punctuation
negation = set(["never","no","nothing","nowhere","noone","none","not","havent","haven't","hasnt","hasn't","hadnt","hadn't", 
                "cant","can't","couldnt","couldn't","shouldnt","shouldn't","wont","won't","wouldnt","wouldn't","dont","don't","doesnt","doesn't","didnt",
                "didn't","isnt","isn't","arent","aren't","aint","ain't"])


clusters = codecs.open('../twitterClusters.txt', 'r', encoding='utf8').read().splitlines()
clusters = [i.split("\t") for i in clusters]
voca_clusters = {i[1]:i[0] for i in clusters}
clusters = list(set([i[0] for i in clusters]))

#f = codecs.open('../my_clusters_h.txt', 'r', encoding='utf8').read().splitlines()
#voca_handmade = {i.split('\t')[0]:i.split('\t')[1] for i in f}


def createDataMatrixEmbeddings(tweetText,categories):
    em1 = embeddings(tweetText, '../../tweetsBatch1/embedding-results/sswe-h.txt')
    em2 = embeddings(tweetText, '../../tweetsBatch1/embedding-results/sswe-r.txt')
    em3 = embeddings(tweetText, '../../tweetsBatch1/embedding-results/sswe-u.txt')
    y=[]
    for i in categories:
        if i=='positive':
            y.append(1)
        elif i == 'negative':
            y.append(-1)
        else:
            print "Problem"
    return normalize(np.hstack((em1,em2,em3))), y



def createDataMatrix(ngram_features, character_gram_features,tweetText, pos, pos_features, different_pos_tags, pos_text, voca_clusters, categories):
    tokenizer_case_preserve = Tokenizer(preserve_case=True)
    tokenizer = Tokenizer(preserve_case=False)
    handmade_features, cll, cll2 = [], [], []
    for tweet in tweetText:
        feat = []
        feat.append(exclamations(tweet))
        feat.append(questions(tweet))
        feat.append(questions_and_exclamation(tweet))
        feat.append(emoticon_negative(tweet))
        feat.append(emoticon_positive(tweet))
        words = tokenizer_case_preserve.tokenize(tweet) #preserving casing
        feat.append(allCaps(words))
        feat.append(elongated(words))
        feat.append(questions_and_exclamation(words[-1]))
        handmade_features.append(np.array(feat))
        words = tokenizer.tokenize(tweet)
        words = [word.strip("_NEG") for word in words]
        cll.append(getClusters(voca_clusters, words))
        #cll2.append(getClusters(voca_handmade, words))


    bl = csr_matrix(bing_lius(tweetText, pos, different_pos_tags, pos_text))
    nrc_emo = csr_matrix(nrc_emotion(tweetText, pos, different_pos_tags, pos_text ))
    mpqa_feat = csr_matrix(mpqa(tweetText,pos, different_pos_tags, pos_text))
    handmade_features = np.array(handmade_features)
    mlb = MultiLabelBinarizer(sparse_output=True, classes = list(set(voca_clusters.values())))
    cluster_memberships_binarized = csr_matrix(mlb.fit_transform(cll))
    #mlb = MultiLabelBinarizer(sparse_output=True, classes = list(set(voca_handmade.values())))
    #cluster_memberships_binarized_2 = csr_matrix(mlb.fit_transform(cll2))
    
    hasht = csr_matrix(sent140aff(tweetText, pos, different_pos_tags, pos_text, '../lexicons/HashtagSentimentAffLexNegLex/HS-AFFLEX-NEGLEX-unigrams.txt'))
#    sent140aff_data = csr_matrix(sent140aff(tweetText, pos, different_pos_tags, pos_text, '../../lexicons/Sentiment140AffLexNegLex/S140-AFFLEX-NEGLEX-unigrams.txt'))
    hasht_bigrams=csr_matrix(sent140aff_bigrams(tweetText, pos, different_pos_tags, pos_text, '../lexicons/HashtagSentimentAffLexNegLex/HS-AFFLEX-NEGLEX-bigrams.txt'))
#    sent140affBigrams=csr_matrix(sent140aff_bigrams(tweetText, pos, different_pos_tags, pos_text, '../../lexicons/Sentiment140AffLexNegLex/S140-AFFLEX-NEGLEX-bigrams.txt'))
    sentQ = csr_matrix(get_sentiwordnet(pos_text, pos))
    pos_features = csr_matrix(pos_features)
    handmade_features = csr_matrix(handmade_features)
    # ffeatures = scipy.sparse.hstack((ngram_features, character_gram_features, cluster_memberships_binarized, handmade_features, pos_features, 
#                             sent140affBigrams, hasht_bigrams, hasht, sent140aff_data, bl, mpqa_feat, nrc_emo), dtype=float)
#    ffeatures = scipy.sparse.hstack((ngram_features, character_gram_features, cluster_memberships_binarized, handmade_features, pos_features, sent140affBigrams, hasht_bigrams, hasht, sent140aff_data, bl, mpqa_feat, nrc_emo), dtype=float)
    ffeatures = scipy.sparse.hstack((ngram_features, character_gram_features, sentQ, handmade_features, pos_features, cluster_memberships_binarized, bl, mpqa_feat, nrc_emo, hasht, hasht_bigrams ), dtype=float)

#     print ngram_features.shape, character_gram_features.shape, cluster_memberships_binarized.shape, handmade_features.shape, pos_features.shape, 
#     sent140affBigrams.shape, hasht_bigrams, hasht.shape, sent140aff_data.shape, bl.shape, mpqa_feat.shape, nrc_emo.shape
    y=[]
    for i in categories:
        if i=='positive':
            y.append(1)
        elif i == 'negative':
            y.append(-1)
        elif i == 'UNKNOWN':
            y.append(0)
        else:
            print i
    ffeatures = normalize(ffeatures)
#     ffeatures, y = shuffle(ffeatures,y)
    return ffeatures, y

def embeddings(tweetText, path2voca):
    f = codecs.open(path2voca).read().splitlines()
    dico = {i.split()[0]:np.array([float(x) for x in i.split()[1:]]) for i in f}
    tokenizer = Tokenizer(preserve_case=False)
    feat = []
    for key, tweet in enumerate(tweetText):
        words = tokenizer.tokenize(tweet)
        my_vec,  cnt,  min_max  = np.zeros(50),  0, [] 
        for i in words:
            j = i.strip("_neg")
            try:
                my_vec += dico[j]
                cnt += 1
                min_max.append(dico[j])
            except:
                pass
        if len(min_max)>1:
            min_max= np.array(min_max)
            my_min = np.amin(min_max, axis=0)
            my_max = np.amax(min_max, axis=0)
        else:
            my_min,my_max = np.zeros(50), np.zeros(50)
        if cnt > 1:
            my_vec /= cnt
        feat.append(np.hstack((my_vec, my_max, my_min)))
    return np.array(feat)
            


def nrc_emotion(tweetText, pos, different_pos_tags, pos_text ):
    with codecs.open('../lexicons/NRC-Emotion-Lexicon-v0.92/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt', 'r') as inFile:
        wds = inFile.read().splitlines()
    positive, negative = [], []
    for i in wds:
        my_i = i.split('\t')
        if my_i[1] == 'positive' and my_i[2]=='1':
            positive.append(my_i[0])
        if my_i[1] == 'negative' and my_i[2]=='1':
            negative.append(my_i[0])
    feat = []
    positive, negative = set(positive), set(negative)
#    for key, tweet in enumerate(pos_text):
    tokenizer = Tokenizer(preserve_case=True)
    for key, tweet in enumerate(tweetText):
        words = tokenizer.tokenize(tweet)
        counters, counters_caps = np.zeros(8), np.zeros(8)
        for i in words:
            if i.isupper():
                counters_caps += np.array(getBingLiusCounters(positive, negative, i.lower()))
            else:
                counters += np.array(getBingLiusCounters(positive, negative, i.lower()))
        pos_sen = OrderedDict({x:[0,0,0,0] for x in different_pos_tags})
        for k_key, k in enumerate(pos_text[key]):
            if k in positive:
                pos_sen[pos[key][k_key]][0]+=1
            if k in negative:
                pos_sen[pos[key][k_key]][2]+=1
            if k.endswith("_NEG"):
                if k.strip("_NEG") in positive:
                    pos_sen[pos[key][k_key]][1]+=1
                if k.strip("_NEG") in negative:
                    pos_sen[pos[key][k_key]][3]+=1
#        my_feat = list(counters)+list(counters_caps)+[g for gg in pos_sen.values() for g in gg]
        my_feat = list(counters+counters_caps)+[g for gg in pos_sen.values() for g in gg]
        feat.append(np.array(my_feat))
    return np.array(feat)


def sent140aff_bigrams(tweetText, pos, different_pos_tags, pos_text, path2lexicon):
    with codecs.open(path2lexicon, 'r') as inFile:
        wds = inFile.read().splitlines()
    lexicon = {}
    for i in wds:
        i = i.split("\t")
        lexicon[i[0]]=float(i[1])
    feat = []
    for key,tweet in enumerate(pos_text):
        scor = []
        bigrams = zip(tweet, tweet[1:])
        for pair in bigrams:
            look = " ".join(pair)
            if look in lexicon:
                scor.append(lexicon[look])
        if len(scor)> 0:
            pos_scores, neg_scores = [x for x in scor if x>0],[x for x in scor if x<0]
            if len(pos_scores) == 0:
                pos_scores= [0]
            if len(neg_scores) == 0:
                neg_scores=[0]
            feat.append([len(scor), len(pos_scores), len(neg_scores), sum(scor), sum(pos_scores), sum(neg_scores), max(scor),
                        max(pos_scores), max(neg_scores), scor[-1], pos_scores[-1], neg_scores[-1]])
        else:
            feat.append(list(np.zeros(12)))
    return np.array(feat)

def sent140aff(tweetText, pos, different_pos_tags, pos_text, path2lexicon):
    with codecs.open(path2lexicon, 'r') as inFile:
        wds = inFile.read().splitlines()
    pos_cont, nega_cont, nega_cont_first = {},{},{}
    for i in wds:
        i = i.split("\t")
        if i[0].endswith("_NEG"):
            name = "".join(i[0].split('_')[:-1])
            nega_cont[name]=float(i[1])
        elif i[0].endswith('_NEGFIRST'):
            name = "".join(i[0].split('_')[:-1])
            nega_cont_first[name]=float(i[1])
        else:
            pos_cont[i[0]]=float(i[1])
    feat = []
    tokenizer = Tokenizer(preserve_case=False)
    for key, tweet in enumerate(tweetText):
        cnt, scor  = 0, []
        words = tokenizer.tokenize(tweet)
        for my_key, i in enumerate(words):
            if i in pos_cont:
                scor.append(pos_cont[i])
            if i.endswith('_neg'):
                j = i.strip("_neg")
                flag = 0
                if not words[my_key-1].endswith('_neg'):
                    if j in nega_cont_first:
                        scor.append(nega_cont_first[j])
                        flag = 1
                    elif j in nega_cont:
                        scor.append(nega_cont[j])
                        flag = 1 
                    else:
                        pass
                if j in nega_cont and flag == 0:
                    scor.append(nega_cont[j])
        if len(scor)> 0:
            pos_scores, neg_scores = [x for x in scor if x>0],[x for x in scor if x<0]
            if len(pos_scores) == 0:
                pos_scores= [0]
            if len(neg_scores) == 0:
                neg_scores=[0]
            feat.append([len(scor), len(pos_scores), len(neg_scores), sum(scor), sum(pos_scores), sum(neg_scores), max(scor), 
                        max(pos_scores), max(neg_scores), scor[-1], pos_scores[-1], neg_scores[-1]])
        else:
            feat.append(list(np.zeros(12)))
    return np.array(feat)



def getBingLiusCounters(positive, negative, i):
    pp, pn, npp, nn, pp_hash, pn_hash, npp_hash, nn_hash = 0,0,0,0,0,0,0,0
    if i in positive:
        pp+=1
    if i in negative:
        npp+=1
    if i.endswith("_neg"):
        if i.strip("_neg") in positive:
            pn+=1
        if i.strip("_neg") in negative:
            nn+=1
    if i[0] == "#":
        if i[1:] in positive:
            pp_hash+=1
        if i[1:] in negative:
            npp_hash+=1
        if i.endswith("_neg"):
            if i[1:].strip("_neg") in positive:
                pn_hash+=1
            if i[1:].strip("_neg") in negative:
                nn_hash+=1
    return pp, pn, npp, nn, pp_hash, pn_hash, npp_hash, nn_hash



def bing_lius(tweetText, pos, different_pos_tags, pos_text ):
    with codecs.open('../lexicons/positive-words_bing_liu.txt', 'r') as inFile:
        positive = set(inFile.read().splitlines())
    with codecs.open('../lexicons/negative-words_bing_liu.txt', 'r') as inFile:
        negative = set(inFile.read().splitlines())
    feat = []
    tokenizer = Tokenizer(preserve_case=True)
    for key, tweet in enumerate(tweetText):
        words = tokenizer.tokenize(tweet)
        counters, counters_cap = np.zeros(8), np.zeros(8)
        for j in words:
            if j.isupper():
                counters_cap += np.array(getBingLiusCounters(positive, negative, j.lower()))
            else:
                counters += np.array(getBingLiusCounters(positive, negative, j.lower()))
        pos_sen = OrderedDict({x:[0,0,0,0] for x in different_pos_tags})
        for k_key, k in enumerate(pos_text[key]):
            if k in positive:
                pos_sen[pos[key][k_key]][0]+=1
            if k in negative:
                pos_sen[pos[key][k_key]][2]+=1
            if k.endswith("_NEG"):
                if k.strip("_NEG") in positive:
                    pos_sen[pos[key][k_key]][1]+=1
                if k.strip("_NEG") in negative:
                    pos_sen[pos[key][k_key]][3]+=1
#        my_feat = list(counters)+list(counters_cap)+[g for gg in pos_sen.values() for g in gg]
        my_feat = list(counters+counters_cap)+[g for gg in pos_sen.values() for g in gg]
        feat.append(np.array(my_feat))
    return np.array(feat)


def mpqa(tweetText, pos, different_pos_tags, pos_text):
    voca = codecs.open('../lexicons/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff', 'r').read().splitlines()
    wds1, wds = {}, {}
    for i in voca:
        i = i.split()
        try:
            if wds1[i[2].split('=')[1]] != i[5].split('=')[1]:
               pass 
        except:
            if i[5].split('=')[1] in ['positive', 'negative']:
                wds1[i[2].split('=')[1]] = i[5].split('=')[1]
                wds[i[2].split('=')[1]]=(i[0].split('=')[1], i[5].split('=')[1])
    feat = []
    tokenizer = Tokenizer(preserve_case=False)
    for key, tweet in enumerate(tweetText):
        direction = {'negative':-1, 'positive':1, 'neutral':0, 'both':0, 'weaksubj':1, 'strongsubj':2}
        pp, pn, npp, nn, pp_hash, pn_hash, npp_hash, nn_hash  = 0,0,0,0,0,0,0,0
        words = tokenizer.tokenize(tweet)
        for i in words:
            if i in wds:
                if direction[wds[i][1]] > 0:
                    pp += direction[wds[i][0]]*direction[wds[i][1]]
                if direction[wds[i][1]] < 0:
                    pn += direction[wds[i][0]]*direction[wds[i][1]]
            if i.endswith("_neg"):
                my_i = i.strip("_neg")
                if my_i in wds:
                    if direction[wds[my_i][1]] > 0:
                        npp += direction[wds[my_i][0]]*direction[wds[my_i][1]]
                    if direction[wds[my_i][1]] < 0:
                        nn += direction[wds[my_i][0]]*direction[wds[my_i][1]]
            if i[0] == "#":
                if i[1:] in wds:
                    if direction[wds[i[1:]][1]] > 0:
                        pp_hash += direction[wds[i[1:]][0]]*direction[wds[i[1:]][1]]
                    if direction[wds[i[1:]][1]] < 0:
                        pn_hash += direction[wds[i[1:]][0]]*direction[wds[i[1:]][1]]
                if i.endswith("_neg"):
                    my_i = i[1:].strip("_neg")
                    if my_i in wds:
                        if direction[wds[my_i][1]] > 0:
                            npp_hash += direction[wds[my_i][0]]*direction[wds[my_i][1]]
                        if direction[wds[my_i][1]] < 0:
                            nn_hash += direction[wds[my_i][0]]*direction[wds[my_i][1]]
        pos_sen = OrderedDict({x:[0,0,0,0] for x in different_pos_tags})
        for k_key, i in enumerate(pos_text[key]):
            if i in wds:
                if direction[wds[i][1]] > 0:
                    pos_sen[pos[key][k_key]][0]+=1
                if direction[wds[i][1]] < 0:
                    pos_sen[pos[key][k_key]][1]+=1
            if i.endswith("_NEG"):
                if i.strip('_NEG') in wds:
                    ii = i.strip('_NEG')
                    if direction[wds[ii][1]] > 0:
                        pos_sen[pos[key][k_key]][2]+=1
                    if direction[wds[ii][1]] < 0:
                        pos_sen[pos[key][k_key]][3]+=1
        my_feat = [pp, pn, npp, nn, pp_hash, pn_hash, npp_hash, nn_hash]+[g for gg in pos_sen.values() for g in gg]
        feat.append(np.array(my_feat))
    return np.array(feat)


def change_userMention_url(tweet):
    user_match = r"(?:@[\w_]+)"
    url_match = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    tweet = re.sub(user_match, r"@someuser", tweet)
    tweet = re.sub(url_match, r"@someurl", tweet)
    return tweet

def get_sentiwordnet(pos_text, pos):
    import nltk
    from nltk.corpus import wordnet as wn
    from nltk.corpus import sentiwordnet as swn
    feat = []
    for key, val in enumerate(pos):
        pos, neg, pos_neg, neg_neg, POS_pos, POS_neg = 0,0,0,0, {'A':0, 'V':0, 'R':0, 'N':0}, {'A':0, 'V':0, 'R':0, 'N':0}
        for key3, val3 in enumerate(val):
            if val3 in 'AVRN':
                text = pos_text[key][key3].strip("_NEG")
                synsets = wn.synsets('%s'%text.decode('utf-8'),val3.lower())
                try:
                    sense=synsets[0]
                except:
                    continue
                k = swn.senti_synset(sense.name())
                if k != None:
                    if pos_text[key][key3].endswith("_NEG"):
                        pos_neg += k.pos_score()
                        neg_neg += k.neg_score()
                        POS_neg[val3]+=1
                    else:
                        pos += k.pos_score()
                        neg += k.neg_score()
                        POS_pos[val3]+=1
        feat.append([pos, neg, pos_neg, neg_neg, pos+neg+pos_neg+neg_neg, sum(POS_pos.values())+sum(POS_neg.values())]+POS_pos.values()+POS_neg.values())
    return np.array(feat)

def get_pos_tags_and_hashtags(tweetText):
    tf = tempfile.NamedTemporaryFile(delete=False)
    with codecs.open(tf.name, 'w', encoding='utf8') as out:
        for i in tweetText:
            out.write("%s\n"%i.decode('utf-8'))
    com = "../ark-tweet-nlp-0.3.2/runTagger.sh %s"%tf.name
    op= subprocess.check_output(com.split())
    op = op.splitlines()
    pos_text = [x.split("\t")[0].split() for x in op]
    pos = [x.split("\t")[1].split() for x in op]
    different_pos_tags = list(set([x for i in pos for x in i]))
    pos_features = []
    for instance in pos:
        tags = []
        instance = Counter(instance)
        for pos_tag in different_pos_tags:
            try:
                tags.append(instance[pos_tag])
            except:
                tags.append(0)
        pos_features.append(np.array(tags))
    pos_features = np.array(pos_features)
    #print "------------\nPOS-tagging finished!\n------------\nThere are %d pos-tags (incl. hashtags). Shape: %d,%d"%(len(different_pos_tags), pos_features.shape[0],  pos_features.shape[1])
    for key1, i in enumerate(pos_text):
        flag = False
        for key, j in enumerate(i):
            i[key] = j.lower()
            if flag:
                if pos[key1][key] in "AVRN" :
                    i[key]+="_NEG"
                else:
                    flag=False
            if j in negation:
                flag = True
    os.remove(tf.name)
    return pos, pos_features, different_pos_tags, pos_text


def getClusters(voca_clusters, words):
    "Input: list of words, Output: the clusters where words of the tweets are present"
    c = []
    for word in words:
        try:
            c.append(voca_clusters[word])
        except:
            pass
    return set(c) # For each word in a tweet, populate the cluster number of words


def allCaps(words):
    """ Input: list of words, Output: how many are all caps """
    return len([word for word in words if word.isupper()])

def elongated(words):
    """ Input: list of words, Output: how many are elongated """
    return len([word for word in words if re.search(r"(.)\1{2}", word.lower())])

def exclamations(tweet):
    """ Input: a tweet, Output: how many exclamations """
    return len(re.findall("!+", tweet))

def questions(tweet):
    """ Input: a tweet, Output: how many question marks """
    return len(re.findall("\?+", tweet))

def questions_and_exclamation(tweet):
    """ Input: a tweet, Output: how many question marks and exclamation marks"""
    return len(re.findall("[\?,!]+", tweet))


def emoticon_positive(tweet):
    """ Input: a tweet, Output: (binary) positive emoticons exist """
    emoticon_string = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\)\]dD\}@]                # mouth      
      |                          # reverse order now! 
      [\)\]dD\}@]                # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
    )"""
    emoticon_re = re.compile(emoticon_string, re.VERBOSE | re.I | re.UNICODE)
    return len(emoticon_re.findall(tweet)) > 0
    
    
def emoticon_negative(tweet):
    """ Input: a tweet, Output: (binary) negative emoticons exist """
    emoticon_string = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\(\[pP/\:\{\|] # mouth      
      |                          # reverse order now! 
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
    )"""
    emoticon_re = re.compile(emoticon_string, re.VERBOSE | re.I | re.UNICODE)
    return len(emoticon_re.findall(tweet)) > 0




def macroMAE(y_true, y_predicted, classes=[-2, -1, 0, 1, 2]):
    y_true, y_predicted = np.array(y_true), np.array(y_predicted)
    indexes = {i:np.where(y_true==i)[0] for i in classes}
    error = 0
    for i in classes:
        diffs = np.subtract(y_true[indexes[i]], y_predicted[indexes[i]])
        diffs = np.absolute(diffs)
        macro_error = np.sum(diffs)/float(len(diffs))
        error += macro_error
    return error/float(len(classes))  


def my_score(estimator, X, true):
    predicted = list(estimator.predict(X))
    nom_pos, denom_pos, nom_neg, denom_neg  = 0, 0, 0, 0
    #precision
    for key, val in enumerate(true):
        if val == 1:
            denom_neg += 1
            if predicted[key] == 1:
                nom_neg += 1
        elif val == 0:
            denom_pos += 1
            if predicted[key] == 0:
                nom_pos += 1
        else:
            pass
    pr = Counter(predicted)
    prec_pos = nom_pos/ float(denom_pos+1)
    prec_neg = nom_neg/ float(denom_neg+1)
    recall_pos=nom_pos/float(pr[0]+1)
    recall_neg=nom_neg/float(pr[1]+1)
    try:
        f1_neg = 2*prec_neg*recall_neg/(prec_neg+recall_neg)
    except:
        f1_neg = 0
    try:
        f1_pos = 2*prec_pos*recall_pos/(prec_pos+recall_pos)
    except:
        f1_pos = 0
#     print "F1 of positive class: %f. - F1 of negative class: %f\n"%(f1_pos, f1_neg)
    return (f1_pos + f1_neg)/float(2)


def KLD(true, pred):
    epsilon = 0.5 / len(pred)
    countsTrue, countsPred = Counter(true), Counter(pred)
    p_pos = countsTrue[0]/len(true)
    p_neg = countsTrue[1]/len(true)
    est_pos = countsPred[0]/len(true)
    est_neg = countsPred[1]/len(true)
    p_pos_s = (p_pos + epsilon)/(p_pos+p_neg+2*epsilon)
    p_neg_s = (p_neg + epsilon)/(p_pos+p_neg+2*epsilon)
    est_pos_s = (est_pos+epsilon)/(est_pos+est_neg+2*epsilon)
    est_neg_s = (est_neg+epsilon)/(est_pos+est_neg+2*epsilon)
    return p_pos_s*math.log10(p_pos_s/est_pos_s)+p_neg_s*math.log10(p_neg_s/est_neg_s)


def showMyKLD(true, pred, l):
    s= []
    for key, val in enumerate(l):
        if key == len(l)-1:
            break
        s.append(KLD(true[val:l[key+1]], pred[val:l[key+1]]))
    return sum(s)/len(s)

def generateOutput4Submission(preds, l):
    s = []
    for key, val in enumerate(l):
        if key == len(l)-1:
            break
        my_preds = preds[val:l[key+1]]
        positive = [i for i in my_preds if i == 1]
        negative = [i for i in my_preds if i == -1]
        s.append((len(positive)/float(len(my_preds)), len(negative)/float(len(my_preds))))
    return s
    





