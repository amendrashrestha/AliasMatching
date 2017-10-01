__author__ = 'amendrashrestha'

import glob
import os
import re
from pathlib import Path

from nltk import FreqDist
from nltk import word_tokenize, pos_tag

import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

document_path = os.path.expanduser('~') + "/Downloads/PAN-15-Test/"
tfidf_filepath = os.path.expanduser('~') + "/repo/AliasMatching/dictionaries/TfIdf"
ngram_filepath = os.path.expanduser('~') + "/repo/AliasMatching/dictionaries/Ngram_char"


def get_document_filenames(document_path):
    files = [file for file in glob.glob(document_path + '/*/**/*.txt', recursive=True)]
    return files


def create_tfIdf(N):
    if not Path(tfidf_filepath).exists():
        print("Creating tfIdf file .... \n")
        vectorizer = TfidfVectorizer(input='filename', analyzer='word', ngram_range=(1, 3), min_df=2, max_df=5,
                                     stop_words='english',
                                     smooth_idf=True,  # prevents zero division for unseen words
                                     sublinear_tf=False)
        tfidf_result = vectorizer.fit_transform(get_document_filenames(document_path))

        scores = zip(vectorizer.get_feature_names(),
                     np.asarray(tfidf_result.sum(axis=0)).ravel())
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        for item in sorted_scores[1:(N + 1)]:
            write_text(str(item[0]), tfidf_filepath)
            # print("{0:80} Score: {1}".format(item[0], item[1]))


def read_text_file(filepath):
    with open(filepath) as content:
        user_text = content.read().replace('\n', ' ')
        return user_text


def read_text_file_wo_new_line(filepath):
    with open(filepath) as content:
        user_text = content.readlines()
        return user_text


def write_text(text, filepath):
    with open(filepath, "a") as content:
        content.write(text)
        content.write("\n")


def create_file_with_header(filepath, features):
    with open(filepath, 'a') as outcsv:
        features = ','.join(features)
        features = features.replace("\\b", "").replace("\w", "")
        outcsv.write(features)
        outcsv.write("\n")


def create_ngram_header(ngrams):
    ngram_feat = []
    for feat in ngrams:
        feat_new = feat.replace(",", "_comma")
        ngram_feat.append(feat_new)
    return ngram_feat


def return_corpus():
    corpus = []
    files = get_document_filenames(document_path)
    for single_file in files:
        user_text = read_text_file(single_file)
        corpus.append(user_text)
    return corpus


def return_corpus_wo_stopwords():
    corpus = []
    files = get_document_filenames(document_path)
    for single_file in files:
        user_text = remove_stopword_from_text(read_text_file(single_file).lower())
        corpus.append(user_text)
    return corpus


def get_userlist():
    userlist = []
    files = get_document_filenames(document_path)

    for single_user in files:
        userlist.append(single_user.split("/")[5].replace("EN", ""))
    return userlist


def get_function_words():
    with open(os.environ['HOME'] + '/repo/AliasMatching/dictionaries/Function', 'r') as f:
        functions = [x.strip() for x in f.readlines()]

        for i in range(0, len(functions)):
            if len(re.findall('\(', functions[i])) == 1 and len(re.findall('\)', functions[i])) == 0:
                functions[i] = functions[i].replace('(', '\(')
            elif len(re.findall('\(', functions[i])) == 0 and len(re.findall('\)', functions[i])) == 1:
                functions[i] = functions[i].replace(')', '\)')
            if functions[i].endswith('*'):
                functions[i] = functions[i].replace('*', '\\w*')
                functions[i] = '\\b' + functions[i]
            else:
                functions[i] = '\\b' + functions[i] + '\\b'
    return functions


def get_wordlist(filepath):
    with open(filepath, 'r') as f:
        tfidf = [x.strip() for x in f.readlines()]

        for i in range(0, len(tfidf)):
            tfidf[i] = '\\b' + tfidf[i] + '\\b'
    return tfidf


def get_most_freq_word(text):
    stopwords = nltk.corpus.stopwords.words('english')
    text = " ".join(text)

    allWords = nltk.tokenize.word_tokenize(text)
    # allWordDist = nltk.FreqDist(w.lower() for w in allWords)

    allWordExceptStopDist = nltk.FreqDist(w.lower() for w in allWords if w.lower() not in stopwords)
    # mostCommon=sorted(w for w in set(allWordExceptStopDist))
    mostCommon = allWordExceptStopDist.most_common(100)
    print(mostCommon)


def get_avg_word_sentence(text):
    sentences = text.split('.')
    sentences = [sentence.split() for sentence in sentences if len(sentence)]
    averages = [sum(len(word) for word in sentence) / len(sentence) for sentence in sentences]
    return averages


def remove_stopword_from_text(text):
    stopwords = nltk.corpus.stopwords.words('english')
    return ' '.join(filter(lambda x: x.lower() not in stopwords, text.split()))


def tokenize(string):
    return re.findall(r'\w+', string.lower())


def ngrams(N, word, strict=True):
    """generate a sequence of N-sized substrings of word.
    if strict is False, also account for P-sized substrings
    at the end of the word where P < N"""
    last = N - 1 if strict else 0
    for i in range(len(word) - last):
        yield word[i:i + N]


def create_ngram_chars(M, N):
    """gets the top M most common substrings of N characters in English words"""
    if not Path(ngram_filepath).exists():
        print("Creating character ngram file .... \n")
        corpus = return_corpus_wo_stopwords()
        n_grams = []

        for word in corpus:
            for ngram in ngrams(N, word, strict=True):
                n_grams.append(ngram)

        f = FreqDist(n_grams)
        for i in range(0, len(f.most_common(M))):
            write_text(f.most_common(M)[i][0], ngram_filepath)
            # print(f.most_common(M)[i][0])


def ngrams_words():
    from nltk import ngrams

    sentence = 'this is a foo bar sentences and i want to ngramize it'
    n = 2
    sixgrams = ngrams(sentence.split(), n)
    for grams in sixgrams:
        print(grams)


def pos_tagger():
    filepath = os.path.expanduser('~') + "/Desktop/Stormfront_Women.txt"
    noun_filepath = os.path.expanduser('~') + "/Desktop/Stormfront_Women_Noun.txt"
    adj_filepath = os.path.expanduser('~') + "/Desktop/Stormfront_Women_Adjective.txt"
    sentences = read_text_file_wo_new_line(filepath)

    nouns_all = []
    adjectives_all = []

    for sentence in sentences:
        nouns = [token for token, pos in pos_tag(word_tokenize(sentence)) if pos.startswith('N')]
        adjs = [token for token, pos in pos_tag(word_tokenize(sentence)) if pos.startswith('J')]

        for noun in nouns:
            nouns_all.append(noun)

        for adj in adjs:
            adjectives_all.append(adj)

    noun_count = nltk.FreqDist(nouns_all)
    most_common_noun = noun_count.most_common(205)

    for i in range(0, len(most_common_noun)):
        write_text(most_common_noun[i][0], noun_filepath)
        # print(most_common_noun[i][0])
    print("-------------")

    adj_count = nltk.FreqDist(adjectives_all)
    most_common_adj = adj_count.most_common(205)

    for i in range(0, len(most_common_adj)):
        write_text(most_common_adj[i][0], adj_filepath)
        # print(most_common_adj[i][0])
    print("****************")
