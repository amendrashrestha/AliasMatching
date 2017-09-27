import sys
import os
import numpy as np
import glob

from sklearn.feature_extraction.text import TfidfVectorizer

from main.featureCreator import StyloFeatures
import utilities.IOReadWrite as utilities


sys.path.append(os.environ['HOME'] + "/repo/AliasMatching")
user_filepath = os.path.expanduser('~') + "/Downloads/PAN-15/"
tfidf_filepath = os.path.expanduser('~') + "/repo/AliasMatching/dictionaries/TfIdf"


def main():
    print("Alias Matching !!!! \n")
    # vectorizer = create_vectorizer()
    # tfidf_result = vectorizer.fit_transform(get_document_filenames())
    # display_scores(vectorizer, tfidf_result)
    # text1 = ":) this is 1 absolutely therefore actually bad dog almost yourselves test :) ? ."
    # text2 = "hello w :) ? . my almost name is sisters cousins yourselves yourselves ya'lltr your Amendra :P"
    # #
    corpus = []
    # corpus.append(text1)
    # corpus.append(text2)
    # #
    all_files = utilities.get_files(user_filepath)
    # # #
    for single_file in all_files:
        user_text = utilities.read_text_file(single_file)
        corpus.append(user_text)

    # print(corpus)

    # get tfidf

    # tfidf_matrix = tfidf.fit_transform(corpus)
    #
    # indices = np.argsort(tfidf.idf_)[::-1]
    # features = tfidf.get_feature_names()
    # print(features.__len__())
    # top_n = 20
    # top_features = [features[i] for i in indices[:top_n]]
    # print(top_features)

    # dense = tfidf_matrix.todense()
    #
    StyloFeatures(corpus)

def get_document_filenames(document_path = user_filepath):
    files = [file for file in glob.glob(document_path + '/*/**/*.txt', recursive=True)]
    return files

def create_vectorizer():
    tfidf = TfidfVectorizer(input='filename', analyzer='word', ngram_range=(1, 3), min_df=2, max_df=15, stop_words='english',
                            smooth_idf=True,  # prevents zero division for unseen words
                            sublinear_tf=False)
    return tfidf

def display_scores(vectorizer, tfidf_result):
    scores = zip(vectorizer.get_feature_names(),
                 np.asarray(tfidf_result.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    for item in sorted_scores:
        utilities.write_text(str(item[0]), tfidf_filepath)
        # print("{0:80} Score: {1}".format(item[0], item[1]))

if __name__ == '__main__':
    main()
