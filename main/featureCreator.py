import nltk
import re
import os
import numpy as np
from nltk.tokenize import TweetTokenizer

import utilities.IOReadWrite as utilities

class TokenizerTransformer():
    def __init__(self, text):
        self.transform(text)

    def transform(self, X):
        tknzr = TweetTokenizer()
        return [tknzr.tokenize(x) for x in X]

class StyloFeatures():
    def __init__(self, text):
        self.transform(text)

    def transform(self, text):
        # print(X)
        feature_vector_filepath = os.path.expanduser('~') + "/Downloads/PAN-15/feature_vector.csv"
        lengths = [str(x) for x in list(range(1, 21))]
        symbols = list('.?!,;:()"-\'')
        smileys = [':\')', ':-)', ';-)', ':P', ':D', ':X', '<3', ':)', ';)', ':@', ':*', ':j', ':$', '%)']
        functions = utilities.get_function_words()
        tdidf = utilities.get_tfidf_words()

        features = lengths + symbols + smileys + functions + tdidf
        vector = np.zeros((len(text), len(features)))

        row = 0
        col = 0

        for x in text:
            text_size = len(x.split())
            x_wo_stopword = utilities.remove_stopword_from_text(x)
            text_size_wo_stopword = len(x_wo_stopword.split())

            x_only_words = []
            for t in x:
                if (len(t) == 1 and t.isalpha()) or \
                        (len(t) > 1 and ("http" not in t and "www" not in t and "@" not in t and "#" not in t)):
                    x_only_words.append(t)

            counts = nltk.FreqDist([len(tok) for tok in x_only_words])

            for feat in features:
                # Count word lengths
                if col < len(lengths):
                    if int(feat) in counts.keys():
                        vector[row][col] = counts.get(int(feat))
                    else:
                        vector[row][col] = 0

                # Count special symbols
                elif col < len(lengths) + len(symbols):
                    vector[row][col] = x.count(feat)/text_size

                # Count smileys
                elif col < len(lengths) + len(symbols) + len(smileys):
                    vector[row][col] = x.count(feat)/text_size

                # Count functions words
                elif col < len(lengths) + len(symbols) + len(smileys) + len(functions):
                    vector[row][col] = sum(1 for i in re.finditer(feat, x))/text_size

                # Count tfidf
                elif col < len(lengths) + len(symbols) + len(smileys) + len(functions) + len(tdidf):
                    vector[row][col] = sum(1 for i in re.finditer(feat, x_wo_stopword))/text_size_wo_stopword
                    # print(feat)
                    print(sum(1 for i in re.finditer(feat, x_wo_stopword)))

                if col == len(features) - 1:
                    col = 0
                    break
                col += 1
            row += 1
        print(vector)
        np.savetxt(feature_vector_filepath, vector, delimiter=",")

        # return vector
