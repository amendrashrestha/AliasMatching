import nltk
import re
import os
import numpy as np
from nltk.tokenize import TweetTokenizer

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
        lengths = [str(x) for x in list(range(1, 21))]
        symbols = list('.?!,;:()"-\'')
        smileys = [':\')', ':-)', ';-)', ':P', ':D', ':X', '<3', ':)', ';)', ':@', ':*', ':j', ':$', '%)']
        with open(os.environ['HOME'] + '/PycharmProjects/AliasMatching/dictionaries/Function', 'r') as f:
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

        features = lengths + symbols + smileys + functions
        vector = np.zeros((len(text), len(features)))

        row = 0
        col = 0

        for x in text:
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
                    vector[row][col] = x.count(feat)

                # Count smileys
                elif col < len(lengths) + len(symbols) + len(smileys):
                    vector[row][col] = x.count(feat)

                # Count functions words
                elif col < len(lengths) + len(symbols) + len(smileys) + len(functions):
                    vector[row][col] = len(re.findall(feat, " ".join(x).lower()))

                if col == len(features) - 1:
                    col = 0
                    break
                col += 1
            row += 1
        print(vector)

        return vector
