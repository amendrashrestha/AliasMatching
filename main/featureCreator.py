import re
import os

import nltk
import numpy as np
from nltk.tokenize import TweetTokenizer

import utilities.IOReadWrite as utilities
import utilities.IOProperties as props


class TokenizerTransformer():
    def __init__(self, text):
        self.transform(text)

    def transform(self, X):
        tknzr = TweetTokenizer()
        return [tknzr.tokenize(x) for x in X]


class StyloFeatures():
    def __init__(self):
        self.transform()

    def transform(self):
        print("Creating Stylometric features ..... \n")


        corpus = utilities.return_corpus()
        userlist = utilities.get_userlist()

        user_id = ['User_ID']
        lengths = ['Text_length']
        word_lengths = [str(x) for x in list(range(1, 21))]
        digits = [str(x) for x in list(range(0, 10))]
        symbols = list('.?!,;:()"-\'')
        smileys = [':\')', ':-)', ';-)', ':P', ':D', ':X', '<3', ':)', ';)', ':@', ':*', ':j', ':$', '%)']
        functions = utilities.get_function_words(props.function_word_filepath)
        tfidf = utilities.get_wordlist(props.tfidf_filepath)
        ngram_char = utilities.get_wordlist(props.ngram_char_filepath)

        LIWC_header = sorted(os.listdir(props.LIWC_filepath))


        digits_header = ['Digit_0', 'Digit_1', 'Digit_2', 'Digit_3', 'Digit_4', 'Digit_5', 'Digit_6', 'Digit_7',
                         'Digit_8', 'Digit_9']
        symbols_header = ['dot', 'question_mark', 'exclamation', 'comma', 'semi_colon', 'colon', 'left_bracket',
                          'right_bracket', 'double_inverted_comma', 'hypen', 'single_inverted_comma']
        smilies_header = ['smily_1', 'smily_2', 'smily_3', 'smily_4', 'smily_5', 'smily_6', 'smily_7', 'smily_8',
                          'smily_9', 'smily_10', 'smily_11', 'smily_12', 'smily_13', 'smily_14']
        ngaram_char_header = utilities.create_ngram_header(ngram_char)

        header_feature = LIWC_header + lengths + word_lengths + digits_header + symbols_header + smilies_header + functions + tfidf + \
                         ngaram_char_header + user_id

        features = LIWC_header + lengths + word_lengths + digits + symbols + smileys + functions + tfidf + ngram_char + user_id
        vector = np.zeros((len(corpus), len(features)))

        utilities.create_file_with_header(props.feature_vector_filepath, header_feature)

        row = 0
        col = 0

        for x in corpus:
            # x = "this about challeng hopefully is abnormal this ability able abilit test"
            # print(userlist[row])
            text_size = len(x.split())
            x_wo_stopword = utilities.remove_stopword_from_text(x)
            text_size_wo_stopword = len(x_wo_stopword.split())

            x_only_words = []
            for t in x.split():
                if (len(t) == 1 and t.isalpha()) or \
                        (len(t) > 1 and ("http" not in t and "www" not in t and "@" not in t and "#" not in t)):
                    x_only_words.append(t)
            # print(x_only_words)
            counts = nltk.FreqDist([len(tok) for tok in x_only_words])

            for feat in features:
                # print(feat)
                if col < len(LIWC_header):
                    LIWC_filepath = props.LIWC_filepath + feat
                    LIWC_words = utilities.get_function_words(LIWC_filepath)
                    count = 0

                    for single_word in LIWC_words:
                        count += sum(1 for i in re.finditer(single_word, x))
                    avg_count = count / text_size
                    # print(avg_count)

                    vector[row][col] = avg_count

                # Count text lengths
                elif col < len(LIWC_header) + len(lengths):
                    vector[row][col] = len(x)

                # Count word lengths
                elif col < len(LIWC_header) + len(lengths) + len(word_lengths):
                    if int(feat) in counts.keys():
                        vector[row][col] = counts.get(int(feat))
                    else:
                        vector[row][col] = 0

                # Count special symbols
                elif col < len(LIWC_header) + len(lengths) + len(word_lengths) + len(digits):
                    vector[row][col] = x.count(feat) / text_size

                # Count special symbols
                elif col < len(LIWC_header) + len(lengths) + len(word_lengths) + len(digits) + len(symbols):
                    vector[row][col] = x.count(feat) / text_size

                # Count smileys
                elif col < len(LIWC_header) + len(lengths) + len(word_lengths) + len(digits) + len(symbols) + len(smileys):
                    vector[row][col] = x.count(feat) / text_size

                # Count functions words
                elif col < len(LIWC_header) + len(lengths) + len(word_lengths) + len(digits) + len(symbols) + len(smileys) + len(
                        functions):
                    vector[row][col] = sum(1 for i in re.finditer(feat, x)) / text_size
                #
                # # Count tfidf without stop words
                elif col < len(LIWC_header) + len(lengths) + len(word_lengths) + len(digits) + len(symbols) + len(smileys) + len(
                        functions) + len(tfidf):
                    vector[row][col] = sum(1 for i in re.finditer(feat, x_wo_stopword)) / text_size_wo_stopword
                # # print(feat)
                #     # print(sum(1 for i in re.finditer(feat, x_wo_stopword)))
                #
                # # Count ngram_char without stop words
                elif col < len(LIWC_header) + len(lengths) + len(word_lengths) + len(digits) + len(symbols) + len(smileys) + len(
                        functions) + len(tfidf) + len(ngram_char):
                    vector[row][col] = sum(1 for i in re.finditer(feat, x_wo_stopword)) / text_size_wo_stopword
                #
                # # Adding userId
                elif col < len(LIWC_header) + len(lengths) + len(word_lengths) + len(digits) + len(symbols) + len(smileys) + len(
                        functions) + len(tfidf) + len(
                    ngram_char) + len(user_id):
                    vector[row][col] = userlist[row]

                if col == len(features) - 1:
                    col = 0
                    break
                col += 1

            row += 1
        with open(props.feature_vector_filepath, 'ab') as f_handle:
            np.savetxt(f_handle, vector, delimiter=",")
