__author__ = 'amendrashrestha'

import glob
import os
import re


def read_text_file(filepath):
    with open(filepath) as content:
        user_text = content.read().replace('\n', ' ')
        return user_text


def get_files(folderPath):
    files = [file for file in glob.glob(folderPath + '/*/**/*.txt', recursive=True)]
    return files


def write_text(text, filepath):
    with open(filepath, "a") as content:
        content.write(text)
        content.write("\n")


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


def get_tfidf_words():
    with open(os.environ['HOME'] + '/repo/AliasMatching/dictionaries/TfIdf', 'r') as f:
        tfidf = [x.strip() for x in f.readlines()]

        for i in range(0, len(tfidf)):
            tfidf[i] = '\\b' + tfidf[i] + '\\b'
    return tfidf


def get_avg_word_sentence(text):
    sentences = text.split('.')
    sentences = [sentence.split() for sentence in sentences if len(sentence)]
    averages = [sum(len(word) for word in sentence) / len(sentence) for sentence in sentences]
    return averages


def remove_stopword_from_text(text):
    with open(os.environ['HOME'] + '/repo/AliasMatching/dictionaries/StopWord', 'r') as f:
        stopwords = [x.strip() for x in f.readlines()]
    return ' '.join(filter(lambda x: x.lower() not in stopwords, text.split()))