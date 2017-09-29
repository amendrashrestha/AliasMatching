__author__ = 'amendrashrestha'

import collections
import re
from nltk import FreqDist

def ngrams(N, word,  strict=True):
    """generate a sequence of N-sized substrings of word.
    if strict is False, also account for P-sized substrings
    at the end of the word where P < N"""
    last = N - 1 if strict else 0
    for i in range(len(word) - last):
        yield word[i:i+N]

def m_most_common_ngram_chars(M=10, N=3):
    """gets the top M most common substrings of N characters in English words"""
    n_grams = []
    text = ["abc d", "abc de", "abc defg"]
    # n_grams = [ngram for ngram in ngrams(N, word) for word in f]
    for word in text:
        for ngram in ngrams(N, word, strict=True):
            n_grams.append(ngram)

    f = FreqDist(n_grams)
    for i in range(0, len(f.most_common(M))):
        print(f.most_common(M)[i][0])
    return f.most_common(M)


if __name__ == "__main__":
    # Uses the default values M=5, N=3
    print(m_most_common_ngram_chars())