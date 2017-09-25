import sys, os
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

from main.featureCreator import StyloFeatures

sys.path.append(os.environ['HOME'] + "/PycharmProjects/AliasMatching")

def main():
    print("Alias Matching !!!!")
    text1 = "This is testing!!!!"
    hyperparameters = [
        (0.01, 0.9),
        (0.1, 0.8),
        (0.001, 0.8),
        (0.001, 0.9)
    ]
    for mindf, maxdf in hyperparameters:
        TfidfVectorizer(input=text1, tokenizer=TweetTokenizer().tokenize, sublinear_tf=True, min_df=mindf, max_df=maxdf)
    StyloFeatures(text1)


if __name__ == '__main__':
    main()
