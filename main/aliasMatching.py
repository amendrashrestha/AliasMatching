import sys, os
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

from main.featureCreator import StyloFeatures

sys.path.append(os.environ['HOME'] + "/PycharmProjects/AliasMatching")

def main():
    print("Alias Matching !!!!")
    text1 = "This is testing!!!!"
    text2 = "This is again testing with same user.."

    text =[text1, text2]

    hyperparameters = [
        (0.01, 0.9),
        (0.1, 0.8),
        (0.001, 0.8),
        (0.001, 0.9)
    ]
    for mindf, maxdf in hyperparameters:
        # get tfidf
        sklearn_tfidf = TfidfVectorizer(norm='l2', tokenizer=TweetTokenizer().tokenize, sublinear_tf=True, min_df=mindf, max_df=maxdf)
        sklearn_representation = sklearn_tfidf.fit_transform(text)
        feature_names = sklearn_tfidf.get_feature_names()
        print(feature_names)
        print("---------------------------")

    StyloFeatures(text)


if __name__ == '__main__':
    main()
