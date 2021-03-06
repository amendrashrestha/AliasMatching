import os

document_path = os.path.expanduser('~') + "/repo/AliasMatching/PAN-15-Train/"
tfidf_filepath = os.path.expanduser('~') + "/repo/AliasMatching/dictionaries/TfIdf"
ngram_filepath = os.path.expanduser('~') + "/repo/AliasMatching/dictionaries/Ngram_char"

feature_vector_filepath = os.path.expanduser('~') + "/repo/AliasMatching/PAN-15-Train/feature_vector_train.csv"
function_word_filepath = os.environ['HOME'] + '/repo/AliasMatching/dictionaries/Function'
tfidf_filepath = os.environ['HOME'] + '/repo/AliasMatching/dictionaries/TfIdf'
ngram_char_filepath = os.environ['HOME'] + '/repo/AliasMatching/dictionaries/Ngram_char'
LIWC_filepath = os.environ['HOME'] + '/repo/AliasMatching/LIWC/'