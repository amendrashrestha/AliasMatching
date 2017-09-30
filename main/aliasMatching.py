from main.featureCreator import StyloFeatures

import utilities.IOReadWrite as utilities


def main():
    print("Alias Matching !!!! \n")
    top_word_size = 300
    char_size = 3

    utilities.create_tfIdf(top_word_size)
    utilities.create_ngram_chars(top_word_size, char_size)
    # utilities.pos_tagger()


    StyloFeatures()


if __name__ == '__main__':
    main()
