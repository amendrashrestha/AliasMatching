import csv
import os
import sys
import time
from itertools import islice
import codecs

import numpy as np
from nltk.tokenize import TweetTokenizer

from main.search_module import SearchModule

csv.field_size_limit(sys.maxsize)


class LingMark():
    def __init__(self, category_path='LIWC', verbose=False, debug=False, relative=True):
        self.category_paths = self.get_sub_paths(category_path)
        self.verbose = verbose
        self.debug = debug

        self.categories, self.words = self.build_dictionaries(self.category_paths)
        self.weights = self.build_category_weights(category_path + '/weights/weights.csv')
        self.category_config = self.build_category_config(category_path + '/config/config.csv')
        self.search_module = SearchModule(verbose=verbose, relative=relative)

    def get_sub_paths(self, path):
        paths = sorted([os.path.join(path, fn) for fn in os.listdir(path)])
        # Separate directories from files:
        cleaned_paths = []
        for obj_path in paths:
            if not os.path.isdir(obj_path):
                cleaned_paths.append(obj_path)
        return cleaned_paths

    def get_filename(self, path):
        exploded_path = path.split('/')
        filename = exploded_path[len(exploded_path) - 1]
        category_name = filename.split('.')[0]
        return category_name

    def get_category_words(self, path):
        category_words = list(open(path, "r", encoding="utf-8").readlines())
        category_words = [word.strip() for word in category_words]

        category_dictionary = {}
        for word in category_words:
            category_dictionary.update({word: 0.0})

        return category_dictionary

    def build_dictionaries(self, paths):
        categories = {}
        words = {}

        for path in paths:
            category_name = self.get_filename(path).lower()
            category_dictionary = self.get_category_words(path)

            for word in category_dictionary:
                if word in words:
                    # Word exists in dictionary of words
                    # Add the category name to the corresponding word 
                    words[word] += [category_name]
                else:
                    # Word does not exist in dictionary of words
                    # Add it and add the category name to the corresponding word
                    words.update({word: [category_name]})

                # Add the category dictionary to the categories
                categories.update({category_name: category_dictionary})

        return categories, words

    def build_category_weights(self, path):
        reader = csv.reader(open(path, 'r'))
        weights = {}
        for row in reader:
            category, weight = row
            weights[category] = float(weight)

        return weights

    def build_category_config(self, path):
        reader = csv.reader(open(path, 'r'))
        configs = {}
        for row in reader:
            category, config = row
            configs[category] = bool(config)
        return configs

    def preprocess_text(self, text):
        tokenizer = TweetTokenizer(preserve_case=False)
        tokenized_text = tokenizer.tokenize(text)
        return tokenized_text

    def test_run(self):
        input_dictionary = {}
        print('Opening CSV')
        start_time = time.time()
        with open('trump.tsv') as file:
            csvreader = csv.reader(file, delimiter="\t")
            print('Reader ready')
            for row in islice(csvreader, 0, 100):
                input_dictionary.update({row[0]: self.preprocess_text(row[3])})

        print('Done. Took {} s'.format(time.time() - start_time))
        return input_dictionary

    def prepare_input(self, tweets):
        input_dictionary = {}
        for tweet in tweets:
            input_dictionary.update({tweet['id']: self.preprocess_text(tweet['text'])})

        return input_dictionary

    def get_all_user_values(self, tweets, sum_values=False):
        search_input_dictionary = self.prepare_input(tweets)
        # print('LENGTH ' + str(len(search_input_dictionary)))
        output_dictionary = self.search_module.search(search_input_dictionary, self.categories,
                                                      self.words, sum_values=sum_values)

        return output_dictionary

    def get_filtered_users(self, input_dictionary=None, filter_dictionary=None):
        if filter_dictionary is None:
            # Dummy example, baseline average
            filter_dictionary = {'power': 0.0, 'they': 0.0, 'anger': 0.0, 'posemo': 0.0, 'excl': 0.0, 'incl': 0.0,
                                 'negemo': 0.0}

        if input_dictionary is None:
            input_dictionary = self.get_all_users()

        filtered_users = {}
        for n, user in enumerate(input_dictionary):
            user_dictionary = input_dictionary[user]
            user_match_array = []
            if self.debug:
                print(user)
            for category in filter_dictionary:
                if self.debug:
                    print('Category: {}, Value: {}'.format(category, sum(user_dictionary[category].values())))
                user_match_array.append(sum(user_dictionary[category].values()) > filter_dictionary[category])
            if self.debug:
                print('User: {}, Match: {}'.format(user, user_match_array))
            if sum(user_match_array) == len(filter_dictionary):
                filtered_users.update({user: user_dictionary})
        if self.debug:
            for user in filtered_users:
                print('User: {}'.format(user))

        return filtered_users

    def get_heat_ranking(self, top=-1, descending=True, reverse_and_normalize=False, input_dictionary=None):
        if input_dictionary is None:
            input_dictionary = self.get_all_users()

        user_heat_rank = {}

        # Add all users to user heat ranking dictionary
        for user in input_dictionary:
            user_heat_rank.update({user: 0})

        # Get user ranking for each category
        for category in self.categories:
            _, category_ranking = self.get_ranked_users_for_category(category, top=top,
                                                                     descending=self.category_config[category],
                                                                     input_dictionary=input_dictionary)

            # Assign ranking value to each user in the user heat ranking dictionary
            for rank, user in enumerate(category_ranking):
                user_heat_rank[user] += rank + 1  # +1 for indexing correction

        sorted_users = sorted(user_heat_rank, key=user_heat_rank.__getitem__)
        if not descending:
            sorted_users = sorted_users[::-1]

        if reverse_and_normalize:
            # TODO
            max_value = max(user_heat_rank.values())
            for user in user_heat_rank:
                user_heat_rank[user] /= float(max_value)

        if self.debug:
            print('Heat values {}'.format(user_heat_rank))
            print('Sorted users: {}'.format(sorted_users))

        return user_heat_rank, sorted_users

    def get_ranked_users(self, top=-1, input_dictionary=None, weighted=False):
        if input_dictionary is None:
            input_dictionary = self.get_all_users()

        weight = 1.0
        user_values = {}
        for n, user in enumerate(input_dictionary):
            user_dictionary = input_dictionary[user]
            total = 0.0
            for category in user_dictionary:
                if weighted:
                    weight = self.weights[category]

                total += (sum(user_dictionary[category].values()) * weight)
            user_values.update({user: total})

        sorted_users = sorted(user_values, key=user_values.__getitem__)[::-1]

        if top > 0:
            sorted_users = sorted_users[:top]
            top_user_values = {}
            for m, user in enumerate(user_values):
                if user in sorted_users:
                    top_user_values.update({user: user_values[user]})
                if top == m + 1:
                    break
            user_values = top_user_values

        if self.debug:
            print('User values: {}'.format(user_values))
            print('Sorted users: {}'.format(sorted_users))

            for user in sorted_users:
                print('User: {}, value: {}'.format(user, user_values[user]))

        return user_values, sorted_users

    def get_ranked_users_for_category(self, category, descending=True, top=-1, input_dictionary=None):
        if input_dictionary is None:
            input_dictionary = self.get_all_users()

        user_values = {}
        for n, user in enumerate(input_dictionary):
            user_dictionary = input_dictionary[user]
            value = sum(user_dictionary[category].values())
            user_values.update({user: value})
            sorted_users = sorted(user_values, key=user_values.__getitem__)

        if descending:
            sorted_users = sorted_users[::-1]

        if top > 0:
            sorted_users = sorted_users[:top]
            top_user_values = {}
            for m, user in enumerate(user_values):
                if user in sorted_users:
                    top_user_values.update({user: user_values[user]})
                if top == m + 1:
                    break
            user_values = top_user_values

        if self.debug:
            print('User values: {}'.format(user_values))
            print('Sorted users: {}'.format(sorted_users))

            for user in sorted_users:
                print('User: {}, Value: {}'.format(user, user_values[user]))

        return user_values, sorted_users

    def get_category_values(self, category, input_dictionary=None):
        if input_dictionary is None:
            input_dictionary = self.get_all_users()

        category_values = []
        for n, user in enumerate(input_dictionary):
            user_dictionary = input_dictionary[user]
            category_values.append(sum(user_dictionary[category].values()))

        if self.debug:
            print('Category values: {}'.format(category_values))

        return category_values

    def get_category_values_for_user(self, category, username, input_dictionary=None):
        if input_dictionary is None:
            input_dictionary = self.get_all_users()

        category_values = []

        user_dictionary = input_dictionary[username]
        category_values.append(sum(user_dictionary[category].values()))

        if self.debug:
            print('Category values: {}'.format(category_values))

        return category_values

    def get_category_values_for_users(self, usernames, input_dictionary=None):
        if input_dictionary is None:
            input_dictionary = self.get_all_users()

        category_value_dictionary = {}
        for category in self.categories:
            values = self.get_category_values_for_user(category, usernames, input_dictionary=input_dictionary)
            category_value_dictionary.update({category: values})

        if self.debug:
            for category in category_value_dictionary:
                print('Category: {}, Value: {}'.format(category, np.max(category_value_dictionary[category])))

        return category_value_dictionary

    def get_all_category_values(self, input_dictionary=None):
        if input_dictionary is None:
            input_dictionary = self.get_all_users()

        category_value_dictionary = {}
        for category in self.categories:
            values = self.get_category_values(category, input_dictionary=input_dictionary)
            category_value_dictionary.update({category: values})

        if self.debug:
            for category in category_value_dictionary:
                print('Category: {}, Value: {}'.format(category, np.max(category_value_dictionary[category])))

        return category_value_dictionary
