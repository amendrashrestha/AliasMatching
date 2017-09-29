import sys
import time
from multiprocessing import Process
from multiprocessing import Queue


class SearchModule():
    def __init__(self, verbose=False, relative=True):
        self.relative = relative
        self.verbose = verbose

    def increment_categories(self, user_categories, word, relative, total_length):
        # Look up the corresponding category dictionaries and increment the word count for this word
        for category_dictionary_name in self.global_words[word]:
            # Get the corresponding category dicionary and increment the word value
            category_dictionary = user_categories[category_dictionary_name]
            if relative:
                increment_value = 1.0 / total_length
            else:
                increment_value = 1.0
            category_dictionary[word] += increment_value
        return user_categories

    def copy_global_categories(self, global_categories):
        copy = {}
        for category_name, category in zip(global_categories.keys(), global_categories.values()):
            copy.update({category_name: category.copy()})
        return copy

    def print_dictionary(self, dictionary):
        for attr, value in sorted(dictionary.items()):
            print("{}={}".format(attr, value))

    def update_progress_bar(self, value, end_value, bar_length=60):
        percent = float(value) / end_value
        arrow = '-' * int(round(percent * bar_length) - 1) + '>'
        spaces = ' ' * (bar_length - len(arrow))

        sys.stdout.write(
            "\rProgress: [{}] {}% - Finished {} of {}. ".format(arrow + spaces, int(round(percent * 100)), value,
                                                                end_value))
        sys.stdout.write("\033[K")
        if value == end_value:
            sys.stdout.write("\n")

        sys.stdout.flush()

    def process_category_word(self, word):
        cut_string = False
        full_word = word
        if word.endswith('*'):  # Regex word
            # Cut the asterisk out and flag that the string has been cut
            word = word[:len(word) - 1]
            cut_string = True
        else:
            cut_string = False

        return word, full_word, cut_string, len(word.split())

    def match_words(self, text_word, word, cut_string):
        match = False
        if cut_string:
            # Only look at the initial subset of the chars to catch regex
            text_word = text_word[:len(word)]
        if text_word == word:  # Match found
            match = True
            if cut_string:
                # Put back asterisk for correct lookup with category dictionaries
                text_word += '*'
        return text_word, match

    def measure_lingmarks(self, username, text, index, output_queue, sum_values=False, print_user_results=False):
        # Obtain a copy of the categories to avoid populating the same dictionary with every user
        user_categories = self.copy_global_categories(self.global_categories)

        for i, window in enumerate(self.global_words):  # Go through each word from the global dictionary of words
            category_window, full_window, cut_string, window_size = self.process_category_word(window)

            # Adjust search range with respect text and window size (+ indexing fix)
            search_range = len(text) - window_size + 1

            # Check for potential matches between category words and text (within window size)
            for m in range(search_range):
                # Search over a window with same size as the category window
                text_window = " ".join(text[m:m + window_size])
                text_window, match = self.match_words(text_window, category_window, cut_string)
                if match:
                    # Increment word count for the given categories
                    user_categories = self.increment_categories(user_categories, text_window, self.relative, len(text))
        # if self.verbose:
        #    print(">>>>> LingMark search for user '{}' finished in {:.3f} seconds.".format(username, time.time() - start_time))

        if sum_values:
            summed_dict = {}
            for category in user_categories.keys():
                summed_dict.update({category: sum(user_categories[category].values())})

        if print_user_results:
            print("Result:")
            for category in user_categories.keys():
                print('{}: {}'.format(category, sum(user_categories[category].values())))
                print("\n")

        # Put resulting output in the output queue
        if not sum_values:
            output_queue.put((username, user_categories))
        else:
            output_queue.put((username, summed_dict))

    def search(self, dictionary, global_categories, global_words, sum_values):
        self.global_categories = global_categories
        self.global_words = global_words

        processes = []
        output_queue = Queue()
        output_dictionary = {}
        if self.verbose:
            print(">>>>> Initiating LingMark search for {} users.".format(len(dictionary)))

        # Start timer
        start_time = time.time()

        for index, user in enumerate(dictionary):
            p = Process(target=self.measure_lingmarks, args=(user, dictionary[user], index, output_queue, sum_values,))
            processes.append(p)

        # Start processes
        for p in processes:
            p.start()

        # Gather results
        results_dictionary = {}
        if self.verbose:
            self.update_progress_bar(len(results_dictionary), len(processes))
        for r in range(len(processes)):
            output = output_queue.get()
            results_dictionary.update({output[0]: output[1]})
            if self.verbose:
                self.update_progress_bar(len(results_dictionary), len(processes))
        if self.verbose:
            print(">>>>> LingMark search for {} users finished in {:.3f} seconds.".format(len(processes),
                                                                                          time.time() - start_time))

        # Wait for all processes to finish
        for p in processes:
            p.join()

        return results_dictionary
