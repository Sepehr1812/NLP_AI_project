"""
program for implementation of unigram and bigram algorithms with back-off smoothing and calculating some values.
"""

import re
from math import pow, log2

# array of category names
cat_names = []
test_cat_names = []


def get_train_dataset(file_path):
    """
    returns an array of some arrays that are categorized in train text file
    :param file_path: file we want to read
    :return: dataset
    """
    dataset = []
    with open(file_path, "r", encoding='utf-8') as f:
        sentences = [re.split("\\s+", line.rstrip('\n')) for line in f]
        for i in range(0, len(sentences)):
            for k in range(0, len(sentences[i])):
                if sentences[i][k].__contains__('@'):
                    cat_name = ""
                    for w in range(0, k):
                        cat_name.__add__(sentences[i][0])
                        sentences[i].remove(sentences[i][0])
                    for j in range(0, sentences[i][0].index('@')):
                        cat_name = cat_name.__add__(sentences[i][0][j])
                    sentences[i].remove(sentences[i][0])

                    # remove @s
                    sentences[i].remove(sentences[i][0])

                    if cat_names.__contains__(cat_name):
                        dataset[cat_names.index(cat_name)].append(sentences[i])
                    else:
                        cat_names.append(cat_name)
                        arr = [sentences[i]]
                        dataset.append(arr)
                    break
        return dataset


def get_test_dataset(file_path):
    """
    returns an array of some arrays that are categorized in test text file
    :param file_path: file we want to read
    :return: dataset
    """
    with open(file_path, "r", encoding='utf-8') as f:
        sentences = [re.split("\\s+", line.rstrip('\n')) for line in f]
        for i in range(0, len(sentences)):
            for k in range(0, len(sentences[i])):
                if sentences[i][k].__contains__('@'):
                    cat_name = ""
                    for w in range(0, k):
                        cat_name.__add__(sentences[i][0])
                        sentences[i].remove(sentences[i][0])
                    for j in range(0, sentences[i][0].index('@')):
                        cat_name = cat_name.__add__(sentences[i][0][j])
                    sentences[i].remove(sentences[i][0])

                    # remove @s
                    sentences[i].remove(sentences[i][0])

                    test_cat_names.append(cat_name)
                    break

        return sentences


class UnigramLanguageModel:
    """
    class for unigram modeling
    """

    def __init__(self, sentences):
        self.unigram_frequencies = dict()
        self.corpus_length = 0
        for s in sentences:
            for word in s:
                self.unigram_frequencies[word] = self.unigram_frequencies.get(word, 0) + 1
                self.corpus_length += 1
        self.unique_words = len(self.unigram_frequencies)

    def calculate_unigram_probability(self, word):
        """
        calculates unigram probability of a word
        :param word: the word we want to model
        :return: zero if word probability is zero else logarithm of probability in base 2
        """
        word_probability_numerator = self.unigram_frequencies.get(word, 0)
        word_probability_denominator = self.corpus_length

        return 0.0 if word_probability_numerator == 0 or word_probability_denominator == 0 \
            else log2(float(word_probability_numerator) / float(word_probability_denominator))

    def calculate_sentence_probability(self, sentence_in):
        """
        probability of existence of a sentence in model
        :param sentence_in: the sentence we want to validate
        :return: the probability
        """
        sentence_probability_log_sum = 0
        for word in sentence_in:
            word_probability = self.calculate_unigram_probability(word)
            sentence_probability_log_sum += word_probability
        return sentence_probability_log_sum


class BigramLanguageModel(UnigramLanguageModel):
    """
    class for bigram modeling
    """

    def __init__(self, sentences):
        UnigramLanguageModel.__init__(self, sentences)
        self.bigram_frequencies = dict()
        for s in sentences:
            previous_word = None
            for word in s:
                if previous_word is not None:
                    self.bigram_frequencies[(previous_word, word)] = \
                        self.bigram_frequencies.get((previous_word, word), 0) + 1
                previous_word = word

    def calculate_bigram_probability(self, previous_word, word, landa_in):
        """
        calculates bigram probability of a word
        :param landa_in: amount of landa2 in back-off algorithm
        :param previous_word: previous word in bigram model
        :param word: the word we want to model
        :return: zero if word probability is zero else logarithm of probability in base 2
        """
        bigram_word_probability_numerator = self.bigram_frequencies.get((previous_word, word), 0) * \
                                            pow(2, UnigramLanguageModel.calculate_unigram_probability(self, word))
        bigram_word_probability_denominator = self.unigram_frequencies.get(previous_word, 0)

        probability = 0.0
        if not bigram_word_probability_numerator == 0 and not bigram_word_probability_denominator == 0:
            probability = float(bigram_word_probability_numerator) / float(bigram_word_probability_denominator)
            # back-off smoothing
            probability = probability * landa_in + \
                          pow(2, UnigramLanguageModel.calculate_unigram_probability(self, word)) * (1 - landa_in)
            probability = log2(probability)

        return probability

    def calculate_bigram_sentence_probability(self, sentence_in, landa_in):
        """
        probability of existence of a sentence in model
        :param landa_in: amount of landa2 in back-off algorithm
        :param sentence_in: the sentence we want to validate
        :return: the probability
        """
        bigram_sentence_probability_log_sum = 0
        previous_word = None
        for word in sentence_in:
            if previous_word is not None:
                bigram_word_probability = self.calculate_bigram_probability(previous_word, word, landa_in)
                bigram_sentence_probability_log_sum += bigram_word_probability
            previous_word = word
        return bigram_sentence_probability_log_sum


def print_validation_values(table_in):
    """
    prints values of precision, recall and f-measure of classes.
    :param table_in: the table of fails and successes in prediction
    :return: nothing
    """
    for i in range(len(cat_names)):
        # calculating precision
        ds = 0
        for j in range(len(table_in)):
            ds += table_in[i][j]
        if ds == 0:
            p = 0
        else:
            p = table_in[i][i] / ds

        # calculating recall
        dr = 0
        for j in range(len(table_in)):
            dr += table_in[j][i]

        if dr == 0:
            r = 0
        else:
            r = table_in[i][i] / dr

        # calculating f-measure
        if p + r == 0:
            f = 0
        else:
            f = 2 * p * r / (p + r)

        print("Validation values for category \"" + cat_names[i] + "\":\nPrecision = " + str(p) +
              "\nRecall = " + str(r) + "\nF-measure = " + str(f) + "\n")


if __name__ == '__main__':
    train_dataset = get_train_dataset("../HAM-Train-Test/HAM-Train.txt")
    test_dataset = get_test_dataset("../HAM-Train-Test/HAM-Test.txt")

    # initializing models
    bigram_models = []
    for x in train_dataset:
        bigram_models.append(BigramLanguageModel(x))

    # getting three landa2 in back-off smoothing algorithm
    row = input("Enter three landa values for back-off modeling: ").split()
    landas = list(map(float, row))

    for landa in landas:
        # precision and recall table
        table = [[0 for i in range(len(cat_names))] for j in range(len(cat_names))]

        for sentence in test_dataset:
            sentence_index = test_dataset.index(sentence)
            probs = []
            for model in bigram_models:
                probs.append(model.calculate_bigram_sentence_probability(sentence, landa))

            p_answer = cat_names[probs.index(min(probs))]
            a_answer = test_cat_names[sentence_index]

            # printing answers
            # print("Document " + str(sentence_index) + " predicted category: " + p_answer)
            # print("Actual category was: " + a_answer)
            # print()

            table[cat_names.index(p_answer)][cat_names.index(a_answer)] += 1

        print("\nFor landa2 = " + str(landa) + ":\n")
        print_validation_values(table)
