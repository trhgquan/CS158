import numpy as np
from tqdm import tqdm

class NaiveBayes:
    '''Naive Bayes implementation, 
    re-implemented with the algorithm in Dan Jurafsky's Speech and Language Processing (3rd ed. draft)
    https://web.stanford.edu/~jurafsky/slp3/4.pdf
    '''

    def __init__(self) -> None:
        '''Initialise variables.
        '''

        self.__stopwords_list = []
        self.__number_docs = 0
        self.__number_classes = 0
        self.__vocabulary = []
        self.__vocab_len = 0
        self.__classes = []
        self.__prior = {}
        self.__likelihood = {}
        self.__doc_in_class = {}

    @staticmethod
    def get_vocabulary(sentence_list : list, stopwords_list : list) -> list:
        '''Get vocabulary from a sentence list and stopwords list.
        '''
        vocabulary = []

        for sentence in sentence_list:
            for word in sentence.split(' '):
                if len(word) > 0 and word not in stopwords_list and word not in vocabulary:
                    vocabulary.append(word)

        return vocabulary
    
    @staticmethod
    def doc_in_class(sentences_list : list, classes : list, classes_list : list) -> dict:
        '''Divide list of sentences to a dictionary: (key : (sentence1, .., sentence n))
        '''
        doc_in_class_dict = {}

        for this_class in classes:
            doc_in_class_dict[this_class] = []
            for sentence, that_class in zip(sentences_list, classes_list):
                if that_class == this_class:
                    doc_in_class_dict[this_class].append(sentence)
        
        return doc_in_class_dict
    
    @staticmethod
    def count(word : str, big_doc : list) -> int:
        '''Count a word's appearance in a doc (list of sentences).
        '''
        count = 0

        for sentence in big_doc:
            for word_ in sentence.split(' '):
                if word_ == word:
                    count += 1

        return count

    def fit(self, sentences_list : list, classes_list : list, stopwords_list = [], verbose = True) -> None:
        '''Fit a model with sentences list and classes list
        '''
        self.__stopwords_list = stopwords_list

        self.__number_docs = len(sentences_list)
        self.__classes = list(set(classes_list))
        self.__number_classes = len(self.__classes)
        self.__vocabulary = NaiveBayes.get_vocabulary(sentences_list, self.__stopwords_list)
        self.__vocab_len = len(self.__vocabulary)
        self.__doc_in_class = NaiveBayes.doc_in_class(sentences_list, self.__classes, classes_list)

        for current_class in self.__classes:
            current_class_doc = len(self.__doc_in_class[current_class])
            self.__prior[current_class] = np.log(self.__number_docs) - np.log(current_class_doc)

            count_all_words = sum([(NaiveBayes.count(word, self.__doc_in_class[current_class]) + 1) for word in self.__vocabulary])

            verbose_set = tqdm(self.__vocabulary) if verbose else self.__vocabulary

            for word in verbose_set:
                count_this_word = NaiveBayes.count(word, self.__doc_in_class[current_class])

                word_prob = np.log(count_this_word + 1) - np.log(count_all_words)

                if current_class not in self.__likelihood:
                    self.__likelihood[current_class] = { word : word_prob }
                else:
                    self.__likelihood[current_class].update({ word : word_prob })

    def __predict(self, sentence : str) -> int:
        '''Predict a sentence.
        '''

        sum_prob = [self.__prior[c] for c in self.__classes]

        for this_class in self.__classes:
            for word in sentence.split(' '):
                if word in self.__vocabulary:
                    sum_prob[this_class] += self.__likelihood[this_class][word]

        return np.argmax(np.array(sum_prob, dtype = np.float64))

    def predict(self, sentences_list : list, verbose = True) -> list:
        '''Predict a given list of sentences.
        '''
        verbose_list = tqdm(sentences_list) if verbose else sentences_list
        return [self.__predict(s) for s in verbose_list]