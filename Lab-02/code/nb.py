from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

class NaiveBayes:
    def __init__(self) -> None:
        self.__alpha = 0.000001 # magic number

        with open('data/stopwords.txt', 'r+', encoding = 'utf8') as f:
            self.__stopwords = []
            for line in f.readlines():
                self.__stopwords.append(line.strip().lower())

        self.__vectorizer = CountVectorizer(
            stop_words = self.__stopwords,
        )

    def fit(self, sentences_list : list, classes_list : list) -> None:
        '''Fit a model with sentences list and classes list

        Input:
            - sentences_list : list of sentences (if you're familiar with sklearn.NB, this is X)
            - classes_list : list of classes (if you're familiar with sklearn.NB, this is y)
        '''
        self.__vectorizer = self.__vectorizer.fit(sentences_list)

        self.__vocabulary_size = len(self.__vectorizer.vocabulary_)

        self.__X = np.array(sentences_list)

        # Đếm số class
        self.__y = np.array(classes_list)
        
        n_samples = len(self.__X)

        self.__classes = np.unique(classes_list)

        n_classes = len(self.__classes)

        # Prior của lớp c = số sample lớp c / tổng số sample
        self.__priors = np.zeros(n_classes, dtype = np.float64)

        # Số lần xuất hiện của từng keyword lưu trong __count,
        # bộ vocabulary lưu trong __vocab
        self.__count, self.__vocab = [None] * n_classes, [None] * n_classes
    
        for i, c in enumerate(self.__classes):
            X_for_class_c = self.__X[self.__y == c]
            
            # Giới hạn lại 86 sentences mỗi class (dataset imbalanced => undersampling)
            X_for_class_c = X_for_class_c[:86]

            self.__priors[i] = np.log(X_for_class_c.shape[0]) - np.log(n_samples)
            
            vect = CountVectorizer(
                stop_words = self.__stopwords,
            )
            vect_fit = vect.fit_transform(X_for_class_c)
            self.__vocab[i] = vect.vocabulary_
            self.__count[i] = vect_fit.toarray().sum(axis = 0)

    def __likelihood(self, class_index : int, x : str) -> list:
        '''Calculate P(x|class_index) probability

        The formula is P(x|class_index) = (alpha + count(x in class_index)) / (len(class_index) + alpha * vocab_size)

        Input:
            - class_index : class index (in this case, the sentence's label).
            - x : in this case a sentence.
        
        Output:
            - list of probabilities.
        '''
        result = []

        for w in x.split(' '):
            if w in self.__vocab[class_index].keys():
                word_index = self.__vocab[class_index][w]
                likelihood = np.log(self.__alpha + self.__count[class_index][word_index]) - np.log(len(self.__vocab[class_index].keys()) + self.__alpha * self.__vocabulary_size)
                result.append(likelihood)

        return result

    def __classify_sample(self, x : str) -> int:
        '''Find the best label for a sentence.

        Input:
            - x : a sentence (yey!)
        
        Output:
            - label for that sentence.
        '''
        posteriors = []

        for class_index, _ in enumerate(self.__classes):
            posterior = self.__priors[class_index] * np.sum(self.__likelihood(class_index, x))
            posteriors.append(posterior)

        return self.__classes[np.argmax(posteriors)]

    def predict(self, sentences_list : list) -> list:
        '''Predict a given list.

        Input:
            - sentences_list (list) : list of sentences to be predicted.

        Output:
            - a list containing predicted labels.
        '''
        y_predicted = np.array([self.__classify_sample(x) for x in sentences_list])

        return y_predicted