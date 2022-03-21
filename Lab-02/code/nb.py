from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

class NaiveBayes:
    def __init__(self) -> None:
        with open('data/stopwords.txt', 'r+', encoding = 'utf8') as f:
            self.__stopwords = []
            for line in f.readlines():
                self.__stopwords.append(line.strip().lower())

        self.__vectorizer = CountVectorizer(
            stop_words = self.__stopwords,
        )

    def fit(self, sentences_list, classes_list):
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
        self.__vect, self.__features = [None] * n_classes, [None] * n_classes
        self.__count, self.__vocab = [None] * n_classes, [None] * n_classes
    
        for i, c in enumerate(self.__classes):
            X_for_class_c = self.__X[self.__y == c]
            self.__priors[i] = X_for_class_c.shape[0] / float(n_samples)
            
            vect = CountVectorizer(stop_words = self.__stopwords, max_features = 50)
            self.__vect[i] = vect.fit_transform(X_for_class_c)
            self.__vocab[i] = vect.vocabulary_
            self.__features[i] = vect.get_feature_names_out()
            self.__count[i] = self.__vect[i].toarray().sum(axis = 0)

            print(self.__vocab[i].keys())

    def __likelihood(self, class_index, x):
        result = []

        for w in x.split(' '):
            if w in self.__features[class_index]:
                word_index = self.__vocab[class_index][w]
                result.append(((1 + self.__count[class_index][word_index]) / (len(self.__features[class_index]) + self.__vocabulary_size)))

        return result

    def __classify_sample(self, x):
        posteriors = []

        for class_index, _ in enumerate(self.__classes):
            prior = np.log(self.__priors[class_index])
            posterior = prior + np.sum(np.log(self.__likelihood(class_index, x)))
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

    @staticmethod
    def score(y_true : list, y_predicted : list):
        '''
        Returning the model's accuracy

        Input:
            - y_true (list) : correct labels
            - y_predicted (list) : predicted labels
        
        Output:
            - Score (total correct labels / total labels)
        '''
        assert len(y_true) == len(y_predicted)

        correct = 0
        for i in range(len(y_true)):
            correct += 1 if y_true[i] == y_predicted[i] else 0
        
        print('True:', correct)
        print('False:', len(y_true) - correct)

        return correct / len(y_true)