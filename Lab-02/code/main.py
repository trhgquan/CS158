from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nb import NaiveBayes

import re, string

def parsing(sentence : str) -> tuple:
    sentence_class = int(sentence[0].strip())
    sentence = sentence[1:].strip()

    punctuations = '[{0}]'.format(string.punctuation)

    sentence = sentence.lower()
    sentence = re.sub(punctuations, '', sentence)

    return (sentence, sentence_class)

def main():
    with open('data/TREC.train.all', 'r+') as f:
        train_sentences, train_classes = [], []
        for line in f.readlines():
            sentence, sentence_class = parsing(line.lower().strip())
            train_sentences.append(sentence)
            train_classes.append(sentence_class)
    
    print(train_sentences[:10])

    nb = NaiveBayes()
    
    nb.fit(train_sentences, train_classes)

    with open('data/TREC.test.all', 'r+') as f:
        test_sentences, test_classes = [], []
        for line in f.readlines():
            sentence, sentence_class = parsing(line.lower().strip())
            test_sentences.append(sentence)
            test_classes.append(sentence_class)

    y_pred = nb.predict(test_sentences)

    print(classification_report(y_pred, test_classes))
    print(confusion_matrix(y_pred, test_classes))
    print('Accuracy score:', nb.score(y_pred, test_classes))

if __name__ == '__main__':
    main()