from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nb import NaiveBayes

import re, string

def parsing(sentence : str) -> tuple:
    sentence_class = int(sentence[0].strip())
    sentence = sentence[1:].strip()

    punctuations = '[{0}]'.format(string.punctuation)

    sentence = sentence.lower()
    sentence = re.sub('\w*\d\w*', '', sentence)
    sentence = re.sub(punctuations, '', sentence)

    return (sentence, sentence_class)

def main():
    item_summerize = dict()
    with open('data/TREC.train.all', 'r+') as f:
        train_sentences, train_classes = [], []
        for line in f.readlines():
            sentence, sentence_class = parsing(line.lower().strip())
            train_sentences.append(sentence)
            train_classes.append(sentence_class)

            if sentence_class not in item_summerize:
                item_summerize[sentence_class] = 1
            else:
                item_summerize[sentence_class] += 1

    print('Class items summerize', item_summerize)

    nb = NaiveBayes()
    
    nb.fit(train_sentences, train_classes)

    with open('data/TREC.test.all', 'r+') as f:
        test_sentences, test_classes = [], []
        for line in f.readlines():
            sentence, sentence_class = parsing(line.lower().strip())
            test_sentences.append(sentence)
            test_classes.append(sentence_class)

    y_pred = nb.predict(test_sentences)

    print('NB-from scratch')
    print(classification_report(test_classes, y_pred))
    print('Accuracy score:', accuracy_score(test_classes, y_pred))
    print('Confusion matrix:\n', confusion_matrix(test_classes, y_pred))

if __name__ == '__main__':
    main()