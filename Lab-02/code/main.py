from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nb import NaiveBayes
from utils import load_data, load_stopwords

def main():
    train_file = 'data/TREC.train.all'
    test_file = 'data/TREC.test.all'
    stopwords_file = 'data/stopwords.txt'

    train_data = load_data(train_file)
    test_data = load_data(test_file)

    train_sentences, train_classes = train_data
    test_sentences, test_classes = test_data

    # Switch to use stopwords to see the differences in precision, recall and F1
    stopword_list = []
    # stopword_list = load_stopwords(stopwords_file)

    nb = NaiveBayes(stopword_list = stopword_list)
    
    print('Training..')
    nb.fit(train_sentences, train_classes)

    print('Predicting test sentences..')
    y_pred = nb.predict(test_sentences)

    print('NB-from scratch')
    print(classification_report(test_classes, y_pred))
    print('Accuracy score:', accuracy_score(test_classes, y_pred))
    print('Confusion matrix:\n', confusion_matrix(test_classes, y_pred))

if __name__ == '__main__':
    main()