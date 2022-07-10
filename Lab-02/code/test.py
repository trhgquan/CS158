from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils import load_data
import pickle as pkl
import sys, getopt

def main(argv):
    test_file, model_file = '', ''
    try:
        opts, args = getopt.getopt(argv, 'hi:o', ['input=', 'model='])
    except getopt.GetoptError:
        print('Command invalid, please try again')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '--input':
            test_file = arg
        elif opt == '--model':
            model_file = arg

    if test_file == '' or model_file == '':
        print('Missing arguments, please try again.')
        sys.exit(2)

    # Load testing data
    test_data = load_data(test_file)
    test_sentences, test_classes = test_data
    print(f'Finished loading testing data from {test_file}')

    try:
        with open(model_file, 'rb+') as f:
            nb = pkl.load(f)
            print(f'Finished loading model from {model_file}')

        print('Predicting test sentences..')
        y_pred = nb.predict(test_sentences)

        print(classification_report(test_classes, y_pred))
        print('Accuracy score:', accuracy_score(test_classes, y_pred))
        print('Confusion matrix:\n', confusion_matrix(test_classes, y_pred))

    except FileNotFoundError:
        print('No pretrain found, try running `train.py` to generate pretrain.')

if __name__ == '__main__':
    main(sys.argv[1:])