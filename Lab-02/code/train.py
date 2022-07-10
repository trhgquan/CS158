from utils import load_data, load_stopwords
from nb import NaiveBayes
import pickle as pkl
import sys, getopt

def main(argv):
    # train_file = 'data/TREC.train.all'
    # stopwords_file = 'data/stopwords.txt'
    train_file, model_file, stopwords_file = '', '', ''

    try:
        opts, args = getopt.getopt(argv, 'hi:o', ['input=', 'model=', 'stopwords='])
    except getopt.GetoptError:
        print('Command invalid, please try again')
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == '--input':
            train_file = arg
        elif opt == '--model':
            model_file = arg
        elif opt == '--stopwords':
            stopwords_file = arg

    if train_file == '' or model_file == '':
        print('Missing arguments, please try again.')
        sys.exit(2)

    # Load training data
    train_data = load_data(train_file)

    train_sentences, train_classes = train_data

    print(f'Finished loading data in {train_file}')

    # Load stopwords (if required)
    if stopwords_file == '':
        stopwords_list = []
    else:
        stopwords_list = load_stopwords(stopwords_file)
        print(f'Finished loading stopwords in {stopwords_file}')

    # Training
    nb = NaiveBayes()
    
    print('Training..')
    nb.fit(train_sentences, train_classes, stopwords_list)

    # Save trained model
    with open(model_file, 'wb+') as f:
        pkl.dump(nb, f)
        print(f'Successfully generated pretrain in {model_file}.')

if __name__ == '__main__':
    main(sys.argv[1:])