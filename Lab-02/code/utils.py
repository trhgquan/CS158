import string, re

def load_stopwords(stopwords_file : str) -> list:
    '''Load stopwords from a list of stopwords file.
    '''
    stopwords = []

    with open(stopwords_file, 'r+', encoding = 'utf8') as f:
        for line in f.readlines():
            stopwords.append(line.strip().lower())
    
    return stopwords

def load_data(data_file : str) -> tuple:
    '''Load data from a data file to a tuple (sentence list, class)
    '''
    sentences, classes = [], []

    with open(data_file, 'r+') as f:
        for line in f.readlines():
            sentence, sentence_class = parsing(line.lower().strip())
            sentences.append(sentence)
            classes.append(sentence_class)

    return sentences, classes

def parsing(sentence : str) -> tuple:
    '''Parse and preprocess a string and class from TREC dataset.
    '''
    sentence_class = int(sentence[0].strip())
    sentence = sentence[1:].strip()

    punctuations = '[{0}]'.format(string.punctuation)

    sentence = sentence.lower()
    sentence = re.sub('\w*\d\w*', '', sentence)
    sentence = re.sub(punctuations, '', sentence)

    return (sentence, sentence_class)