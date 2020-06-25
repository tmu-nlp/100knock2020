from knock50 import write_data
import re

def dataCleaning(text):
    stopwords = open('english').read()
    sentences = []
    for line in text:
        line = line.strip().split('\t')
        cleaned = line[0].lower() # lowercase
        cleaned = re.sub('[^0-9a-zA-Z]+',' ', cleaned) # remove non-alphanumeric
        cleaned = ' '.join([i for i in cleaned.split() if not i in stopwords]) # remove stopwords
        sentences.append(cleaned + '\t' + line[1] + '\n')
    return sentences

if __name__ == "__main__":    
    train_data = open('train.txt').readlines()
    train_clean = dataCleaning(train_data)
    write_data(train_clean, 'train.feature.txt')

    valid_data = open('valid.txt').readlines()
    valid_clean = dataCleaning(valid_data)
    write_data(valid_clean, 'valid.feature.txt')

    test_data = open('test.txt').readlines()
    test_clean = dataCleaning(test_data)
    write_data(test_clean, 'test.feature.txt')