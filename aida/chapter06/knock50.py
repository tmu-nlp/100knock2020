import random
#import numpy as np
import pandas as pd
from collections import defaultdict

def read_data(file_path='./NewsAggregatorDataset/newsCorpora.csv'):
    df = []
    target_publisher = ['Reunters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']
    with open(file_path) as fp:
        for article in fp:
            fields = article.strip().split('\t')
            if fields[3] in target_publisher:
                df.append(fields)
    return df

def write_file(df, ids, file_name):
    category_freq = defaultdict(lambda: 0)
    with open(file_name, 'w') as fp:
        for index in ids:
            headline = df[index][1]
            category = df[index][4]
            category_freq[category] += 1
            category_headline = [category, headline]
            fp.write('{}\n'.format('\t'.join(category_headline)))
    print('Business: {}, Technology: {}, Entertainment: {}, Health: {}'.format(category_freq['b'], category_freq['t'], category_freq['e'], category_freq['m']))

if __name__ == '__main__':
    df = read_data()
    random.seed(1)
    random.shuffle(df)
    train_ids = random.sample(range(len(df)), int(len(df)*0.8))
    dev_ids = random.sample(list(set(range(len(df))) - set(train_ids)), int(len(df)*0.1))
    test_ids = list(set(range(len(df))) - set(dev_ids) - set(train_ids))

    print('train: {} articles'.format(len(train_ids)))
    write_file(df, train_ids, file_name='./train.txt')
    print('dev: {} articles'.format(len(dev_ids)))
    write_file(df, dev_ids, file_name='./dev.txt')
    print('test: {} articles'.format(len(test_ids)))
    write_file(df, test_ids, file_name='./test.txt')

