# import pandas as pd
import random

def read_news(text,publisher):
    text = text.readlines()
    dataset = []
    for line in text:
        line = line.split('\t')
        if line[3] == publisher:
            data = line[1] + '\t' + line[4] + '\n'
            dataset.append(data)   
    return dataset 

def write_data(text,output):
    with open(output,'w') as fout:
        for line in text:
            fout.writelines(line)

if __name__ == "__main__":
    news_data = open('newsCorpora.csv')
    news = read_news(news_data,'Huffington Post')
    news = random.sample(news, len(news))

    test = news[:int(0.1*len(news))]                    # 10%
    valid = news[int(0.1*len(news)):int(0.2*len(news))] # 10%
    train = news[int(0.2*len(news)):]                   # 80%

    write_data(test,'test.txt')
    write_data(valid,'valid.txt')
    write_data(train,'train.txt')

'''
Number of instance in each category

    Overall     train   valid   test
b   442         355     42      45
e   1228        989     124     115
m   323         260     36      27
t   465         363     44      58
'''