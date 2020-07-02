import pandas as pd

from knock60 import load_model

if __name__ == '__main__':
    model = load_model()
    df = pd.read_csv('./drive/My Drive/100knock_chap7/wordsim353/combined.csv')
    sims = []
    for i in range(len(df)):
        line = df.iloc[i]
        sims.append(model.similarity(line['Word 1'], line['Word 2']))
    df['Word2Vec'] = sims
    df_corr = df[['Human (mean)', 'Word2Vec']].corr(method='spearman')
    print(df_corr)

"""
	        Human (mean)	Word2Vec
Human (mean)	1.000000	0.700017
Word2Vec	0.700017	1.000000
"""
