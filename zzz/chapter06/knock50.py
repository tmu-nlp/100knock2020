import pandas as pd
from sklearn.model_selection import train_test_split

FILENAME = 'newsCorpora.csv'

if __name__ == '__main__':
    df = pd.read_csv(FILENAME, sep='\t')

    cols = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP']
    df.columns = cols
    # print(df['PUBLISHER'])
    # new_df = df[df['PUBLISHER'] in ['Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']]
    new_df = df[(df['PUBLISHER'] == 'Huffington Post') |
                (df['PUBLISHER'] == 'Businessweek') |
                (df['PUBLISHER'] == 'Contactmusic.com') |
                (df['PUBLISHER'] == 'Daily Mail')]
    new_df = new_df.sample(frac=1)
    print(new_df)
    length = new_df.shape[0]
    train, val, test = new_df[: int(0.8 * length)], new_df[int(0.8 * length): int(0.9 * length)], new_df[int(0.9 * length):]
    print(train.shape, val.shape, test.shape)

    train.to_csv('train.txt', sep='\t')
    val.to_csv('valid.txt', sep='\t')
    test.to_csv('test.txt', sep='\t')