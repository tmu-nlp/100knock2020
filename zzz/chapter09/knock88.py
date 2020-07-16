from gensim import models
from keras.layers import Embedding, SimpleRNN, Dense, Bidirectional, Flatten
from keras.models import Sequential
import pandas as pd
from zzz.chapter09.knock80 import text_encode
from zzz.chapter09.knock84 import get_embedding_matrix

PATH = '/Users/zz_zhang/勉強会/100本ノック/100knock2020/zzz/chapter06/'

WORD_VEC_PATH = '/Users/zz_zhang/勉強会/100本ノック/100knock2020/zzz/chapter07/'
WORD_VEC_FILE = 'GoogleNews-vectors-negative300.bin'

DW = 300
DH = 110
DROPOUT = [0.0, 0.2]
RUCURRENT_DROPOUT = [0.0, 0.1]

if __name__ == '__main__':
    train = pd.read_csv(PATH + 'train.txt', sep='\t')
    val = pd.read_csv(PATH + 'valid.txt', sep='\t')
    test = pd.read_csv(PATH + 'test.txt', sep='\t')
    train_id, train_label, val_id, val_label, test_id, test_label, vocab, word_index = text_encode(train, val, test,
                                                                                                   type='seq')
    word_vec = models.KeyedVectors.load_word2vec_format(WORD_VEC_PATH + WORD_VEC_FILE, binary=True)

    embedding_matrix = get_embedding_matrix(word_vec, word_index)
    # print(embedding_matrix.shape)

    for dh in range(50, DH + 1, 20):
        for dr in DROPOUT:
            for rdr in RUCURRENT_DROPOUT:
                model = Sequential()
                model.add(Embedding(len(vocab) + 1, DW, weights=[embedding_matrix], trainable=False))
                model.add(Bidirectional(SimpleRNN(dh, dropout=dr, recurrent_dropout=rdr, return_sequences=True),
                                        merge_mode='concat'))
                model.add(Bidirectional(SimpleRNN(int(dh / 2), dropout=dr, recurrent_dropout=rdr),
                                        merge_mode='concat'))

                model.add(Flatten())
                model.add(Dense(4, activation='softmax'))

                # model.summary()

                model.compile(
                    loss='categorical_crossentropy',
                    optimizer='sgd',
                    metrics=["accuracy"],
                )

                model.fit(train_id, train_label, validation_data=(val_id, val_label), epochs=20, verbose=0)

                score = model.evaluate(test_id, test_label)
                print('=' * 20)
                print('DH={}, DR={}, RDR={}'.format(dh, dr, rdr))
                print(score)

'''
epoch = 5
====================
DH=50, DR=0.0, RDR=0.0
[0.49594953656196594, 0.8315678238868713]
====================
DH=50, DR=0.0, RDR=0.1
[0.5484288334846497, 0.804025411605835]
====================
DH=50, DR=0.2, RDR=0.0
[0.49440765380859375, 0.8336864113807678]
====================
DH=50, DR=0.2, RDR=0.1
[0.5417965650558472, 0.8008474707603455]
====================
DH=70, DR=0.0, RDR=0.0
[0.6096945405006409, 0.8019067645072937]
====================
DH=70, DR=0.0, RDR=0.1
[0.4559648931026459, 0.8516949415206909]
====================
DH=70, DR=0.2, RDR=0.0
[0.47044968605041504, 0.8379237055778503]
====================
DH=70, DR=0.2, RDR=0.1
[0.5287325978279114, 0.8146186470985413]
====================
DH=90, DR=0.0, RDR=0.0
[0.4798346757888794, 0.8358050584793091]
====================
DH=90, DR=0.0, RDR=0.1
[0.49407362937927246, 0.8262711763381958]
====================
DH=90, DR=0.2, RDR=0.0
[0.4806162714958191, 0.8368644118309021]
====================
DH=90, DR=0.2, RDR=0.1
[0.5167549848556519, 0.8305084705352783]
====================
DH=110, DR=0.0, RDR=0.0
[0.46122780442237854, 0.8474576473236084]
====================
DH=110, DR=0.0, RDR=0.1
[0.4703715741634369, 0.8463982939720154]
====================
DH=110, DR=0.2, RDR=0.0
[0.47877809405326843, 0.8421609997749329]
====================
DH=110, DR=0.2, RDR=0.1
[0.47122442722320557, 0.8336864113807678]

epoch = 20
====================
DH=110, DR=0.2, RDR=0.1
[0.4522106945514679, 0.8591101765632629]
'''
