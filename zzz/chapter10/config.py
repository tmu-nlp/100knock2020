path = '/Users/zz_zhang/勉強会/100本ノック/100knock2020/zzz/chapter10/kftt-data-1.0/data/orig'
model_path = 'model.pkl'
en_file = 'kyoto-{0}.en'
ja_file = 'kyoto-{0}.ja'
en_file2 = 'kyoto-train.cln.en'
ja_file2 = 'kyoto-train.cln.ja'

en_pretrained_model = 'bert-base-uncased'
ja_pretrained_model = 'bert-base-japanese'
# for subword tokenizer
# en_pretrained_model = 'xlnet-base-cased'

max_len = 1024
batch_size = 10
embedding_size = 300
drop_prob = 0.1
data_workers = 0
ntokens = 30000  # the size of vocabulary
emsize = 200  # embedding dimension
nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # the number of heads in the multiheadattention models
dropout = 0.2  # the dropout value