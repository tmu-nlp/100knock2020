#64の実行結果を用い，
# 意味的アナロジー（semantic analogy）と文法的アナロジー（syntactic analogy）の正解率を測定せよ．

from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin", binary=True
)

ana_dict = model.evaluate_word_analogies("questions-words.txt")

dict_acc = {}
for i in range(len(ana_dict[1])):
  cor_len = len(ana_dict[1][i]['correct'])
  inc_len = len(ana_dict[1][i]['incorrect'])
  len_all = cor_len + inc_len
  dict_acc[ana_dict[1][i]['section']] = cor_len/len_all

syn = 0
syn_len = 0
sem = 0
sem_len = 0
for key, value in dict_acc.items():
  if len(key.split("-")[0]) == 5:
    syn_len += 1
    syn += value
  else:
    sem_len += 1
    sem += value

print(syn/syn_len)
#0.7124540902991301
print(sem/sem_len)
#0.7087966313593812