'''
65. アナロジータスクでの正解率Permalink
64の実行結果を用い，意味的アナロジー（semantic analogy）と
文法的アナロジー（syntactic analogy）の正解率を測定せよ．
'''
from pprint import pprint
from tqdm import tqdm

# sem_cnt, sem_predはそれぞれsemanticの時のサンプル数と正解したサンプル数を記録するカウンター
sem_cnt, syn_cnt, sem_pred, syn_pred = 0, 0, 0, 0
with open("./data/questions-words-results.txt") as file:
    bar = tqdm(total=19558)
    for line in f:
        if line[0] == ":":
            cols = line.split()
            # 文字列が'gram'から始まるか？
            #'gram'から始まるものは，syntactic analogy（比較級の形が合っているかとか，，，）
            if cols[1].startswith("gram"):
                flag = False
            # semantic analogy
            else:
                flag = True
        else:
            # semanticの時
            if flag:
                sem_cnt += 1
                cols = line.split()
                # cols[3]:正解, cols[4]:類推した単語
                if cols[3] == cols[4]:
                    sem_pred += 1
                else:
                    pass
            # syntacticの時
            else:
                syn_cnt += 1
                cols = line.split()
                if cols[3] == cols[4]:
                    syn_pred += 1
                else:
                    pass
        bar.update(1)

print(f"semantic analogy: {sem_pred/sem_cnt}")
print(f"syntactic analogy: {syn_pred/syn_cnt}")

'''
semantic analogy: 0.7308602999210734
syntactic analogy: 0.7400468384074942
'''