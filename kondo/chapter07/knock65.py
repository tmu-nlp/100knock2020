"""
65. アナロジータスクでの正解率Permalink
64の実行結果を用い，意味的アナロジー（semantic analogy）と文法的アナロジー（syntactic analogy）の正解率を測定せよ．
"""

my_ans_file = "64_my_ans_file"
ans_file = "64_ans_file"

if __name__ == "__main__":
    sem = 0
    sem_acc = 0
    syn = 0
    syn_acc = 0
    with open(my_ans_file, encoding="utf-8") as file_my_ans,\
            open(ans_file, encoding="utf-8") as file_ans:
        for x, y in zip(file_my_ans, file_ans):
            y = y.rstrip("\n")
            category, my_ans = x.split("\t")
            my_ans = my_ans.replace("\'", "").replace("(", "").replace(",", "").replace(")", "")
            my_ans, prob = my_ans.split()
            if category.startswith("gram"):
                syn += 1
                if my_ans == y:
                    syn_acc += 1
            else:
                sem += 1
                if my_ans == y:
                    sem_acc += 1
    
    print(f"semantic analogy: {sem_acc/sem}\nsyntactic analogy: {syn_acc/syn}")

