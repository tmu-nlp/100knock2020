with open("ai.ja.txt", "r") as s:
    s = s.read().replace("\n", "")
with open("ai.ja1.txt", "r") as f:
    f = f.read().replace("。", "。"+"\n")
with open("ai.ja1.txt", "w") as f1:
    f1.write(f)