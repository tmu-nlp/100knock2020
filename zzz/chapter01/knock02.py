s1 = 'パトカー'
s2 = 'タクシー'
res = ''
for (w1, w2) in zip(s1, s2):
    res = res + w1 + w2
print(res)