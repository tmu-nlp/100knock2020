str1 = "パトカー"
str2 = "タクシー"
str3 = ""

"""
for i in range(len(str1)+len(str2)):
    if i%2 == 0:
        str3 += str1[int(i/2)]
    else:
        str3 += str2[int(i/2)]

print(str3)
"""

for (a, b) in zip(str1, str2):
    str3 += a + b

print(str3)

#print(zip(str1, str2))
#<zip object at 0x03FE8B20>
#print(list(zip(str1, str2)))
#[('パ', 'タ'), ('ト', 'ク'), ('カ', 'シ'), ('ー', 'ー')]
